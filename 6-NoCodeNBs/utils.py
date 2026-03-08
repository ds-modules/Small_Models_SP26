"""
utils.py — Helper functions for LLM Context Management notebook.

All heavy logic lives here so the notebook stays clean and readable.
Import what you need:  from utils import build_system_message, ChatAssistant, ...

Modules:
  - System-Prompt Builders     : build_system_message, build_course_prompt
  - Token Utilities            : count_tokens, estimated_wait
  - History Compression        : summarize_full_text, extract_key_entities, compress_semantic
  - Profile Change Detection   : detect_profile_changes
  - Chat Assistant             : ChatAssistant
  - Context Window Visualizer  : visualize_context_window
  - RAG Helpers                : retrieve, rag_chat
  - Chat Bubble Renderer       : bubble, token_pill
"""

import re
import json
from IPython.display import display, HTML


# ═══════════════════════════════════════════════════════════════════════
#  System-Prompt Builders
# ═══════════════════════════════════════════════════════════════════════

def build_system_message(profile, project_profile=None):
    """
    Build a personalised system prompt from a user profile dict.

    The system prompt tells the LLM who it's talking to and how to behave.
    By constructing it dynamically from the profile, we get a different
    prompt for every user — without writing separate prompts by hand.

    Args:
        profile (dict): User info — name, expertise, course, style_preferences, etc.
        project_profile (dict, optional): Info about the user's current project.

    Returns:
        str: A formatted system prompt string ready to pass to the LLM.
    """
    name      = profile.get("name", "the user")
    expertise = profile.get("expertise", "intermediate")
    project   = profile.get("current_project", "")
    style     = profile.get("style_preferences", [])

    lines = [
        "You are a helpful AI assistant.",
        "",
        "## About the User",
        f"- Name: {name}",
        f"- Skill level: {expertise}",
    ]

    if project:
        lines.append(f"- Current project: {project}")

    if style:
        lines += ["", "## Response Style"]
        for s in style:
            lines.append(f"- {s}")

    # Recommend the logical next course in Berkeley's data science curriculum:
    # Data 8 → Data 100 → Data 102 → CS 189
    course = profile.get("course", "").strip().lower()
    NEXT_COURSE = {
        "data 8":   ("Data 100", "to strengthen Python and pandas skills"),
        "data 100": ("Data 102", "to go deeper into inference and decisions"),
        "data 102": ("CS 189",   "if you want to study ML theory rigorously"),
    }
    next_rec = NEXT_COURSE.get(course)
    if next_rec:
        lines += [
            "",
            "## Course Recommendation",
            f"- When relevant, suggest the student considers {next_rec[0]} next — {next_rec[1]}.",
        ]

    # Adjust tone and depth based on expertise level
    if expertise.lower() in ["beginner", "new to coding"]:
        lines += [
            "",
            "## Important",
            "- Use simple language, avoid jargon",
            "- Always include a short code example",
            "- Explain each step clearly",
        ]
    elif expertise.lower() in ["expert", "senior", "advanced"]:
        lines += [
            "",
            "## Important",
            "- Be concise and technical",
            "- Skip basic explanations",
            "- Focus on edge cases and best practices",
        ]

    if project_profile:
        proj_name  = project_profile.get("name", "")
        proj_desc  = project_profile.get("description", "")
        proj_goal  = project_profile.get("current_goal", "")
        proj_tools = project_profile.get("tools", [])
        lines += ["", "## Current Project"]
        if proj_name:
            lines.append(f"- Project: {proj_name}" + (f" — {proj_desc}" if proj_desc else ""))
        if proj_goal:
            lines.append(f"- Goal: {proj_goal}")
        if proj_tools:
            lines.append(f"- Tools: {', '.join(proj_tools)}")

    return "\n".join(lines)


def build_course_prompt(course):
    """
    Return a tailored system prompt for Data 8 or Data 100 students.

    Data 8 and Data 100 use different libraries (datascience vs pandas),
    so we need separate prompts to avoid confusing students with the wrong syntax.

    Args:
        course (str): Either "Data 8" or "Data 100".

    Returns:
        str: A course-specific system prompt string.
    """
    if course == "Data 8":
        lib_guidance = (
            "The student is in Data 8 (Foundations of Data Science) at UC Berkeley. "
            "This course uses ONLY Python. "
            "ALWAYS use the `datascience` package. The correct syntax is: "
            "from datascience import Table; t = Table.read_table('file.csv'). "
            "Never mention R, pandas, or any other language or library."
        )
    else:
        lib_guidance = (
            "The student is in Data 100 (Principles and Techniques of Data Science). "
            "Always recommend `pandas` (pd.DataFrame, pd.read_csv, etc.). "
            "You may use technical terminology; the student knows Python."
        )
    return (
        "You are a helpful TA for UC Berkeley's data science program.\n"
        f"{lib_guidance}\n"
        "Keep answers concise and include a short code snippet."
    )


# ═══════════════════════════════════════════════════════════════════════
#  Token Utilities
# ═══════════════════════════════════════════════════════════════════════

def count_tokens(messages, model=None):
    """
    Count the total number of tokens across a list of messages.

    Token counting matters because the LLM has a fixed context window —
    if we exceed it, older messages get cut off and the model loses memory.

    Args:
        messages (list): List of {"role": ..., "content": ...} dicts.
        model: A loaded llama_cpp model. If None, falls back to a heuristic.

    Returns:
        int: Estimated total token count.
    """
    if model is not None:
        # Use the model's actual tokenizer for an exact count
        return sum(len(model.tokenize(m["content"].encode("utf-8"))) for m in messages)
    # Fallback heuristic: ~4 characters per token (works reasonably well for English)
    return sum(len(m.get("content", "")) // 4 for m in messages)


def estimated_wait(tokens, speed_tps=25):
    """
    Estimate how long the model will take to process a given number of tokens.

    Useful for setting user expectations before a slow generation call.

    Args:
        tokens (int): Number of tokens to process.
        speed_tps (int): Model speed in tokens per second.
                         25 tps is a conservative estimate for llama-cpp on a shared CPU.

    Returns:
        float: Estimated wait time in seconds.
    """
    return tokens / speed_tps


# ═══════════════════════════════════════════════════════════════════════
#  History Compression
# ═══════════════════════════════════════════════════════════════════════

def _conversation_text(messages):
    """Format a messages list into a plain-text transcript for summarisation prompts."""
    return "".join(
        f"[{'User' if m['role'] == 'user' else 'AI'}]: {m['content']}\n\n"
        for m in messages
    )


def summarize_full_text(messages_to_summarize, model):
    """
    Compress an entire conversation into 1-2 sentences of prose.

    When the conversation history grows too long, we can't keep all of it
    in the context window. This function distills it into a short summary
    that preserves the main thread without consuming too many tokens.

    Args:
        messages_to_summarize (list): The portion of history to compress.
        model: A loaded llama_cpp model.

    Returns:
        str: A 1-2 sentence summary of the conversation.
    """
    prompt = f"Summarize this conversation in 1-2 sentences only:\n\n{_conversation_text(messages_to_summarize)}"
    resp = model.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=60,   # 1-2 sentences fit comfortably within 60 tokens
        temperature=0.7,
    )
    return resp["choices"][0]["message"]["content"]


def extract_key_entities(messages_to_summarize, model):
    """
    Extract important facts and decisions from a conversation as bullet points.

    Unlike a prose summary, bullet points make it easy for the LLM to
    quickly reference specific facts (e.g. "User is working in Python 3.11").

    Args:
        messages_to_summarize (list): The portion of history to compress.
        model: A loaded llama_cpp model.

    Returns:
        str: 3-4 bullet points of key facts.
    """
    prompt = (
        f"Extract key facts from this conversation as 3-4 bullet points only:\n\n"
        f"{_conversation_text(messages_to_summarize)}\nFormat as: - Fact\n- Fact"
    )
    resp = model.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=80,   # enough headroom for 3-4 short bullet points
        temperature=0.3, # low temperature = more deterministic, factual output
    )
    return resp["choices"][0]["message"]["content"]


def compress_semantic(messages_to_summarize, model):
    """
    Compress a conversation using both a prose summary and key bullet points.

    This is the best-of-both-worlds strategy: the summary gives narrative
    continuity, while the bullets preserve specific facts. Together they
    use fewer tokens than keeping the raw history.

    Args:
        messages_to_summarize (list): The portion of history to compress.
        model: A loaded llama_cpp model.

    Returns:
        str: A one-sentence summary followed by 2-3 key fact bullets.
    """
    prompt = (
        f"Compress this conversation:\n\n{_conversation_text(messages_to_summarize)}\n"
        "Provide:\n1. One-sentence summary\n2. Key facts (2-3 bullets only)"
    )
    resp = model.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=80,   # tight limit forces the model to be concise
        temperature=0.5,
    )
    return resp["choices"][0]["message"]["content"]


# ═══════════════════════════════════════════════════════════════════════
#  Profile Change Detection
# ═══════════════════════════════════════════════════════════════════════

def detect_profile_changes(user_message, user_profile, project_profile, model):
    """
    Ask the model whether the user's latest message should update their profile.

    Users sometimes reveal new information mid-conversation (e.g. "I actually
    have 5 years of experience"). This function detects those signals and
    returns the fields that should be updated — so the assistant can adapt
    its tone immediately without the user having to re-configure anything.

    Args:
        user_message (str): The latest message from the user.
        user_profile (dict): Current user profile.
        project_profile (dict): Current project profile.
        model: A loaded llama_cpp model.

    Returns:
        dict: Fields to update, with optional 'conflict' key if the new info
              contradicts the existing profile. Returns {} if no changes needed.
    """
    detection_prompt = f"""A user said: "{user_message}"

Their current profile:
{json.dumps({'user_profile': user_profile, 'project_profile': project_profile}, indent=2)}

Does this message contradict or update any profile field?

Rules:
- If the user claims more experience than their current expertise, update expertise to "professional"
- If no change needed, return {{}}
- Always return ONLY valid JSON, nothing else

Examples:
- User says "I have 5 years of experience" + current expertise is "beginner" → {{"conflict": true, "user_profile": {{"expertise": "professional"}}}}
- User says "I switched to using R" → {{"user_profile": {{"tools": "R"}}}}
- User says "Thanks!" → {{}}

Your response (JSON only):"""

    resp = model.create_chat_completion(
        messages=[{"role": "user", "content": detection_prompt}],
        max_tokens=100,  # JSON responses are short; 100 tokens is generous
        temperature=0.1, # near-zero temperature for consistent, structured output
    )
    raw = resp["choices"][0]["message"]["content"].strip()
    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # The model failed to return valid JSON — fall back to keyword matching
        # as a safety net so profile updates are never silently dropped
        experience_keywords = [
            "years of experience", "years experience", "professionally",
            "i'm an expert", "i am an expert", "senior developer",
            "professional developer",
        ]
        if any(kw in user_message.lower() for kw in experience_keywords):
            return {"conflict": True, "user_profile": {"expertise": "professional"}}
        return {}

    return result


# ═══════════════════════════════════════════════════════════════════════
#  Chat Assistant
# ═══════════════════════════════════════════════════════════════════════

class ChatAssistant:
    """
    A complete chat system that combines all the techniques from this notebook.

    Features:
    - Dynamic system prompt built from user & project profiles
    - Sliding window history (keeps only the last `max_turns` turns in context)
    - Semantic compression (older turns are summarised, not discarded)
    - Automatic profile updates when the user reveals new information
    - Conflict detection when new info contradicts the existing profile

    Args:
        user_profile (dict): Initial user profile (name, expertise, etc.).
        project_profile (dict): Initial project profile (name, tools, goal, etc.).
        model: A loaded llama_cpp model.
        max_turns (int): How many recent turns to keep in full before compressing.
                         Default 4 = 8 messages (4 user + 4 assistant).
        chunk_size (int): How many turns to compress at a time when the window overflows.
                          Default 2 keeps compression fast and cheap.
    """

    def __init__(self, user_profile, project_profile, model, max_turns=4, chunk_size=2):
        self.user_profile       = user_profile.copy()
        self.project_profile    = project_profile.copy()
        self.model              = model
        self.max_turns          = max_turns
        self.chunk_size         = chunk_size
        self.recent_history     = []   # full-text turns within the sliding window
        self.summary            = None # compressed summary of older turns
        self.total_turns        = 0
        self.conflicts_detected = []   # log of profile conflicts for show_state()

    def _build_system_message(self):
        """Construct the system prompt from the current profile state."""
        u, p = self.user_profile, self.project_profile
        lines = [
            "You are a helpful AI assistant.",
            "",
            "## About the User",
            f"- Name: {u.get('name', 'the user')}",
            f"- Skill level: {u.get('expertise', 'intermediate')}",
            f"- Language: {u.get('language', 'English')}",
        ]
        if p.get("name"):
            lines.append(f"- Project: {p['name']} — {p.get('description', '')}")
        if p.get("tools"):
            lines.append(f"- Tools: {', '.join(p['tools'])}")
        if p.get("current_goal"):
            lines.append(f"- Current goal: {p['current_goal']}")

        # Adjust response style based on expertise level
        exp = u.get("expertise", "").lower()
        if exp in ["beginner", "new to coding", "python beginner"]:
            lines += ["", "## Style",
                      "- Use simple language, avoid jargon",
                      "- Always include a short code example",
                      "- Explain each step clearly"]
        elif exp in ["expert", "senior", "advanced", "professional"]:
            lines += ["", "## Style",
                      "- Be concise and technical",
                      "- Skip basic explanations",
                      "- Focus on edge cases and best practices"]

        for pref in u.get("style_preferences", []):
            lines.append(f"- {pref}")

        # Inject compressed summary of older turns so the model retains
        # context from before the sliding window
        if self.summary:
            lines += ["", "## Conversation Summary", self.summary]

        return "\n".join(lines)

    def _build_messages(self, user_message):
        """Assemble the full messages list: system prompt + history + new message."""
        msgs = [{"role": "system", "content": self._build_system_message()}]
        msgs.extend(self.recent_history)
        msgs.append({"role": "user", "content": user_message})
        return msgs

    def _maybe_compress(self):
        """
        Compress the oldest chunk of history if the window is full.

        Returns True if compression happened (useful for logging).
        """
        if len(self.recent_history) > self.max_turns * 2:
            to_compress = self.recent_history[: self.chunk_size * 2]
            self.recent_history = self.recent_history[self.chunk_size * 2:]
            self.summary = compress_semantic(to_compress, self.model)
            return True
        return False

    def chat(self, user_message, show_workflow=True):
        """
        Send a message and get a response from the assistant.

        Under the hood this:
          1. Builds the messages list (system + history + new message)
          2. Calls the model
          3. Appends the exchange to history
          4. Checks for profile updates
          5. Compresses history if the window is full

        Args:
            user_message (str): The user's input.
            show_workflow (bool): If True, prints a step-by-step log of what's happening.

        Returns:
            str: The assistant's reply.
        """
        if show_workflow:
            print(f"\n{'='*60}")
            print(f"Turn {self.total_turns + 1}")
            print(f"{'='*60}")
            print(f'👤 User: "{user_message}"')

        msgs     = self._build_messages(user_message)
        resp     = self.model.create_chat_completion(
            messages=msgs,
            max_tokens=150,  # typical tutoring response fits within 150 tokens
            temperature=0.7,
        )
        ai_reply = resp["choices"][0]["message"]["content"].strip()

        self.recent_history.append({"role": "user",      "content": user_message})
        self.recent_history.append({"role": "assistant", "content": ai_reply})

        # Check if the user's message reveals new info that should update the profile
        updates = detect_profile_changes(
            user_message, self.user_profile, self.project_profile, self.model
        )
        if updates:
            is_conflict = updates.pop("conflict", False)
            if is_conflict:
                self.conflicts_detected.append({
                    "turn":       self.total_turns,
                    "message":    user_message,
                    "resolution": updates,
                })
                if show_workflow:
                    print(f"\n⚠️  MEMORY CONFLICT DETECTED!")
            if "user_profile" in updates:
                self.user_profile.update(updates["user_profile"])
                if show_workflow:
                    print(f"♻️  Profile updated: {updates['user_profile']}")
                    print(f"    Next response will adjust accordingly.")
            if "project_profile" in updates:
                self.project_profile.update(updates["project_profile"])

        if self._maybe_compress() and show_workflow:
            print(f"📦 History compressed.")

        self.total_turns += 1

        if show_workflow:
            print(f"\n🤖 AI: {ai_reply}")

        return ai_reply

    def show_state(self):
        """Render a visual dashboard of the current session state as HTML."""
        profile_rows = "".join(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
                    padding:8px 12px;border-bottom:1px solid #1e293b;
                    background:{'#0d1420' if i%2==0 else '#111827'}">
          <span style="color:#7ea8c9;font-size:0.75em;font-family:'IBM Plex Mono',monospace">{k}</span>
          <span style="color:#e2e8f0;font-size:0.75em;font-family:'IBM Plex Mono',monospace">{v}</span>
        </div>""" for i, (k, v) in enumerate(self.user_profile.items()))

        conflict_rows = ""
        for i, c in enumerate(self.conflicts_detected):
            res = c["resolution"]
            changes = []
            if isinstance(res, dict):
                for section, fields in res.items():
                    if isinstance(fields, dict):
                        for k, v in fields.items():
                            changes.append(f"{k} → <span style='color:#34d399'>{v}</span>")
            conflict_rows += f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:8px 12px;border-bottom:1px solid #1e293b;
                        background:{'#0d1420' if i%2==0 else '#111827'}">
              <span style="color:#7ea8c9;font-size:0.75em;font-family:'IBM Plex Mono',monospace">Turn {c['turn']}</span>
              <span style="color:#e2e8f0;font-size:0.75em;font-family:'IBM Plex Mono',monospace">{", ".join(changes) if changes else str(res)}</span>
            </div>"""

        conflicts_section = f"""
        <div style="margin-top:12px">
          <div style="font-size:0.65em;color:#f87171;text-transform:uppercase;
                      letter-spacing:0.1em;margin-bottom:6px">
            ⚠️ Conflicts Resolved ({len(self.conflicts_detected)})
          </div>
          <div style="background:#0d1420;border:1px solid #f8717133;border-radius:8px;overflow:hidden">
            {conflict_rows}
          </div>
        </div>""" if self.conflicts_detected else ""

        stat_cards = "".join(f"""
        <div style="background:#0d1420;border:1px solid #1e293b;border-radius:8px;
                    padding:10px;text-align:center">
          <div style="font-size:0.63em;color:#7ea8c9;margin-bottom:4px">{label}</div>
          <div style="font-size:1.05em;font-weight:700;color:{color}">{val}</div>
        </div>""" for label, val, color in [
            ("Total turns",  self.total_turns,                        "#f0f6ff"),
            ("History",      f"{len(self.recent_history)//2} turns",  "#f0f6ff"),
            ("Summary",      "Yes" if self.summary else "No",         "#34d399" if self.summary else "#94b8d4"),
            ("Conflicts",    len(self.conflicts_detected),             "#f87171" if self.conflicts_detected else "#94b8d4"),
        ])

        display(HTML(f"""
        <div style="font-family:'IBM Plex Mono','Fira Code',monospace;
                    background:#080c12;border-radius:12px;padding:18px 20px;
                    color:#e2e8f0;margin-top:10px">
          <div style="font-size:0.63em;color:#7ea8c9;text-transform:uppercase;
                      letter-spacing:0.15em;margin-bottom:4px">Session Summary</div>
          <h3 style="color:#f0f6ff;margin:0 0 14px;font-size:1.0em">📋 Current State</h3>
          <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:14px">
            {stat_cards}
          </div>
          <div style="background:#0d1420;border:1px solid #1e293b;border-radius:8px;overflow:hidden">
            <div style="padding:8px 12px;background:#0a1628;font-size:0.65em;
                        color:#7ea8c9;text-transform:uppercase;letter-spacing:0.1em">
              User Profile
            </div>
            {profile_rows}
          </div>
          {conflicts_section}
        </div>
        """))


# ═══════════════════════════════════════════════════════════════════════
#  Context Window Visualizer
# ═══════════════════════════════════════════════════════════════════════

def visualize_context_window(messages, model, n_ctx=4096, title="Context Window Snapshot"):
    """
    Render a stacked bar chart showing how each message fills the context window.

    This makes the abstract idea of a "context window" tangible — students can
    see exactly how many tokens each role consumes and how close we are to the limit.

    Args:
        messages (list): The full messages list passed to the model.
        model: A loaded llama_cpp model (used for exact token counts).
        n_ctx (int): The model's maximum context window size in tokens.
        title (str): Title displayed above the visualisation.
    """
    COLORS = {
        "system":    ("#f9e2af", "System Prompt"),
        "user":      ("#89b4fa", "User"),
        "assistant": ("#a6e3a1", "Assistant"),
    }
    SPEED = 25  # conservative tokens/sec estimate for llama-cpp on a shared CPU

    segments = []
    for m in messages:
        toks = len(model.tokenize(m["content"].encode("utf-8")))
        segments.append({
            "role":    m["role"],
            "tokens":  toks,
            "preview": m["content"][:60].replace("<", "&lt;").replace(">", "&gt;"),
        })

    total    = sum(s["tokens"] for s in segments)
    used_pct = total / n_ctx * 100

    bar_segs = ""
    for s in segments:
        color = COLORS.get(s["role"], ("#cdd6f4", s["role"]))[0]
        w     = max(0.5, s["tokens"] / n_ctx * 100)
        bar_segs += (
            f'<div title="{s["role"]}: &quot;{s["preview"]}&quot; ({s["tokens"]} tok)" '
            f'style="width:{w}%;background:{color};height:100%;'
            f'display:inline-block;border-right:1px solid #1e1e2e"></div>'
        )
    remain_pct = max(0, 100 - used_pct)
    bar_segs += (
        f'<div style="width:{remain_pct}%;background:#313244;height:100%;'
        f'display:inline-block;opacity:0.4"></div>'
    )

    role_counts = {}
    for s in segments:
        role_counts[s["role"]] = role_counts.get(s["role"], 0) + s["tokens"]
    legend = ""
    for role, toks in role_counts.items():
        color, label = COLORS.get(role, ("#cdd6f4", role))
        legend += (
            f'<span style="display:inline-flex;align-items:center;gap:5px;margin-right:14px">'
            f'<span style="width:12px;height:12px;background:{color};border-radius:2px;display:inline-block"></span>'
            f'<span style="color:#cdd6f4;font-size:0.82em">{label}: {toks} tok</span></span>'
        )

    rows = ""
    for s in segments:
        color, label = COLORS.get(s["role"], ("#cdd6f4", s["role"]))
        bar_w = max(2, int(s["tokens"] / total * 280)) if total else 2
        rows += (
            f'<tr style="border-bottom:1px solid #313244">'
            f'<td style="padding:5px 10px;color:{color};font-weight:bold;width:90px;font-size:0.82em">{label}</td>'
            f'<td style="padding:5px 10px;color:#a6adc8;font-size:0.78em;max-width:280px;'
            f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{s["preview"]}...</td>'
            f'<td style="padding:5px 10px;text-align:right">'
            f'<div style="display:inline-block;width:{bar_w}px;height:8px;background:{color};'
            f'border-radius:3px;vertical-align:middle;margin-right:6px"></div>'
            f'<span style="color:{color};font-size:0.82em">{s["tokens"]}</span></td></tr>'
        )

    pct_str    = f"{used_pct:.1f}"
    wait_str   = f"{total / SPEED:.1f}"
    remain_tok = n_ctx - total
    fill_color = "#f38ba8" if used_pct > 75 else "#f9e2af" if used_pct > 40 else "#a6e3a1"

    html = (
        f'<div style="background:#1e1e2e;border-radius:12px;padding:18px 20px;margin:12px 0">'
        f'<h4 style="color:#cdd6f4;margin:0 0 14px 0">{title}</h4>'
        f'<div style="width:100%;height:22px;border-radius:6px;overflow:hidden;'
        f'border:1px solid #45475a;margin-bottom:8px">{bar_segs}</div>'
        f'<div style="margin-bottom:12px">{legend}'
        f'<span style="color:#585b70;font-size:0.82em">| Empty: {remain_tok} tok remaining</span></div>'
        f'<div style="display:flex;gap:10px;margin-bottom:14px">'
        f'<div style="flex:1;background:#313244;border-radius:8px;padding:10px 14px;text-align:center">'
        f'<div style="color:#a6adc8;font-size:0.75em">Total tokens used</div>'
        f'<div style="color:#cdd6f4;font-weight:bold;font-size:1.4em">{total}</div></div>'
        f'<div style="flex:1;background:#313244;border-radius:8px;padding:10px 14px;text-align:center">'
        f'<div style="color:#a6adc8;font-size:0.75em">Context filled</div>'
        f'<div style="color:{fill_color};font-weight:bold;font-size:1.4em">{pct_str}%</div></div>'
        f'<div style="flex:1;background:#313244;border-radius:8px;padding:10px 14px;text-align:center">'
        f'<div style="color:#a6adc8;font-size:0.75em">Est. generation wait</div>'
        f'<div style="color:#f9e2af;font-weight:bold;font-size:1.4em">~{wait_str}s</div></div>'
        f'<div style="flex:1;background:#313244;border-radius:8px;padding:10px 14px;text-align:center">'
        f'<div style="color:#a6adc8;font-size:0.75em">Messages</div>'
        f'<div style="color:#cdd6f4;font-weight:bold;font-size:1.4em">{len(messages)}</div></div>'
        f'</div>'
        f'<table style="width:100%;border-collapse:collapse">'
        f'<tr style="color:#585b70;font-size:0.78em;border-bottom:1px solid #45475a">'
        f'<th style="text-align:left;padding:4px 10px">Role</th>'
        f'<th style="text-align:left;padding:4px 10px">Content preview</th>'
        f'<th style="text-align:right;padding:4px 10px">Tokens</th></tr>'
        f'{rows}</table></div>'
    )
    display(HTML(html))


# ═══════════════════════════════════════════════════════════════════════
#  RAG Helpers
# ═══════════════════════════════════════════════════════════════════════

def retrieve(query, knowledge_base, top_k=2):
    """
    Find the most relevant documents from a knowledge base for a given query.

    Uses keyword overlap (bag-of-words) to score relevance — simple but effective
    for small, curated knowledge bases like course FAQs.

    ⚠️ Limitation: this approach matches on exact words, not meaning.
    For example, "car" and "vehicle" would NOT match. For production use,
    replace with vector similarity (e.g. sentence-transformers + cosine similarity).

    Args:
        query (str): The user's question.
        knowledge_base (list): List of {"title": ..., "content": ...} dicts.
        top_k (int): Maximum number of documents to return.

    Returns:
        list: Up to top_k most relevant documents, sorted by relevance score.
    """
    query_words = set(query.lower().split())
    scored = []
    for doc in knowledge_base:
        doc_words = set((doc["title"] + " " + doc["content"]).lower().split())
        scored.append((len(query_words & doc_words), doc))
    scored.sort(key=lambda x: -x[0])
    return [doc for score, doc in scored[:top_k] if score > 0]


def rag_chat(user_question, model, knowledge_base, system_base=None):
    """
    Answer a question by first retrieving relevant knowledge, then generating a response.

    This is the core RAG (Retrieval-Augmented Generation) pattern:
      1. Retrieve the most relevant docs from the knowledge base
      2. Inject them into the system prompt as context
      3. Let the model answer using that context

    Without RAG, the model can only use what it learned during training.
    With RAG, it can answer questions about course-specific content it has never seen.

    Args:
        user_question (str): The student's question.
        model: A loaded llama_cpp model.
        knowledge_base (list): List of {"title": ..., "content": ...} dicts.
        system_base (str, optional): Base system prompt to prepend context to.

    Returns:
        tuple: (reply: str, retrieved_docs: list, token_count: int)
    """
    if system_base is None:
        system_base = "You are a helpful AI assistant for Berkeley students."

    docs = retrieve(user_question, knowledge_base)

    if docs:
        context_block = "\n\n## Retrieved Knowledge\n"
        for d in docs:
            context_block += f"\n### {d['title']}\n{d['content']}\n"
        system_with_context = system_base + context_block
    else:
        system_with_context = system_base

    msgs = [
        {"role": "system", "content": system_with_context},
        {"role": "user",   "content": user_question},
    ]
    tok_count = count_tokens(msgs, model)
    resp      = model.create_chat_completion(
        messages=msgs,
        max_tokens=180,  # slightly more generous than tutoring responses
                         # to allow for multi-step explanations
        temperature=0.7,
    )
    reply     = resp["choices"][0]["message"]["content"].strip()
    return reply, docs, tok_count


# ═══════════════════════════════════════════════════════════════════════
#  Chat Bubble Renderer  (used in Part 1 interactive)
# ═══════════════════════════════════════════════════════════════════════

def bubble(role, content, extra_class=""):
    """
    Render a single chat message as a styled HTML bubble.

    Each role gets a distinct colour and alignment so students can
    visually distinguish who said what in the conversation.

    Args:
        role (str): One of "system", "user", or "assistant".
        content (str): The message text to display.
        extra_class (str): Optional CSS class for additional styling hooks.

    Returns:
        str: An HTML string containing the rendered bubble.
    """
    cfg = {
        "system":    ("#f9e2af", "#f9e2af18", "left",  "⚙️ system"),
        "user":      ("#89b4fa", "#89b4fa18", "right", "👤 user"),
        "assistant": ("#a6e3a1", "#a6e3a118", "left",  "🤖 assistant"),
    }
    color, bg, side, label = cfg.get(role, ("#cdd6f4", "#cdd6f418", "left", role))
    align  = "flex-end"  if side == "right" else "flex-start"
    radius = "18px 18px 4px 18px" if side == "right" else "18px 18px 18px 4px"
    return f"""
    <div class="bubble-wrap {extra_class}"
         style="display:flex;justify-content:{align};margin:5px 0">
      <div style="max-width:75%">
        <div style="font-size:0.72em;color:{color};margin-bottom:3px;
                    text-align:{'right' if side=='right' else 'left'}">{label}</div>
        <div style="background:{bg};border:1px solid {color}44;
                    border-radius:{radius};padding:9px 14px;
                    color:#cdd6f4;font-size:0.87em;line-height:1.6">{content}</div>
      </div>
    </div>"""


def token_pill(msgs, model):
    """
    Render a small token-count badge for a list of messages.

    Colour-coded by urgency:
      - Green  : under 150 tokens (plenty of room)
      - Yellow : 150–500 tokens  (getting full)
      - Red    : over 500 tokens  (approaching limit)

    Args:
        msgs (list): List of {"role": ..., "content": ...} dicts.
        model: A loaded llama_cpp model. Falls back to heuristic if unavailable.

    Returns:
        str: An HTML string containing the rendered badge.
    """
    try:
        n = sum(len(model.tokenize(m["content"].encode())) for m in msgs)
    except Exception:
        # Fall back to the ~4 chars/token heuristic if tokenizer is unavailable
        n = sum(len(m.get("content", "")) // 4 for m in msgs)
    cls = "crit" if n > 500 else "warn" if n > 150 else ""
    return f'<span class="token-pill {cls}">🪙 {n} tokens</span>'
