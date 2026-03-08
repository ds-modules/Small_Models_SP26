import os
import streamlit as st

APP_DIRECTORY = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

# ============================================================
# EASY-TO-EDIT SETTINGS
# ============================================================
DEFAULT_DATAHUB_MODEL_PATH = "/home/jovyan/shared/qwen2-1_5b-instruct-q4_0.gguf"

DEFAULT_DOTENV_PATHS = [
    os.path.join(APP_DIRECTORY, ".env"),
    "/home/jovyan/shared/.env",
]

# The app first tries to load a local reranker from this folder.
# If the folder does not contain the model files, it falls back to Hugging Face.
DEFAULT_RERANKER_PATH = os.path.join(APP_DIRECTORY, "ms-marco-MiniLM-L6-v2")
DEFAULT_RERANKER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"

DEFAULT_EMBEDDING_MODELS = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
]

# OpenAI: most capable + efficient
DEFAULT_OPENAI_MODELS = [
    "gpt-5-mini",
    "gpt-5",
]

# Anthropic: most capable + efficient
DEFAULT_CLAUDE_MODELS = [
    "claude-sonnet-4-20250514",
    "claude-3-5-haiku-20241022",
]

DEFAULT_EXAMPLE_QUESTION = "When can students apply for financial aid?"
DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_TOP_K = 3
DEFAULT_RERANK_CANDIDATE_K = 8

# The local model should load only when needed.
PRELOAD_LOCAL_MODEL_ON_START = False
DEFAULT_LOCAL_CONTEXT_SIZE = 2048
DEFAULT_LOCAL_THREAD_COUNT = 4
DEFAULT_MAX_GENERATION_TOKENS = 160
DEFAULT_GENERATION_TEMPERATURE = 0.2

# Retrieval-first workflow defaults to no generation.
DEFAULT_LLM_MODE_OPTIONS = ["None", "OpenAI", "Claude", "Local"]
DEFAULT_LLM_MODE_INDEX = 0

missing_required_packages = []
missing_optional_packages = []

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    missing_required_packages.append("sentence-transformers")

try:
    from sentence_transformers import CrossEncoder
    cross_encoder_support = True
except Exception:
    cross_encoder_support = False
    missing_optional_packages.append("sentence-transformers (CrossEncoder)")

try:
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    missing_required_packages.append("scikit-learn")

try:
    import matplotlib.pyplot as plt
    plot_support = True
except Exception:
    plot_support = False
    missing_optional_packages.append("matplotlib")

try:
    from pypdf import PdfReader
    pdf_support = True
except Exception:
    pdf_support = False
    missing_optional_packages.append("pypdf")

try:
    from docx import Document
    docx_support = True
except Exception:
    docx_support = False
    missing_optional_packages.append("python-docx")

try:
    from dotenv import load_dotenv
    dotenv_support = True
except Exception:
    dotenv_support = False
    missing_optional_packages.append("python-dotenv")

try:
    from llama_cpp import Llama
    local_llm_support = True
except Exception:
    local_llm_support = False
    missing_optional_packages.append("llama-cpp-python")

try:
    from openai import OpenAI
    openai_support = True
except Exception:
    openai_support = False
    missing_optional_packages.append("openai")

try:
    from anthropic import Anthropic
    anthropic_support = True
except Exception:
    anthropic_support = False
    missing_optional_packages.append("anthropic")

loaded_dotenv_path = ""

if dotenv_support is True:
    for dotenv_path in DEFAULT_DOTENV_PATHS:
        if os.path.exists(dotenv_path) is True:
            load_dotenv(dotenv_path)
            loaded_dotenv_path = dotenv_path
            break

st.set_page_config(page_title="RAG Lab App", layout="wide")

if len(missing_required_packages) > 0:
    names = []
    for package_name in missing_required_packages:
        if package_name not in names:
            names.append(package_name)

    st.error("Some required packages are missing.")
    st.write(names)
    st.code("python -m pip install " + " ".join(names))
    st.code("python -m pip install -r requirements.txt")
    st.stop()

if len(missing_optional_packages) > 0:
    names = []
    for package_name in missing_optional_packages:
        if package_name not in names:
            names.append(package_name)

    st.warning("Some optional packages are missing.")
    st.write(names)
    st.code("python -m pip install " + " ".join(names))
    st.code("python -m pip install -r requirements.txt")
    st.write("Retrieval may still work even if some optional packages are missing.")

st.title("RAG Lab App")
st.write(
    "Upload documents, inspect retrieval, compare methods, and optionally generate an answer."
)

st.info(
    "Recommended order: upload, inspect retrieval, compare fixes, then read the final answer."
)

st.caption(
    "Workflow note: ingest, retrieve, compare, trace, answer."
)


@st.cache_resource
def load_embedding_model(model_name):
    return SentenceTransformer(model_name)


@st.cache_resource
def load_local_llm(model_path, context_size, thread_count):
    if local_llm_support is False:
        return None, "llama-cpp-python is not installed."

    if os.path.exists(model_path) is False:
        return None, "The local GGUF model file was not found."

    local_model = Llama(
        model_path=model_path,
        n_ctx=context_size,
        n_threads=thread_count,
        verbose=False
    )
    return local_model, ""


@st.cache_resource
def load_reranker():
    if cross_encoder_support is False:
        return None, "CrossEncoder support is not available."

    local_config_path = os.path.join(DEFAULT_RERANKER_PATH, "config.json")

    if os.path.isdir(DEFAULT_RERANKER_PATH) and os.path.exists(local_config_path):
        return CrossEncoder(DEFAULT_RERANKER_PATH), "Loaded reranker from the local folder."

    try:
        return CrossEncoder(DEFAULT_RERANKER_MODEL_ID), "Loaded reranker from Hugging Face."
    except Exception:
        return None, "The cross-encoder reranker could not be loaded from the local folder or Hugging Face."


@st.cache_resource
def load_openai_client():
    if openai_support is False:
        return None, "openai is not installed."

    api_key = os.getenv("OPENAI_API_KEY")

    if api_key is None or api_key.strip() == "":
        if loaded_dotenv_path != "":
            return None, "OPENAI_API_KEY was not found in " + loaded_dotenv_path + "."
        return None, "OPENAI_API_KEY was not found in the environment or the local .env file."

    return OpenAI(api_key=api_key), ""


@st.cache_resource
def load_claude_client():
    if anthropic_support is False:
        return None, "anthropic is not installed."

    api_key = os.getenv("ANTHROPIC_API_KEY")

    if api_key is None or api_key.strip() == "":
        if loaded_dotenv_path != "":
            return None, "ANTHROPIC_API_KEY was not found in " + loaded_dotenv_path + "."
        return None, "ANTHROPIC_API_KEY was not found in the environment or the local .env file."

    return Anthropic(api_key=api_key), ""


def read_uploaded_file(uploaded_file):
    file_name = uploaded_file.name.lower()

    try:
        if file_name.endswith(".txt") or file_name.endswith(".md"):
            return uploaded_file.read().decode("utf-8", errors="ignore"), ""

        if file_name.endswith(".pdf"):
            if pdf_support is False:
                return "", "Could not read " + uploaded_file.name + " because pypdf is not installed."

            pdf_reader = PdfReader(uploaded_file)
            text_parts = []

            for pdf_page in pdf_reader.pages:
                page_text = pdf_page.extract_text()
                if page_text is not None:
                    text_parts.append(page_text)

            return "\n".join(text_parts), ""

        if file_name.endswith(".docx"):
            if docx_support is False:
                return "", "Could not read " + uploaded_file.name + " because python-docx is not installed."

            document = Document(uploaded_file)
            text_parts = []

            for paragraph in document.paragraphs:
                text_parts.append(paragraph.text)

            return "\n".join(text_parts), ""
    except Exception as error:
        return "", "Could not read " + uploaded_file.name + ": " + str(error)

    return "", "Could not read " + uploaded_file.name + " because the file type is not supported."


def chunk_fixed(text, chunk_size, overlap):
    chunks = []
    start_position = 0
    text_length = len(text)

    while start_position < text_length:
        end_position = start_position + chunk_size
        chunk_text = text[start_position:end_position]

        if chunk_text.strip() != "":
            chunks.append(chunk_text.strip())

        step_size = chunk_size - overlap
        if step_size <= 0:
            step_size = chunk_size

        start_position = start_position + step_size

    return chunks


def chunk_paragraph(text):
    chunks = []
    paragraphs = text.split("\n")

    for paragraph_text in paragraphs:
        if paragraph_text.strip() != "":
            chunks.append(paragraph_text.strip())

    return chunks


def build_chunk_records(uploaded_files, chunk_strategy, chunk_size, overlap):
    chunk_records = []
    file_messages = []

    for uploaded_file in uploaded_files:
        full_text, file_message = read_uploaded_file(uploaded_file)

        if file_message != "":
            file_messages.append(file_message)

        if full_text.strip() == "":
            continue

        if chunk_strategy == "paragraph":
            file_chunks = chunk_paragraph(full_text)
        else:
            file_chunks = chunk_fixed(full_text, chunk_size, overlap)

        chunk_number = 0

        for chunk_text in file_chunks:
            chunk_record = {
                "source_name": uploaded_file.name,
                "chunk_number": chunk_number,
                "text": chunk_text
            }
            chunk_records.append(chunk_record)
            chunk_number = chunk_number + 1

    return chunk_records, file_messages


def encode_chunk_records(embedding_model, chunk_records):
    chunk_embeddings = []

    for chunk_record in chunk_records:
        chunk_embedding = embedding_model.encode(chunk_record["text"])
        chunk_embeddings.append(chunk_embedding)

    return chunk_embeddings


def rewrite_question_generic(question_text):
    return "Provide detailed information about: " + question_text + ". Include important dates, rules, and requirements if available."


def create_hypothetical_document(question_text):
    return "This document explains the topic: " + question_text + ". It contains important facts, dates, requirements, and explanations."


def retrieve_top_chunks(query_text, embedding_model, chunk_records, chunk_embeddings, top_k):
    query_embedding = embedding_model.encode(query_text)
    retrieval_results = []
    chunk_index = 0

    for chunk_embedding in chunk_embeddings:
        similarity_score = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
        retrieval_result = {
            "score": float(similarity_score),
            "source_name": chunk_records[chunk_index]["source_name"],
            "chunk_number": chunk_records[chunk_index]["chunk_number"],
            "text": chunk_records[chunk_index]["text"]
        }
        retrieval_results.append(retrieval_result)
        chunk_index = chunk_index + 1

    retrieval_results = sorted(retrieval_results, key=lambda item: item["score"], reverse=True)

    if top_k > len(retrieval_results):
        top_k = len(retrieval_results)

    return retrieval_results[:top_k]


def rerank_results(question_text, retrieval_results, embedding_model):
    reranker_model, reranker_message = load_reranker()

    if reranker_model is None:
        return [], reranker_message

    pair_inputs = []

    for retrieval_result in retrieval_results:
        pair_inputs.append((question_text, retrieval_result["text"]))

    rerank_scores = reranker_model.predict(pair_inputs)
    reranked_results = []
    score_index = 0
    original_rank = 1

    for retrieval_result in retrieval_results:
        reranked_results.append({
            "score": retrieval_result["score"],
            "rerank_score": float(rerank_scores[score_index]),
            "original_rank": original_rank,
            "source_name": retrieval_result["source_name"],
            "chunk_number": retrieval_result["chunk_number"],
            "text": retrieval_result["text"],
            "rerank_source": "cross-encoder"
        })
        score_index = score_index + 1
        original_rank = original_rank + 1

    reranked_results = sorted(reranked_results, key=lambda item: item["rerank_score"], reverse=True)
    return reranked_results, reranker_message


def build_display_context_from_results(retrieval_results):
    context_text = ""
    result_index = 1

    for retrieval_result in retrieval_results:
        context_text = context_text + "[Chunk " + str(result_index) + "]\n"
        context_text = context_text + "Source: " + retrieval_result["source_name"] + "\n"

        if "score" in retrieval_result:
            context_text = context_text + "Score: " + str(round(retrieval_result["score"], 4)) + "\n"

        if "rerank_score" in retrieval_result:
            context_text = context_text + "Rerank score: " + str(round(retrieval_result["rerank_score"], 4)) + "\n"

        context_text = context_text + retrieval_result["text"] + "\n\n"
        result_index = result_index + 1

    return context_text


def build_generation_context_from_results(retrieval_results):
    context_text = ""

    for retrieval_result in retrieval_results:
        context_text = context_text + retrieval_result["text"] + "\n\n"

    return context_text


def build_evidence_based_answer(question_text, retrieval_results):
    if len(retrieval_results) == 0:
        return "No relevant evidence was retrieved."

    best_result = retrieval_results[0]
    answer_text = ""
    answer_text = answer_text + "Question: " + question_text + "\n\n"
    answer_text = answer_text + "Most relevant evidence:\n"
    answer_text = answer_text + best_result["text"] + "\n\n"
    answer_text = answer_text + "Source: " + best_result["source_name"]
    answer_text = answer_text + " | Chunk: " + str(best_result["chunk_number"])
    return answer_text


def build_generation_prompt(question_text, context_text):
    prompt_text = ""
    prompt_text = prompt_text + "Answer the question using only the context.\n\n"
    prompt_text = prompt_text + "Context:\n"
    prompt_text = prompt_text + context_text
    prompt_text = prompt_text + "\nQuestion:\n"
    prompt_text = prompt_text + question_text
    return prompt_text


def generate_local_answer(local_model, question_text, context_text, max_tokens, temperature):
    prompt_text = build_generation_prompt(question_text, context_text)
    llm_output = local_model(prompt_text, max_tokens=max_tokens, temperature=temperature)
    return llm_output["choices"][0]["text"]


def generate_openai_answer(client, model_name, question_text, context_text, max_tokens):
    prompt_text = build_generation_prompt(question_text, context_text)

    response = client.responses.create(
        model=model_name,
        instructions="You are a helpful assistant. Use only the retrieved context. If the context is not enough, say so.",
        input=prompt_text,
        max_output_tokens=max_tokens
    )

    return response.output_text


def generate_claude_answer(client, model_name, question_text, context_text, max_tokens, temperature):
    prompt_text = build_generation_prompt(question_text, context_text)

    response = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt_text}]
    )

    output_text = ""

    for content_block in response.content:
        if hasattr(content_block, "text"):
            output_text = output_text + content_block.text

    return output_text


def run_method_pipeline(question_text, method_name, embedding_model, chunk_records, chunk_embeddings, top_k, rerank_candidate_k):
    query_used_for_retrieval = question_text
    initial_results = []
    final_results = []
    reranker_message = ""
    rerank_applied = False

    if method_name == "basic":
        initial_results = retrieve_top_chunks(question_text, embedding_model, chunk_records, chunk_embeddings, top_k)
        final_results = initial_results

    if method_name == "rewrite":
        query_used_for_retrieval = rewrite_question_generic(question_text)
        initial_results = retrieve_top_chunks(query_used_for_retrieval, embedding_model, chunk_records, chunk_embeddings, top_k)
        final_results = initial_results

    if method_name == "hyde":
        query_used_for_retrieval = create_hypothetical_document(question_text)
        initial_results = retrieve_top_chunks(query_used_for_retrieval, embedding_model, chunk_records, chunk_embeddings, top_k)
        final_results = initial_results

    if method_name == "rerank":
        initial_results = retrieve_top_chunks(question_text, embedding_model, chunk_records, chunk_embeddings, rerank_candidate_k)
        final_results, reranker_message = rerank_results(question_text, initial_results, embedding_model)
        final_results = final_results[:top_k]

        if len(final_results) > 0:
            rerank_applied = True

    return {
        "question_text": question_text,
        "method_name": method_name,
        "query_used_for_retrieval": query_used_for_retrieval,
        "initial_results": initial_results,
        "final_results": final_results,
        "display_context_text": build_display_context_from_results(final_results),
        "generation_context_text": build_generation_context_from_results(final_results),
        "evidence_answer": build_evidence_based_answer(question_text, final_results),
        "reranker_message": reranker_message,
        "rerank_applied": rerank_applied,
        "context_character_count": len(build_generation_context_from_results(final_results))
    }


def highlight_question_words(question_text, chunk_text):
    question_words = question_text.lower().split()
    highlighted_text = chunk_text

    for question_word in question_words:
        if len(question_word) > 3:
            highlighted_text = highlighted_text.replace(question_word, "**" + question_word + "**")
            highlighted_text = highlighted_text.replace(question_word.capitalize(), "**" + question_word.capitalize() + "**")

    return highlighted_text


def build_observability_rows(retrieval_results):
    observability_rows = []

    for retrieval_result in retrieval_results:
        row = {
            "source": retrieval_result["source_name"],
            "chunk_number": retrieval_result["chunk_number"],
            "score": round(retrieval_result["score"], 4),
            "text": retrieval_result["text"]
        }

        if "rerank_score" in retrieval_result:
            row["rerank_score"] = round(retrieval_result["rerank_score"], 4)

        if "rerank_source" in retrieval_result:
            row["rerank_source"] = retrieval_result["rerank_source"]

        if "original_rank" in retrieval_result:
            row["original_rank"] = retrieval_result["original_rank"]

        observability_rows.append(row)

    return observability_rows


model_file_path = DEFAULT_DATAHUB_MODEL_PATH

with st.sidebar:
    st.header("Retrieval Settings")
    st.caption("These settings control chunking and retrieval.")

    embedding_model_name = st.selectbox("Embedding model", DEFAULT_EMBEDDING_MODELS)
    chunk_strategy = st.selectbox("Chunk strategy", ["paragraph", "fixed"])
    retrieval_method_label = st.selectbox(
        "Retrieval method",
        ["basic", "rewrite", "hyde", "rerank (advanced)"]
    )

    with st.expander("Advanced Retrieval Settings", expanded=False):
        chunk_size = st.slider("Chunk size", min_value=100, max_value=1500, value=DEFAULT_CHUNK_SIZE, step=50)
        chunk_overlap = st.slider("Chunk overlap", min_value=0, max_value=300, value=DEFAULT_CHUNK_OVERLAP, step=10)
        top_k = st.slider("Top-k retrieval", min_value=1, max_value=10, value=DEFAULT_TOP_K, step=1)
        rerank_candidate_k = st.slider("Rerank candidate count", min_value=3, max_value=20, value=DEFAULT_RERANK_CANDIDATE_K, step=1)

retrieval_method = retrieval_method_label

if retrieval_method_label == "rerank (advanced)":
    retrieval_method = "rerank"

    st.caption("Advanced option: retrieve more candidates, then rerank them.")

with st.sidebar:
    st.header("Generation Settings")
    st.caption("Generation is optional.")

    llm_mode = st.selectbox("LLM mode", DEFAULT_LLM_MODE_OPTIONS, index=DEFAULT_LLM_MODE_INDEX)

    with st.expander("Advanced Generation Settings", expanded=False):
        model_file_path = st.text_input("Local GGUF model path", model_file_path)
        local_context_size = st.slider("Local context size", min_value=512, max_value=4096, value=DEFAULT_LOCAL_CONTEXT_SIZE, step=256)
        local_thread_count = st.slider("Local thread count", min_value=1, max_value=16, value=DEFAULT_LOCAL_THREAD_COUNT, step=1)
        openai_model_name = st.selectbox("OpenAI model name", DEFAULT_OPENAI_MODELS)
        claude_model_name = st.selectbox("Claude model name", DEFAULT_CLAUDE_MODELS)
        max_generation_tokens = st.slider("Generation max tokens", min_value=50, max_value=400, value=DEFAULT_MAX_GENERATION_TOKENS, step=10)
        generation_temperature = st.slider("Generation temperature", min_value=0.0, max_value=1.0, value=DEFAULT_GENERATION_TEMPERATURE, step=0.1)

if rerank_candidate_k < top_k:
    rerank_candidate_k = top_k

preloaded_local_model = None
preloaded_local_message = ""

if PRELOAD_LOCAL_MODEL_ON_START is True and llm_mode == "Local":
    with st.spinner("Loading the local model at startup..."):
        preloaded_local_model, preloaded_local_message = load_local_llm(model_file_path, local_context_size, local_thread_count)

    if preloaded_local_model is not None:
        st.success("Local model loaded successfully.")
    else:
        st.warning(preloaded_local_message)

uploaded_files = st.file_uploader("Upload documents", type=["txt", "md", "pdf", "docx"], accept_multiple_files=True)

if uploaded_files is None or len(uploaded_files) == 0:
    st.info("Upload documents to begin.")
    st.write("Suggested first question:")
    st.code(DEFAULT_EXAMPLE_QUESTION)
    st.write("Then compare Basic, Rewrite, HyDE, and the optional advanced reranker.")
    st.stop()

embedding_model = load_embedding_model(embedding_model_name)
chunk_records, file_messages = build_chunk_records(uploaded_files, chunk_strategy, chunk_size, chunk_overlap)

for file_message in file_messages:
    st.warning(file_message)

if len(chunk_records) == 0:
    st.warning("No chunks were created from the uploaded files.")
    st.stop()

chunk_embeddings = encode_chunk_records(embedding_model, chunk_records)

st.subheader("System Overview")
overview_column_1, overview_column_2, overview_column_3, overview_column_4 = st.columns(4)
overview_column_1.metric("Chunk count", len(chunk_records))
overview_column_2.metric("Chunk strategy", chunk_strategy)
overview_column_3.metric("Top-k", top_k)
overview_column_4.metric("LLM mode", llm_mode)

st.write("**Workflow checkpoints**")
st.write("1. Ingest documents and inspect the chunks.")
st.write("2. Ask a question and inspect retrieved chunks first.")
st.write("3. Compare Rewrite, HyDE, and optional Rerank.")
st.write("4. Inspect scores and rank movement.")
st.write("5. Read the final context and answer.")

if retrieval_method == "rerank":
    st.caption(
        "Advanced option: rerank first retrieves the top " + str(rerank_candidate_k) +
        " candidates with embeddings, then reorders them with a reranker."
    )

question_text = st.text_input("Ask a question", value=DEFAULT_EXAMPLE_QUESTION)

if question_text.strip() == "":
    st.warning("Please enter a question.")
    st.stop()

pipeline_output = run_method_pipeline(question_text, retrieval_method, embedding_model, chunk_records, chunk_embeddings, top_k, rerank_candidate_k)

basic_output = run_method_pipeline(question_text, "basic", embedding_model, chunk_records, chunk_embeddings, top_k, rerank_candidate_k)
rewrite_output = run_method_pipeline(question_text, "rewrite", embedding_model, chunk_records, chunk_embeddings, top_k, rerank_candidate_k)
hyde_output = run_method_pipeline(question_text, "hyde", embedding_model, chunk_records, chunk_embeddings, top_k, rerank_candidate_k)
rerank_output = run_method_pipeline(question_text, "rerank", embedding_model, chunk_records, chunk_embeddings, top_k, rerank_candidate_k)

generated_answer = ""
generation_message = ""

if llm_mode == "Local":
    local_model = preloaded_local_model
    local_message = preloaded_local_message

    if local_model is None and PRELOAD_LOCAL_MODEL_ON_START is False:
        with st.spinner("Loading the local model..."):
            local_model, local_message = load_local_llm(model_file_path, local_context_size, local_thread_count)

    if local_model is None:
        generation_message = local_message
    else:
        try:
            generated_answer = generate_local_answer(local_model, question_text, pipeline_output["generation_context_text"], max_generation_tokens, generation_temperature)
        except Exception as error:
            generation_message = "Local generation failed: " + str(error)

if llm_mode == "OpenAI":
    openai_client, openai_message = load_openai_client()

    if openai_client is None:
        generation_message = openai_message
    else:
        try:
            generated_answer = generate_openai_answer(openai_client, openai_model_name, question_text, pipeline_output["generation_context_text"], max_generation_tokens)
        except Exception as error:
            generation_message = "OpenAI generation failed: " + str(error)

if llm_mode == "Claude":
    claude_client, claude_message = load_claude_client()

    if claude_client is None:
        generation_message = claude_message
    else:
        try:
            generated_answer = generate_claude_answer(claude_client, claude_model_name, question_text, pipeline_output["generation_context_text"], max_generation_tokens, generation_temperature)
        except Exception as error:
            generation_message = "Claude generation failed: " + str(error)

if retrieval_method == "rerank" and pipeline_output["reranker_message"] != "":
    st.info(pipeline_output["reranker_message"])

tab_retrieval, tab_compare, tab_scores, tab_trace, tab_chat = st.tabs([
    "1. Retrieval",
    "2. Compare Fixes",
    "3. Scores",
    "4. Workflow Trace",
    "5. Answer",
])

with tab_retrieval:
    st.subheader("Step 2: Retrieval Inspection")
    st.write("Use this tab first. Inspect rank, score, and chunk text.")
    retrieval_rank = 1

    for retrieval_result in pipeline_output["final_results"]:
        with st.expander("Rank " + str(retrieval_rank) + " | " + retrieval_result["source_name"]):
            if "score" in retrieval_result:
                st.write("Similarity score:", round(retrieval_result["score"], 4))
            if "rerank_score" in retrieval_result:
                st.write("Rerank score:", round(retrieval_result["rerank_score"], 4))
            if "rerank_source" in retrieval_result:
                st.write("Rerank type:", retrieval_result["rerank_source"])
            if "original_rank" in retrieval_result:
                st.write("Original rank before reranking:", retrieval_result["original_rank"])
            st.write("Chunk number:", retrieval_result["chunk_number"])
            st.write(highlight_question_words(question_text, retrieval_result["text"]))
        retrieval_rank = retrieval_rank + 1

with tab_compare:
    st.subheader("Step 3: Compare Retrieval Fixes")
    st.write("Compare the top result from Basic, Rewrite, HyDE, and Rerank.")

    compare_column_1, compare_column_2, compare_column_3, compare_column_4 = st.columns(4)

    with compare_column_1:
        st.write("### Basic")
        if len(basic_output["final_results"]) > 0:
            st.write("Top score:", round(basic_output["final_results"][0]["score"], 4))
            st.write("Source:", basic_output["final_results"][0]["source_name"])
            st.write(basic_output["final_results"][0]["text"])

    with compare_column_2:
        st.write("### Rewrite")
        if len(rewrite_output["final_results"]) > 0:
            st.write("Top score:", round(rewrite_output["final_results"][0]["score"], 4))
            st.write("Source:", rewrite_output["final_results"][0]["source_name"])
            st.write(rewrite_output["final_results"][0]["text"])

    with compare_column_3:
        st.write("### HyDE")
        if len(hyde_output["final_results"]) > 0:
            st.write("Top score:", round(hyde_output["final_results"][0]["score"], 4))
            st.write("Source:", hyde_output["final_results"][0]["source_name"])
            st.write(hyde_output["final_results"][0]["text"])

    with compare_column_4:
        st.write("### Rerank")
        if len(rerank_output["final_results"]) > 0:
            if "rerank_score" in rerank_output["final_results"][0]:
                st.write("Top rerank score:", round(rerank_output["final_results"][0]["rerank_score"], 4))
            if "original_rank" in rerank_output["final_results"][0]:
                st.write("Original rank:", rerank_output["final_results"][0]["original_rank"])
            st.write("Source:", rerank_output["final_results"][0]["source_name"])
            st.write(rerank_output["final_results"][0]["text"])

with tab_scores:
    st.subheader("Step 4: Scores and Rank Movement")
    st.write("Use this tab after reading the text results.")

    query_for_scores = pipeline_output["query_used_for_retrieval"]
    query_embedding = embedding_model.encode(query_for_scores)

    all_scores = []

    for chunk_embedding in chunk_embeddings:
        similarity_score = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
        all_scores.append(similarity_score)

    if plot_support is True:
        st.write("**All chunk scores**")
        st.caption("This line plot shows the similarity score for each chunk.")
        figure = plt.figure()
        plt.plot(all_scores)
        plt.title("Similarity score for each chunk")
        plt.xlabel("Chunk index")
        plt.ylabel("Similarity score")
        st.pyplot(figure)

        if retrieval_method == "rerank":
            st.write("**Reranked result scores**")
            st.caption("Cross-encoder scores use a different scale, so they are shown separately.")

            rerank_labels = []
            embedding_scores = []
            rerank_scores = []
            original_rank_positions = []
            rank_number = 1

            for retrieval_result in pipeline_output["final_results"]:
                rerank_labels.append("Rank " + str(rank_number))
                embedding_scores.append(retrieval_result["score"])
                rerank_scores.append(retrieval_result["rerank_score"])
                original_rank_positions.append(retrieval_result["original_rank"])
                rank_number = rank_number + 1

            figure, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].bar(rerank_labels, embedding_scores)
            axes[0].set_title("Original embedding scores")
            axes[0].set_xlabel("Final reranked position")
            axes[0].set_ylabel("Cosine similarity")

            axes[1].bar(rerank_labels, rerank_scores)
            axes[1].set_title("Cross-encoder rerank scores")
            axes[1].set_xlabel("Final reranked position")
            axes[1].set_ylabel("Rerank score")

            plt.tight_layout()
            st.pyplot(figure)

            st.write("**Rank movement after reranking**")
            st.caption("A lower original rank means the chunk started closer to the top.")
            figure = plt.figure()
            plt.bar(rerank_labels, original_rank_positions)
            plt.title("Original rank of each final reranked chunk")
            plt.xlabel("Final reranked position")
            plt.ylabel("Original rank")
            plt.gca().invert_yaxis()
            st.pyplot(figure)
        else:
            st.write("**Top-k scores**")
            st.caption("This bar chart shows cosine similarity for the retrieved top-k chunks.")
            top_k_labels = []
            top_k_scores = []
            rank_number = 1

            for retrieval_result in pipeline_output["final_results"]:
                top_k_labels.append("Rank " + str(rank_number))
                top_k_scores.append(retrieval_result["score"])
                rank_number = rank_number + 1

            figure = plt.figure()
            plt.bar(top_k_labels, top_k_scores)
            plt.title("Top-k retrieved chunk scores")
            plt.xlabel("Retrieved rank")
            plt.ylabel("Cosine similarity")
            st.pyplot(figure)

        st.write("**Embedding method comparison**")
        st.caption("Basic, Rewrite, and HyDE all use cosine similarity.")
        method_names = ["Basic", "Rewrite", "HyDE"]
        method_scores = [
            basic_output["final_results"][0]["score"],
            rewrite_output["final_results"][0]["score"],
            hyde_output["final_results"][0]["score"],
        ]

        figure = plt.figure()
        plt.bar(method_names, method_scores)
        plt.title("Top cosine similarity by embedding-based method")
        plt.xlabel("Method")
        plt.ylabel("Cosine similarity")
        st.pyplot(figure)

        if len(rerank_output["final_results"]) > 0 and rerank_output["rerank_applied"] is True:
            st.write("**Rerank summary**")
            st.caption("Rerank uses a different scoring model.")
            rerank_metric_column_1, rerank_metric_column_2, rerank_metric_column_3 = st.columns(3)
            rerank_metric_column_1.metric("Top rerank score", round(rerank_output["final_results"][0]["rerank_score"], 4))
            rerank_metric_column_2.metric("Original rank", rerank_output["final_results"][0]["original_rank"])
            rerank_metric_column_3.metric("Candidate count", rerank_candidate_k)
        else:
            st.info("The advanced reranker is unavailable.")
    else:
        st.warning("matplotlib is not installed, so plots are unavailable.")

with tab_trace:
    st.subheader("Workflow Trace")
    st.write("Use this tab to inspect the full pipeline.")

    trace_column_1, trace_column_2, trace_column_3, trace_column_4 = st.columns(4)
    trace_column_1.metric("Uploaded files", len(uploaded_files))
    trace_column_2.metric("Chunk count", len(chunk_records))
    trace_column_3.metric("Initial candidates", len(pipeline_output["initial_results"]))
    trace_column_4.metric("Context characters", pipeline_output["context_character_count"])

    st.write("**Step 1: User question**")
    st.write(question_text)

    st.write("**Step 2: Retrieval query**")
    st.write(pipeline_output["query_used_for_retrieval"])

    if retrieval_method == "rerank":
        st.write("**Step 3: Initial embedding retrieval candidates**")
        st.json(build_observability_rows(pipeline_output["initial_results"]))

        st.write("**Step 4: Final reranked results**")
        st.json(build_observability_rows(pipeline_output["final_results"]))
    else:
        st.write("**Step 3: Retrieved chunks**")
        st.json(build_observability_rows(pipeline_output["final_results"]))

    st.write("**Step 5: Display context for inspection**")
    st.text(pipeline_output["display_context_text"])

    st.write("**Step 6: Plain generation context**")
    st.text(pipeline_output["generation_context_text"])

    st.write("**Step 7: Failure attribution guide**")
    st.write("1. If the wrong chunk appears at the top, the problem is mainly retrieval.")
    st.write("2. If a better chunk appears in the candidate list but not at the top, reranking may help.")
    st.write("3. If the right chunk is in the final context but the answer is still wrong, the problem is mainly generation.")
    st.write("Generation mode:", llm_mode)

with tab_chat:
    st.subheader("Step 5: Answer")
    st.write("Use this tab last.")
    st.write("**Question**")
    st.write(question_text)
    st.write("**Retrieval method**")
    st.write(retrieval_method_label)
    st.write("**Query used for retrieval**")
    st.write(pipeline_output["query_used_for_retrieval"])

    if retrieval_method == "rerank" and pipeline_output["reranker_message"] != "":
        st.info(pipeline_output["reranker_message"])

    st.write("**Evidence-based answer**")
    st.caption("This answer uses the top retrieved evidence directly.")
    st.text(pipeline_output["evidence_answer"])
    st.write("**Context shown in the workflow view**")
    st.caption("This view includes source and score information.")
    st.text(pipeline_output["display_context_text"])
    st.write("**Generated answer**")
    st.caption("This answer uses only the plain chunk text.")

    if generated_answer.strip() != "":
        st.text(generated_answer)
    elif generation_message != "":
        st.warning(generation_message)
    else:
        st.info("Generation is disabled because LLM mode is set to None.")
