# Small_Models_SP26

Introductory AI learning materials for freshmen and community college students. All notebooks can run on CPU on a shared JupyterHub but run faster on GPU enabled Juptyerhubs.  

This work was presented as part of a tutorial at the 2026 NAIRR Annual Meeting. 


## Repository Layout

### 1 — Instructor Setup Utils (`1-Instructor_Utils/`)
Notebooks the instructor runs once before class. Covers downloading GGUF model weights from Hugging Face Hub to a shared directory, verifying GPU availability, and configuring API keys. Students do not need to run these.

### 2 — Intro Notebooks (`2-Beginner_NBs/`)
First notebooks a student touches. Introduces loading a GGUF model with `llama-cpp-python`, sending a prompt, and reading the response. Also covers how model weights and tokens work at a conceptual level. No prior ML knowledge assumed.

### 3 — API Notebooks (`3-API_NBs/`)
Shows students how to call hosted LLM APIs instead of running a local model. Includes examples for the OpenAI API and the Anthropic API. Good for comparing local model output to a larger cloud model.

### 4 — SAT Test Taker (`4-SAT_Test_Taker/`)
Students load a small model and have it answer SAT-style multiple choice questions pulled from the PineSAT open question bank. Covers prompt engineering, answer extraction with regex, and batch scoring. Students can filter by difficulty and subject and see a results table.

### 5 — RAG Notebooks (`5-RAG/`)
Three approaches each applying Retrieval-Augmented Generation (RAG) to a different corpus. RAG lets a model answer questions about a specific document collection by searching for relevant passages before generating a response. The three corpuses demonstrate how the same technique applies across economics applications.

### 6 — No-Code Notebooks (`6-No_Code/`)
Narrative notebooks with interactive elements through widgets. Used for conceptual lessons, discussion prompts, and reading assignments. Good for class sessions where the goal is understanding rather than coding.  However this does provide a completely customizable format for any lesson, so it could be adapted to local use cases.

---

## JupyterHub Links  [These links were for the March 2026 Tutorial and may not stay live]

- Workshop Hub (Shared Password): [Launch](https://gpu-demo.cloudbank.2i2c.cloud/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2Fds-modules%2FSmall_Models_SP26&urlpath=lab%2Ftree%2FSmall_Models_SP26%2F)
- NRP Hub (CILogon): [Launch](https://training.nrp-nautilus.io/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2Fds-modules%2FSmall_Models_SP26&branch=main&urlpath=lab%2Ftree%2FSmall_Models_SP26%2F) — click **Sign in with Authentik**, then look for CILogon, then refresh the page
- Jetstream2 Hub (Shared Password): [Launch](https://hub.soc260005.projects.jetstream-cloud.org/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2Fds-modules%2FSmall_Models_SP26&urlpath=lab%2Ftree%2FSmall_Models_SP26%2F)
