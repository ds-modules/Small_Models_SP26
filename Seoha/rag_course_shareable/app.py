import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# =========================
# EASY-TO-EDIT SETTINGS
# =========================
# HOW TO CHANGE THIS:
# - If you want different default local model paths, edit the lists below.
# - If you want different API model names, edit the lists below.
# - If you want different default chunk settings, edit the constants below.

DEFAULT_DATAHUB_MODEL_PATH = "/home/jovyan/shared/qwen2-1_5b-instruct-q4_0.gguf"

DEFAULT_LOCAL_MODEL_PATHS = [
    os.path.expanduser("~/Downloads/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"),
    os.path.expanduser("~/Downloads/qwen2-1_5b-instruct-q4_0.gguf"),
    os.path.expanduser("~/Downloads/model.gguf"),
]

DEFAULT_EMBEDDING_MODELS = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2"
]

DEFAULT_GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile"
]

DEFAULT_OPENAI_MODELS = [
    "gpt-4o-mini",
    "gpt-4.1-mini"
]

DEFAULT_EXAMPLE_QUESTION = "When can students apply for financial aid?"
DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_TOP_K = 3

# =========================
# OPTIONAL IMPORTS
# =========================
# These packages enable optional features.
# The app can still work for retrieval even if some are missing.

missing_packages = []

try:
    from pypdf import PdfReader
    pdf_support = True
except Exception:
    pdf_support = False
    missing_packages.append("pypdf")

try:
    from docx import Document
    docx_support = True
except Exception:
    docx_support = False
    missing_packages.append("python-docx")

try:
    from llama_cpp import Llama
    local_llm_support = True
except Exception:
    local_llm_support = False
    missing_packages.append("llama-cpp-python")

try:
    from dotenv import load_dotenv
    from groq import Groq
    groq_support = True
except Exception:
    groq_support = False
    missing_packages.append("python-dotenv")
    missing_packages.append("groq")

try:
    from openai import OpenAI
    openai_support = True
except Exception:
    openai_support = False
    missing_packages.append("openai")


st.set_page_config(page_title="RAG Lab App", layout="wide")

if len(missing_packages) > 0:
    unique_missing_packages = []

    for package_name in missing_packages:
        if package_name not in unique_missing_packages:
            unique_missing_packages.append(package_name)

    install_command = "python -m pip install " + " ".join(unique_missing_packages)

    st.warning("Some optional packages are missing.")
    st.write("Missing packages:")
    st.write(unique_missing_packages)

    st.write("Install them in a terminal with:")
    st.code(install_command)

    st.write("Or install everything with:")
    st.code("python -m pip install -r requirements.txt")

    st.write("The app may still work for retrieval even if some optional packages are missing.")


st.title("RAG Lab App")
st.write(
    "Upload documents, inspect retrieval, compare methods, and optionally generate an answer."
)


# =========================
# MODEL PATH HELPERS
# =========================
# LOCAL VS DATAHUB:
# - The app first tries the DataHub path.
# - If it does not exist, it tries common local paths.
# - Users can always override the path in the sidebar.

def choose_default_model_path():
    if os.path.exists(DEFAULT_DATAHUB_MODEL_PATH):
        return DEFAULT_DATAHUB_MODEL_PATH

    for local_path in DEFAULT_LOCAL_MODEL_PATHS:
        if os.path.exists(local_path):
            return local_path

    return DEFAULT_DATAHUB_MODEL_PATH


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
def load_groq_client():
    if groq_support is False:
        return None, "groq or python-dotenv is not installed."

    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

    if api_key is None or api_key.strip() == "":
        return None, "GROQ_API_KEY was not found."

    return Groq(api_key=api_key), ""


@st.cache_resource
def load_openai_client():
    if openai_support is False:
        return None, "openai is not installed."

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key is None or api_key.strip() == "":
        return None, "OPENAI_API_KEY was not found."

    return OpenAI(api_key=api_key), ""


# =========================
# FILE READING
# =========================
# HOW TO CHANGE THIS:
# - Add more file types here if needed.

def read_uploaded_file(uploaded_file):
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".txt") or file_name.endswith(".md"):
        return uploaded_file.read().decode("utf-8", errors="ignore")

    if file_name.endswith(".pdf"):
        if pdf_support is False:
            return ""

        pdf_reader = PdfReader(uploaded_file)
        text_parts = []

        for pdf_page in pdf_reader.pages:
            page_text = pdf_page.extract_text()

            if page_text is not None:
                text_parts.append(page_text)

        return "\n".join(text_parts)

    if file_name.endswith(".docx"):
        if docx_support is False:
            return ""

        document = Document(uploaded_file)
        text_parts = []

        for paragraph in document.paragraphs:
            text_parts.append(paragraph.text)

        return "\n".join(text_parts)

    return ""


# =========================
# CHUNKING
# =========================
# HOW TO CHANGE THIS:
# - You can add a new chunking function here.
# - Then add it to the sidebar options.

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

    for uploaded_file in uploaded_files:
        full_text = read_uploaded_file(uploaded_file)

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

    return chunk_records


# =========================
# EMBEDDING + RETRIEVAL
# =========================
def encode_chunk_records(embedding_model, chunk_records):
    chunk_embeddings = []

    for chunk_record in chunk_records:
        chunk_embedding = embedding_model.encode(chunk_record["text"])
        chunk_embeddings.append(chunk_embedding)

    return chunk_embeddings


def rewrite_question_generic(question_text):
    rewritten_question = ""
    rewritten_question = rewritten_question + "Provide detailed information about: "
    rewritten_question = rewritten_question + question_text
    rewritten_question = rewritten_question + ". Include important dates, rules, and requirements if available."
    return rewritten_question


def create_hypothetical_document(question_text):
    hypothetical_document = ""
    hypothetical_document = hypothetical_document + "This document explains the topic: "
    hypothetical_document = hypothetical_document + question_text
    hypothetical_document = hypothetical_document + ". It contains important facts, dates, requirements, and explanations."
    return hypothetical_document


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

    top_results = []
    result_index = 0

    for retrieval_result in retrieval_results:
        if result_index < top_k:
            top_results.append(retrieval_result)

        result_index = result_index + 1

    return top_results


def rerank_results_with_embeddings(question_text, retrieval_results, embedding_model):
    question_embedding = embedding_model.encode(question_text)
    reranked_results = []

    for retrieval_result in retrieval_results:
        result_embedding = embedding_model.encode(retrieval_result["text"])
        rerank_score = cosine_similarity([question_embedding], [result_embedding])[0][0]

        new_result = {
            "score": retrieval_result["score"],
            "rerank_score": float(rerank_score),
            "source_name": retrieval_result["source_name"],
            "chunk_number": retrieval_result["chunk_number"],
            "text": retrieval_result["text"]
        }

        reranked_results.append(new_result)

    reranked_results = sorted(reranked_results, key=lambda item: item["rerank_score"], reverse=True)
    return reranked_results


def build_context_from_results(retrieval_results):
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


# =========================
# GENERATION
# =========================
# HOW TO CHANGE THIS:
# - Change the prompt format below.
# - Change default API models in the settings section.
# - Keep local generation as the default if you want easy classroom use.

def build_generation_prompt(question_text, context_text):
    prompt_text = ""
    prompt_text = prompt_text + "You are a helpful assistant.\n"
    prompt_text = prompt_text + "Answer the question using only the context.\n\n"
    prompt_text = prompt_text + "Context:\n"
    prompt_text = prompt_text + context_text
    prompt_text = prompt_text + "\nQuestion:\n"
    prompt_text = prompt_text + question_text
    prompt_text = prompt_text + "\n\nAnswer:\n"
    return prompt_text


def generate_local_answer(local_model, question_text, context_text, max_tokens, temperature):
    prompt_text = build_generation_prompt(question_text, context_text)
    llm_output = local_model(prompt_text, max_tokens=max_tokens, temperature=temperature)
    return llm_output["choices"][0]["text"]


def generate_groq_answer(client, model_name, question_text, context_text, temperature):
    prompt_text = build_generation_prompt(question_text, context_text)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=temperature
    )
    return response.choices[0].message.content


def generate_openai_answer(client, model_name, question_text, context_text, temperature):
    prompt_text = build_generation_prompt(question_text, context_text)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=temperature
    )
    return response.choices[0].message.content


def run_method_pipeline(question_text, method_name, embedding_model, chunk_records, chunk_embeddings, top_k):
    query_used_for_retrieval = question_text
    initial_results = []
    final_results = []

    if method_name == "basic":
        initial_results = retrieve_top_chunks(query_used_for_retrieval, embedding_model, chunk_records, chunk_embeddings, top_k)
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
        initial_results = retrieve_top_chunks(question_text, embedding_model, chunk_records, chunk_embeddings, top_k)
        final_results = rerank_results_with_embeddings(question_text, initial_results, embedding_model)

    return {
        "question_text": question_text,
        "method_name": method_name,
        "query_used_for_retrieval": query_used_for_retrieval,
        "final_results": final_results,
        "context_text": build_context_from_results(final_results),
        "evidence_answer": build_evidence_based_answer(question_text, final_results)
    }


def highlight_question_words(question_text, chunk_text):
    question_words = question_text.lower().split()
    highlighted_text = chunk_text

    for question_word in question_words:
        if len(question_word) > 3:
            highlighted_text = highlighted_text.replace(question_word, "**" + question_word + "**")
            highlighted_text = highlighted_text.replace(question_word.capitalize(), "**" + question_word.capitalize() + "**")

    return highlighted_text


default_model_path = choose_default_model_path()

with st.sidebar:
    st.header("Retrieval Settings")

    embedding_model_name = st.selectbox(
        "Embedding model",
        DEFAULT_EMBEDDING_MODELS
    )

    chunk_strategy = st.selectbox(
        "Chunk strategy",
        ["fixed", "paragraph"]
    )

    chunk_size = st.slider(
        "Chunk size",
        min_value=100,
        max_value=1500,
        value=DEFAULT_CHUNK_SIZE,
        step=50
    )

    chunk_overlap = st.slider(
        "Chunk overlap",
        min_value=0,
        max_value=300,
        value=DEFAULT_CHUNK_OVERLAP,
        step=10
    )

    top_k = st.slider(
        "Top-k retrieval",
        min_value=1,
        max_value=10,
        value=DEFAULT_TOP_K,
        step=1
    )

    retrieval_method = st.selectbox(
        "Retrieval method",
        ["basic", "rewrite", "hyde", "rerank"]
    )

    st.header("LLM Settings")

    llm_mode = st.selectbox(
        "LLM mode",
        ["Local", "API (optional)", "None"]
    )

    model_file_path = st.text_input(
        "Local GGUF model path",
        default_model_path
    )

    local_context_size = st.slider(
        "Local context size",
        min_value=512,
        max_value=4096,
        value=2048,
        step=256
    )

    local_thread_count = st.slider(
        "Local thread count",
        min_value=1,
        max_value=16,
        value=4,
        step=1
    )

    api_provider = st.selectbox(
        "API provider (optional)",
        ["Groq", "OpenAI"]
    )

    groq_model_name = st.selectbox(
        "Groq model name",
        DEFAULT_GROQ_MODELS
    )

    openai_model_name = st.selectbox(
        "OpenAI model name",
        DEFAULT_OPENAI_MODELS
    )

    max_generation_tokens = st.slider(
        "Generation max tokens",
        min_value=50,
        max_value=400,
        value=160,
        step=10
    )

    generation_temperature = st.slider(
        "Generation temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1
    )

uploaded_files = st.file_uploader(
    "Upload documents",
    type=["txt", "md", "pdf", "docx"],
    accept_multiple_files=True
)

if uploaded_files is None or len(uploaded_files) == 0:
    st.info("Upload documents to begin.")
    st.write("Suggested first question:")
    st.code(DEFAULT_EXAMPLE_QUESTION)
    st.write("After uploading a file, compare Basic, Rewrite, HyDE, and Rerank.")
    st.stop()

embedding_model = load_embedding_model(embedding_model_name)
chunk_records = build_chunk_records(uploaded_files, chunk_strategy, chunk_size, chunk_overlap)

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

question_text = st.text_input("Ask a question", value=DEFAULT_EXAMPLE_QUESTION)

if question_text.strip() == "":
    st.warning("Please enter a question.")
    st.stop()

pipeline_output = run_method_pipeline(question_text, retrieval_method, embedding_model, chunk_records, chunk_embeddings, top_k)

generated_answer = ""
generation_message = ""

if llm_mode == "Local":
    local_model, local_message = load_local_llm(model_file_path, local_context_size, local_thread_count)

    if local_model is None:
        generation_message = local_message
    else:
        generated_answer = generate_local_answer(
            local_model,
            question_text,
            pipeline_output["context_text"],
            max_generation_tokens,
            generation_temperature
        )

if llm_mode == "API (optional)":
    if api_provider == "Groq":
        groq_client, groq_message = load_groq_client()

        if groq_client is None:
            generation_message = groq_message
        else:
            generated_answer = generate_groq_answer(
                groq_client,
                groq_model_name,
                question_text,
                pipeline_output["context_text"],
                generation_temperature
            )

    if api_provider == "OpenAI":
        openai_client, openai_message = load_openai_client()

        if openai_client is None:
            generation_message = openai_message
        else:
            generated_answer = generate_openai_answer(
                openai_client,
                openai_model_name,
                question_text,
                pipeline_output["context_text"],
                generation_temperature
            )

tab_chat, tab_retrieval, tab_compare, tab_scores = st.tabs(
    ["Chat", "Retrieval", "Compare", "Scores"]
)

with tab_chat:
    st.subheader("Chat")
    st.write("**Question**")
    st.write(question_text)

    st.write("**Retrieval method**")
    st.write(retrieval_method)

    st.write("**Query used for retrieval**")
    st.write(pipeline_output["query_used_for_retrieval"])

    st.write("**Evidence-based answer**")
    st.text(pipeline_output["evidence_answer"])

    st.write("**Context used by the system**")
    st.text(pipeline_output["context_text"])

    st.write("**Generated answer**")

    if generated_answer.strip() != "":
        st.text(generated_answer)
    elif generation_message != "":
        st.warning(generation_message)
    else:
        st.info("Generation is disabled because LLM mode is set to None.")

with tab_retrieval:
    st.subheader("Retrieved Chunks")
    retrieval_rank = 1

    for retrieval_result in pipeline_output["final_results"]:
        with st.expander("Rank " + str(retrieval_rank) + " | " + retrieval_result["source_name"]):
            if "score" in retrieval_result:
                st.write("Similarity score:", round(retrieval_result["score"], 4))

            if "rerank_score" in retrieval_result:
                st.write("Rerank score:", round(retrieval_result["rerank_score"], 4))

            st.write("Chunk number:", retrieval_result["chunk_number"])
            st.write(highlight_question_words(question_text, retrieval_result["text"]))

        retrieval_rank = retrieval_rank + 1

with tab_compare:
    st.subheader("Compare Methods")

    basic_output = run_method_pipeline(question_text, "basic", embedding_model, chunk_records, chunk_embeddings, top_k)
    rewrite_output = run_method_pipeline(question_text, "rewrite", embedding_model, chunk_records, chunk_embeddings, top_k)
    hyde_output = run_method_pipeline(question_text, "hyde", embedding_model, chunk_records, chunk_embeddings, top_k)
    rerank_output = run_method_pipeline(question_text, "rerank", embedding_model, chunk_records, chunk_embeddings, top_k)

    compare_column_1, compare_column_2, compare_column_3, compare_column_4 = st.columns(4)

    with compare_column_1:
        st.write("### Basic")
        if len(basic_output["final_results"]) > 0:
            st.write(basic_output["final_results"][0]["text"])

    with compare_column_2:
        st.write("### Rewrite")
        if len(rewrite_output["final_results"]) > 0:
            st.write(rewrite_output["final_results"][0]["text"])

    with compare_column_3:
        st.write("### HyDE")
        if len(hyde_output["final_results"]) > 0:
            st.write(hyde_output["final_results"][0]["text"])

    with compare_column_4:
        st.write("### Rerank")
        if len(rerank_output["final_results"]) > 0:
            st.write(rerank_output["final_results"][0]["text"])

with tab_scores:
    st.subheader("Similarity Scores")

    query_for_scores = pipeline_output["query_used_for_retrieval"]
    query_embedding = embedding_model.encode(query_for_scores)

    all_scores = []

    for chunk_embedding in chunk_embeddings:
        similarity_score = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
        all_scores.append(similarity_score)

    figure = plt.figure()
    plt.plot(all_scores)
    plt.title("Similarity score for each chunk")
    plt.xlabel("Chunk index")
    plt.ylabel("Similarity score")
    st.pyplot(figure)

st.markdown("---")
st.subheader("Teaching Prompts")
st.write("- Compare the evidence-based answer and the generated answer.")
st.write("- Try a vague question and then a specific question.")
st.write("- Change chunk size and see whether the top chunk changes.")
st.write("- Compare Basic, Rewrite, HyDE, and Rerank.")