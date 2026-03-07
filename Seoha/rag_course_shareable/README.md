# RAG Course Shareable Pack

This package is designed so that other people can use it easily in:

- local environments
- DataHub / JupyterHub

It contains:

```text
rag_course_shareable/
├── rag_tutorial_one_notebook.ipynb
├── app.py
├── requirements.txt
└── README.md
```

---
![Deme](./demo.mp4)
## Quick start


### Step 1. Open a terminal

### Step 2. Install libraries

```bash
python -m pip install -r requirements.txt
```

### Step 3. Run the app

Local machine:

```bash
streamlit run app.py --server.port 8502
```

DataHub / JupyterHub:

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

If port `8501` is busy, try:

```bash
streamlit run app.py --server.port 8502 --server.address 0.0.0.0
```

---

## Recommended learning order

1. Open the notebook first
2. Run the app second

The notebook teaches the ideas.
The app is the experiment lab.

---

## How to change models easily

The app is designed so that users can change models in two ways.

### A. Change models in the sidebar
Inside the app, users can change:

- embedding model
- retrieval method
- local GGUF model path
- API provider
- API model name
- generation settings

### B. Change the default values in `app.py`
At the top of `app.py`, there is a section called:

```python
# =========================
# EASY-TO-EDIT SETTINGS
# =========================
```

This is the main place to modify settings.

You can change:

- default local GGUF paths
- default DataHub GGUF path
- default embedding model list
- default API model names
- default example question
- default chunk size
- default overlap
- default top-k

---

## Local model path

The app tries these local model paths in order:

1. the DataHub path
2. a few common local paths
3. any path the user enters in the sidebar

The default DataHub path is:

```text
/home/jovyan/shared/qwen2-1_5b-instruct-q4_0.gguf
```

If that file does not exist, the app tries common local paths such as:

```text
~/Downloads/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
~/Downloads/qwen2-1_5b-instruct-q4_0.gguf
```

Users can always override the path in the sidebar.

---

## API models

API generation is optional.

The app currently supports:

- Groq
- OpenAI

The default model name lists are easy to edit at the top of `app.py`.

If you want to add or remove API models, edit the lists in the settings section.

---

## Other easy-to-modify parts

The easiest parts to modify are clearly marked with comments in `app.py`.

Look for comments such as:

- `EASY-TO-EDIT SETTINGS`
- `HOW TO CHANGE THIS`
- `OPTIONAL FEATURE`
- `LOCAL VS DATAHUB`

These comments are written for students and instructors.

---

## Troubleshooting

### Problem: `ModuleNotFoundError`
Run:

```bash
python -m pip install -r requirements.txt
```

Then restart the notebook kernel or rerun the app.

---

### Problem: `Port 8501 is already in use`
Use another port:

```bash
streamlit run app.py --server.port 8502 --server.address 0.0.0.0
```

---

### Problem: the app shows no results before upload
That is normal.

The app waits for uploaded files before retrieval starts.

---

### Problem: `No chunks were created from the uploaded files`
Possible causes:

- the file is empty
- the PDF is image-only and text extraction failed
- the text is too short or unreadable

Try a simple `.txt` file first.

Example:

```text
Students can apply for financial aid between March 1 and April 15.
```

---

### Problem: the local model does not load
Check whether the file exists.

DataHub example:

```bash
ls -lh /home/jovyan/shared/qwen2-1_5b-instruct-q4_0.gguf
```

Local example:

```bash
ls -lh ~/Downloads/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
```

If the local model is missing, retrieval still works.
Only local generation is skipped.

---

### Problem: generation does not work
Set:

- `LLM mode` → `None`

First confirm that retrieval works.
Then turn generation back on.

---

## Suggested first experiment

Question:

```text
When can students apply for financial aid?
```

Then try:

```text
When is aid?
```

Then compare:

- Basic
- Rewrite
- HyDE
- Rerank
