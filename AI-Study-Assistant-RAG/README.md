# AI Study Assistant using RAG (Embeddings + Groq)

A PDF-based AI study assistant built with FastAPI, Sentence Transformers, ChromaDB and Groq. Upload up to five PDF chapters, extract and split their text into chunks, create local embeddings, retrieve relevant passages, and generate grounded answers with Groq.

## Features
- Upload multiple PDF chapters (assignment requirement: 5 PDFs)
- Extract text with PyPDF
- Split text into overlapping chunks
- Generate embeddings locally with `all-MiniLM-L6-v2`
- Persist vectors locally in ChromaDB
- Semantic retrieval of the 4 most relevant chunks
- Groq-powered question answering grounded in retrieved PDF context
- Simple responsive web interface, document list, answer sources, copy answer
- Clear message when context does not contain an answer

## Tech Stack
Python, FastAPI, PyPDF, Sentence Transformers, ChromaDB, Groq API, HTML, CSS, JavaScript.

## Requirements
- Python 3.10 or 3.11 recommended
- Internet connection for installing packages, downloading the embedding model on first run, and using Groq
- Groq API key

## Setup (Windows / VS Code PowerShell)
Open the project folder in VS Code, then run:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and replace `your_groq_api_key_here` with your own Groq API key. Never upload `.env` to GitHub.

## Run
```powershell
uvicorn api:app --reload
```
Open http://127.0.0.1:8000 in your browser. API documentation is at http://127.0.0.1:8000/docs and health check is http://127.0.0.1:8000/health.

## How to use
1. Select five text-based PDF chapters in the upload area.
2. Click **Upload & Index** and wait for indexing to finish.
3. Ask a question about the uploaded content.
4. Read the generated answer and review the source filename/chunk chips.
5. Uploading a same-named PDF replaces its indexed chunks.

Scanned/image-only PDFs may not contain extractable text; use text-based PDFs or OCR them before uploading.

## RAG workflow
1. **Load:** PDF files are uploaded and text is extracted.
2. **Chunk:** Extracted text is split into overlapping segments to preserve context.
3. **Embed:** Sentence Transformers converts each chunk into a numerical vector.
4. **Store:** ChromaDB stores vectors, text and filename/chunk metadata on disk in `chroma_db/`.
5. **Retrieve:** A question is embedded and ChromaDB returns the four nearest chunks.
6. **Generate:** Retrieved context and the question are sent to Groq, which generates an answer instructed to stay within the context.

## Project Structure
```text
AI-Study-Assistant-RAG/
├── api.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── static/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── uploaded_pdfs/   # created automatically; ignored by Git
└── chroma_db/       # created automatically; ignored by Git
```

## Example questions
- What are the main concepts in this chapter?
- Explain the difference between a list and a tuple.
- Summarize the key points from Chapter 2.
- Create short revision notes from the uploaded material.

Answers depend on the contents of the uploaded PDFs. If the documents do not contain the answer, the assistant is instructed to say so rather than invent information.

## Submission checklist
- [ ] Upload five PDF chapters
- [ ] Confirm the API status is online
- [ ] Ask at least three different questions
- [ ] Save screenshots of PDF upload, embeddings/indexing, and AI responses
- [ ] Include your GitHub repository link in the submission
- [ ] Keep `.env`, `venv`, `chroma_db`, and private PDFs out of GitHub

## Notes
The app runs locally. For deployment, configure a persistent vector store and securely provide the Groq API key through the host's environment variables. Do not commit API keys or private study documents.
