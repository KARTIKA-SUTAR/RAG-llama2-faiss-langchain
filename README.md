# Retrieval-Augmented Generation (RAG) with LangChain, FAISS and Llama-2

An end-to-end RAG pipeline in a single Google Colab notebook. It loads a long text document, splits it into chunks, indexes them in a FAISS vector store, and answers questions with a locally run **Llama-2-13B-chat** model using only the retrieved passages as context.

## Objective

Show how to build a question-answering system over a document that is far too large to fit into an LLM prompt: retrieve the few passages that matter, then let the LLM answer from them.

## How it works

```
Text file ─► TextLoader ─► RecursiveCharacterTextSplitter ─► Embeddings (bge-base-en-v1.5)
                                                                     │
Question ─► Embedding ─► FAISS similarity search (top-2 chunks) ◄────┘
                                  │
                                  ▼
                 Prompt = chunks + question ─► Llama-2-13B-chat ─► Answer
```

| Step | Detail |
|---|---|
| Load | `TextLoader` reads the file as 1 document (about 1.29 million characters) |
| Split | `RecursiveCharacterTextSplitter`, chunk size 2000, overlap 200, giving 715 chunks |
| Embed | `BAAI/bge-base-en-v1.5`, 768-dimensional vectors |
| Store | FAISS index with 715 vectors |
| Generate | `TheBloke/Llama-2-13B-chat-GGUF` (`Q5_K_M`) via `llama-cpp-python`, temperature 0.01, context 4096, 43 GPU layers |
| Chain | LangChain `RetrievalQA`, `stuff` chain type, retriever `k=2` |

## Sample result

**Question:** How often does the company review inventory, and what is considered in this inventory calculation?

**Answer:** The company reviews inventory quarterly. This calculation considers demand forecasts, product life cycle status, product development plans, current sales levels, and component cost trends.

Both retrieved chunks contain this information, so the answer is grounded in the source text.

## Repository contents

| File | Description |
|---|---|
| `RAG_Llama2_FAISS_Notebook.ipynb` | The notebook, with an objective and interpretation for each step and saved outputs |
| `AAPL-MDA.txt` | Text dataset used as the knowledge source |
| `README.md` | This file |

## Requirements

- Google Colab with a **GPU runtime (T4)**
- Roughly 10 GB of free disk space (the Llama model file is about 9.23 GB)
- Internet access to download models from the Hugging Face Hub

Main libraries (pinned in the notebook): `langchain==0.3.27`, `langchain-community==0.3.29`, `langchain-huggingface==0.3.1`, `llama-cpp-python==0.2.73` (built with CUDA), `faiss-cpu`, `sentence-transformers`.

## How to run

1. Open the notebook in Google Colab and set **Runtime > Change runtime type > T4 GPU**.
2. Upload `AAPL-MDA.txt` to your Google Drive (default path used in the notebook: `MyDrive/Dataset/GenAIDataset/AAPL-MDA.txt`), or edit the path in the data-loading cell.
3. Run the install cells, then **restart the runtime** when prompted.
4. Run the remaining cells in order. The first run downloads the embedding model and the Llama model, which takes several minutes.
5. Change the `query` variable to ask your own question about the document.

## Notes and limitations

- The answer quality depends on retrieval: only the top 2 chunks are passed to the model, so questions whose answer is spread across many passages may be answered incompletely.
- The legacy `langchain.*` import paths and `SentenceTransformerEmbeddings` show deprecation warnings; migrating to `langchain-huggingface` is the recommended next step.
- `llama-cpp-python` is compiled from source with CUDA, so the install step takes a few minutes.
