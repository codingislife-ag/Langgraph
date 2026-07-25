# Conversation export — understanding 13_rag.ipynb (PyPDFLoader → Splitter → Embeddings → FAISS)

Context: working in `/Users/arpitgupta/Documents/Langgraph/13_rag.ipynb`, a RAG pipeline using
`langchain`, `langchain-openai` (`ChatOpenAI`, `OpenAIEmbeddings`), `langchain-community`
(`PyPDFLoader`, `FAISS`), and `langgraph`. PDF is `intro-to-ml.pdf` (392 pages).

---

## 1. Loading the PDF

```python
loader = PyPDFLoader('intro-to-ml.pdf')
docs = loader.load()
```

- `PyPDFLoader` (from `langchain_community.document_loaders`) just constructs a loader object
  pointed at the file path. Nothing is read from disk yet — it's lazy.
  - **Datatype of `loader`**: `PyPDFLoader` instance.
- `.load()` is where the work happens. It uses the `pypdf` library to open the PDF and extract
  text **page by page**. For each page it creates one `Document` object containing:
  - `page_content`: the extracted plain text of that page (formatting/images are lost — text only).
  - `metadata`: dict with info like `{'source': 'intro-to-ml.pdf', 'page': 0, 'total_pages': 392, ...}`.
  - **Datatype of `docs`**: `list[Document]` — confirmed in notebook: `len(docs)` → `392`,
    `type(docs)` → `<class 'list'>`, `type(docs[0])` → `<class 'langchain_core.documents.base.Document'>`.

Mental model: `docs[0].page_content` = text of page 1, `docs[0].metadata` = its metadata, etc.
One `Document` per PDF page — not per paragraph, not per whole PDF.

---

## 2. Splitting into chunks

```python
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
```

- `RecursiveCharacterTextSplitter` tries a hierarchy of separators (`\n\n`, then `\n`, then `" "`,
  then char-by-char) so it prefers breaking at paragraph/sentence boundaries rather than mid-word.
  - `chunk_size=1000`: target max ~1000 characters per chunk.
  - `chunk_overlap=200`: each chunk repeats the last 200 characters of the previous chunk, so a
    sentence split across a boundary still appears whole in at least one chunk (important for
    retrieval quality).
  - **Datatype**: `RecursiveCharacterTextSplitter` instance.
- `.split_documents(docs)` re-splits each page's `page_content` into smaller pieces, creating a
  **new** `Document` per piece and copying over the original metadata (`source`, `page`, etc.).
  - Confirmed in notebook: 392 page-docs → **973 chunks**.
  - **Datatype of `chunks`**: `list[Document]` — same type as `docs`, just more/shorter elements.
  - Example from notebook (`chunks[0]`):
    ```
    page_content='Andreas C. Müller & Sarah Guido\nIntroduction to \nMachine \nLearning  \nwith P y t h o n ...'
    metadata={'producer': '...', 'creator': '...', 'creationdate': '...', 'author': 'Andreas C. Müller and Sarah Guido',
              'title': 'Introduction to Machine Learning with Python', 'source': 'intro-to-ml.pdf',
              'total_pages': 392, 'page': 0, 'page_label': 'Cover'}
    ```

---

## 3. Embeddings + FAISS vector store

```python
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
vector_store = FAISS.from_documents(chunks, embedding=embeddings)
```

- `OpenAIEmbeddings` is just a client wrapper — a *function* that converts text → vector by
  calling OpenAI's embeddings API. No data is embedded yet at this line.
  - **Datatype**: `OpenAIEmbeddings` instance (`langchain_core.embeddings.Embeddings` subclass).
- `FAISS.from_documents(chunks, embedding=embeddings)` is the expensive step:
  - For every one of the 973 chunks, it calls `embeddings.embed_documents(...)`, sending the
    chunk's text to OpenAI's API and getting back a **1536-dimensional vector**
    (dimensionality of `text-embedding-3-small`).
  - It then builds a **FAISS index** (a data structure for fast similarity search over vectors)
    and stores each vector alongside a reference back to its original `Document`.
  - **Datatype**: `FAISS` instance (confirmed: `type(vector_store)` →
    `langchain_community.vectorstores.faiss.FAISS`).

### Why is it called an "index"?

Borrowed from classic databases: a DB index lets you find a row fast without scanning the whole
table. A **vector index** does the same for vectors — given a query vector, find the nearest ones
fast, without necessarily comparing against every vector one-by-one. FAISS offers different index
*types*:
- `IndexFlatL2` — brute-force **exact** search (compares against every vector). **This is the
  default used by `FAISS.from_documents()`.**
- `IndexIVFFlat`, `IndexHNSW`, etc. — approximate methods that skip most comparisons, used at
  million-vector scale.

With 973 chunks, exact/flat search is plenty fast — no approximation needed.

### Where are the vectors actually stored? (cloud vs local)

**Entirely in RAM, inside your local Python process.** Nothing goes to any cloud vector database.

The *only* network call in the whole pipeline is: for each chunk's text, `OpenAIEmbeddings` sends
the text to OpenAI's API and gets back the vector (list of floats). Once that response returns,
the vector lives locally — FAISS itself never talks to any external service.

Restarting the Jupyter kernel destroys `vector_store` (it's not persisted anywhere). To persist it
explicitly:
```python
vector_store.save_local("my_index")   # writes local files: index.faiss + index.pkl
```
Both are **local disk files**, not cloud storage.

### Internal structure of `vector_store` — how the reference is maintained

```
vector_store  (FAISS object)
 ├── .index                   → an actual faiss.IndexFlatL2 object
 │                               conceptually a matrix of shape (973, 1536):
 │                               973 vectors, each 1536 floats, stored contiguously
 │
 ├── .docstore                → InMemoryDocstore: {doc_id (uuid str) -> Document}
 │                               holds the REAL data: page_content + metadata
 │                               (the same stuff shown in chunks[0] above)
 │
 └── .index_to_docstore_id     → dict: {0: 'uuid-a', 1: 'uuid-b', ..., 972: 'uuid-z'}
                                  maps FAISS's internal integer position -> docstore's uuid key
```

Deliberate separation:
- FAISS's index only ever holds raw numbers (vectors) — no knowledge of your text.
- The docstore holds the human-readable `Document`s (text + metadata), completely separately.
- `index_to_docstore_id` is the glue tying a numeric vector position back to an actual PDF chunk.

### What happens on search — `retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':4})`

When `retriever.invoke("some question")` runs:
1. Query text → sent to OpenAI → becomes a 1536-dim vector (same embedding model, so comparable).
2. FAISS compares that vector against all 973 stored vectors (L2/Euclidean distance) and returns
   the **top `k=4` closest positions** (e.g. `[57, 812, 3, 401]`) with their distances.
3. LangChain looks up `index_to_docstore_id[57]` → uuid → `docstore[uuid]` → returns the original
   `Document` (real `page_content` + `metadata`, e.g. `{'source': 'intro-to-ml.pdf', 'page': 187}`).
4. Result: `list[Document]` of length 4 — the most semantically similar chunks to the query.

This feeds directly into the notebook's `rag_tool`:
```python
result = retriever.invoke(query)
context = [doc.page_content for doc in result]
metadata = [doc.metadata for doc in result]
```

---

## Full pipeline recap

```
PDF file (intro-to-ml.pdf, 392 pages)
  → docs          : list[Document]   (392 items, 1 per page, full page text)
  → chunks        : list[Document]   (973 items, ~1000 chars each, 200-char overlap)
  → embeddings    : Embeddings object (text -> 1536-dim vector, on demand via OpenAI API)
  → vector_store  : FAISS object      (973 vectors indexed locally for similarity search,
                                        linked to original chunk Documents via docstore + id map)
  → retriever     : takes a query, returns top-k (k=4) most similar Document chunks
```
