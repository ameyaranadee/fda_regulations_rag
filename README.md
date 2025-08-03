# Ask questions on FDA regulations

## Setup

1. Ensure your environment is set up:

- `OPENAI_API_KEY` set in your environment
- Dependencies installed via `pip install -r requirements.txt`

2. Store your PDFs under `data/raw`.

## Commands

### 1. Upload PDFs to a new vector store

```
python3 scripts/main.py upload --pdf_dir data/raw --store_name fda_regulations_store
```

This creates a new vector store and uploads all PDFs.
Note: Copy the printed vector_store_id from the output, you’ll need it for querying.

### 2. Run a raw vector search

```
python3 scripts/main.py search --query "What are FDA drug approval requirements?" --store_id <store_id>
```

This will return relevant text chunks with similarity scores.

### 3. Generate a response for your query:

python3 scripts/main.py ask --query "What are FDA drug approval requirements?" --store_id <store_id>
