import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

PDF_DIR = "data/raw/"
VECTOR_STORE_DIR = "data/vector_stores/"

DEFAULT_STORE_NAME = "fda_regulations_store"
MAX_WORKERS = 10

SUPPORTED_EXTENSIONS = ['.pdf']

DEFAULT_TOP_K = 5

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")