import os
import sys
from pathlib import Path
from loguru import logger

class Config:
    SEED = 42
    ALLOWED_FILE_EXTENSIONS = set(['.pdf', '.md', '.txt'])

    class Model:
        # NAME = "deepseek-r1:1.5b"
        NAME = "gemma3:1b"
        TEMPERATURE = 0.6

    class Preprocessing:
        CHUNK_SIZE = 2048
        CHUNK_OVERLAP = 128
        # EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"      
        EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"      
        RERANKER = "ms-marco-MiniLM-L-12-v2"       
        LLM = "llama3.2"
        CONTEXUALIZE_CHUNKS = True
        N_SEMENTIC_RESULTS = 5
        N_BM25_RESULTS = 5

    class Chatbot:
        N_CONTEXT_RESULTS = 3

    class VectorDB:
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
        PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
        PINECONE_EMBEDDING_DIMENSION = os.getenv("-PINECONE_EMBEDDING_DIMENSION")
        PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE")

    class Path:
        APP_HOME = Path(os.getenv("APP_HOME", Path(__file__).parent.parent))      # Path to the root of the project
        DATA_DIR = APP_HOME / "data"                                              # Path to the data directory

def configure_logging():
    config = {
        "handlers": [
            {
                "sink": sys.stdout,     # Log to stdout (console) ; beneficial for docker container
                "colorize": True,       # Colorize the output
                "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}",      # Format of the log message
            },
        ]
    }
    return logger.configure(**config)
