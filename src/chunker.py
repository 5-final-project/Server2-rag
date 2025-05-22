import nltk
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config import get_settings
import logging
from .logging_utils import setup_json_logger

setup_json_logger("logs/server2_rag.log", "server2-rag")
logger = logging.getLogger("server2_rag")

nltk.download("punkt", quiet=True, raise_on_error=True)

def chunk_sentences(text: str) -> list[str]:
    """문장 단위 분할 후 의미 단위(토큰 수 기준)로 재그룹."""
    logger.info({
        "event": "chunk_split_start",
        "input_length": len(text)
    })
    settings = get_settings()
    sentences = nltk.sent_tokenize(text, language="english")  # 한/영 혼합에도 안전
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ".", "?", "!"],
    )
    chunks = splitter.split_text(" ".join(sentences))
    logger.info({
        "event": "chunk_split_completed",
        "num_chunks": len(chunks)
    })
    return chunks
