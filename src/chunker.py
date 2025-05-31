import nltk
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config import get_settings
import logging
import re
from .logging_utils import setup_json_logger

setup_json_logger("logs/server2_rag.log", "server2-rag")
logger = logging.getLogger("server2_rag")

nltk.download("punkt", quiet=True, raise_on_error=True)

def chunk_sentences(text: str) -> list[str]:
    """
    문장 단위 분할 후 의미 단위로 재그룹.
    한국어와 영어 혼합 텍스트에 최적화된 청킹 적용
    """
    logger.info({
        "event": "chunk_split_start",
        "input_length": len(text)
    })
    
    settings = get_settings()
    
    # 한국어 특화 패턴: 마침표, 물음표, 느낌표 등의 문장 종결 부호를 기준으로 나누되
    # 숫자 사이의 점(.)은 분리하지 않도록 예외 처리
    # (?<!\d\.)(?<=[\.\?\!])\s+(?=[가-힣A-Za-z]) 패턴은 "숫자."으로 끝나지 않고, 문장 부호로 끝나며, 그 뒤에 공백이 있고, 그 다음에 한글이나 영문이 오는 경우를 찾습니다.
    
    # 1. 텍스트 전처리: 불필요한 연속 공백 제거
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 2. 문장 분리
    # 한국어 문장 종결 패턴 (마침표, 물음표, 느낌표 뒤에 공백이 오고 그 다음에 한글/영어가 오는 경우)
    korean_sent_pattern = r'(?<!\d\.)(?<=[\.\?\!])\s+(?=[가-힣A-Za-z])'
    
    # 패턴으로 텍스트 분리
    sentences = []
    
    if re.search(korean_sent_pattern, text):
        # 한국어 패턴이 발견된 경우
        raw_sentences = re.split(korean_sent_pattern, text)
        
        # 분리된 문장에 구두점 다시 붙이기 (split에서 제외된 부분)
        for i, sent in enumerate(raw_sentences[:-1]):
            if not sent.rstrip().endswith(('.', '?', '!')):
                sentences.append(sent.strip() + '.')
            else:
                sentences.append(sent.strip())
        sentences.append(raw_sentences[-1].strip())
    else:
        # 한국어 패턴이 발견되지 않은 경우 nltk 사용
        sentences = nltk.sent_tokenize(text)
    
    # 3. 문장을 의미 단위로 재그룹
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ".", "?", "!", "。", "？", "！", ";", ":", "，", ","],
    )
    
    # 문장들을 합쳐서 splitter에 전달
    joined_text = " ".join(sentences)
    chunks = splitter.split_text(joined_text)
    
    # 로깅
    logger.info({
        "event": "chunk_split_completed",
        "num_chunks": len(chunks),
        "chunk_sizes": [len(chunk) for chunk in chunks],
        "avg_chunk_size": sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0
    })
    
    # 각 청크를 로그에 기록
    for i, chunk in enumerate(chunks):
        logger.info({
            "event": "chunk_created",
            "chunk_id": i,
            "chunk_size": len(chunk),
            "chunk_preview": chunk[:100] + ("..." if len(chunk) > 100 else "")
        })
    
    return chunks
