import sys, json, time
from src.config import get_settings
from src.graph import run_pipeline

def main() -> None:
    text = sys.stdin.read()          # 단일 문자열 STDIN 입력
    start = time.time()
    result = run_pipeline(text)
    result["_elapsed_total"] = time.time() - start
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
