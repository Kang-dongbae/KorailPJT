import os, json
import pandas as pd  # 표 출력용(선택)
import rag  # 같은 폴더의 rag.py 사용
import pickle

INDEX_DIR   = os.getenv("TM_INDEX_DIR",  "C:\Dev\KorailPJT\index")
PARSED_DIR  = os.getenv("TM_PARSED_DIR", "C:\Dev\KorailPJT\parsed")

# ====== 하드코딩 설정 ======
STEP4_QUERY = "베어링 과열 그리스 Mobilith SHC 100 5 g"
STEP5_QUERY = "절연저항 1000 V 50 MΩ 전동기 과열"
TOPK_RETRIEVE = 5
TOPK_CARD = 8
OUTPUT_CARD_PATH = "outputs/cards/tm_card.json"

def main():
    print("\n=== STEP1: PDF 파싱 ===")
    pages = rag.step1_parse_pdf(save=True)
    print(f"[STEP1] pages: {pages}")

    print("\n=== STEP2: 섹션/표/안전 청크 생성 ===")
    chunks = rag.step2_build_chunks(save=True)
    print(f"[STEP2] chunks: {chunks}")

    print("\n=== STEP3: TF-IDF 인덱스 생성 ===")
    idx_dir = rag.step3_build_index(save=True)
    print(f"[STEP3] index built at: {idx_dir}")

    print("\n=== STEP4: 검색 데모 ===")
    df = rag.step4_retrieve(STEP4_QUERY, top_k=TOPK_RETRIEVE)
    print(f"[STEP4] query: {STEP4_QUERY}")
    try:
        # 보기 좋게 표로 출력
        from tabulate import tabulate
        print(tabulate(df, headers="keys", tablefmt="psql", showindex=False))
    except Exception:
        # tabulate 미설치 시 기본 출력
        print(df.to_string(index=False))

    print("\n=== STEP5: 작업카드(JSON) 생성 ===")
    card = rag.step5_generate_workcard(STEP5_QUERY, top_k=TOPK_CARD)
    os.makedirs(os.path.dirname(OUTPUT_CARD_PATH), exist_ok=True)
    with open(OUTPUT_CARD_PATH, "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=2)
    print(f"[STEP5] saved card: {OUTPUT_CARD_PATH}")


    print("\n✅ Done. (5차 실행 완료)")

if __name__ == "__main__":
    main()