import os
import json
import pandas as pd  # 표 출력용(선택)
import rag  # 같은 폴더의 rag.py 사용
import pickle
from pathlib import Path
from report import show_retrieval, show_workcard, save_card_md, save_card_html

INDEX_DIR   = os.getenv("TM_INDEX_DIR",  r"C:\Dev\KorailPJT\index")
PARSED_DIR  = os.getenv("TM_PARSED_DIR", r"C:\Dev\KorailPJT\parsed")

# ====== 하드코딩 설정 ======
STEP4_QUERY = "전동기 베어링 과열"
STEP5_QUERY = "전동기 과열"
TOPK_RETRIEVE = 5
TOPK_CARD = 8
OUTPUT_CARD_PATH = "outputs/cards/tm_card.json"

def main(force_parse=False):
    print("\n=== STEP1: PDF 파싱 ===")
    pdf_files = list(Path(rag.DATA_DIR).glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in: {rag.DATA_DIR}")
    
    pages = {}
    to_parse = []
    for pdf in pdf_files:
        pkl_path = Path(PARSED_DIR) / f"{pdf.name}_pages.pkl"
        # PDF의 수정 시간과 .pkl 파일의 수정 시간 비교
        if not force_parse and pkl_path.exists() and pdf.stat().st_mtime <= pkl_path.stat().st_mtime:
            print(f"[STEP1] 재사용: {pdf.name}")
            pages[pdf.name] = pickle.load(open(pkl_path, "rb"))
        else:
            to_parse.append(pdf)
    
    if to_parse:
        print(f"[STEP1] 새로 파싱: {len(to_parse)} files")
        new_pages = rag.step1_parse_pdf(save=True)  # 새 파일만 파싱
        pages.update(new_pages)
    else:
        print(f"[STEP1] 모든 PDF의 기존 파싱 결과 재사용")
    
    print(f"[STEP1] pages: {len(pages)} files processed\n")


    print("\n=== STEP2: 섹션/표/안전 청크 생성 ===")
    chunks = rag.step2_build_chunks(save=True)

    print("\n=== STEP3: TF-IDF 인덱스 생성 ===")
    idx_dir = rag.step3_build_index(save=True)

    print("\n=== STEP4: 검색 데모 ===")
    df = rag.step4_retrieve(STEP4_QUERY, top_k=TOPK_RETRIEVE)
    print(f"[STEP4] query: {STEP4_QUERY}")
    show_retrieval(df)

    print("\n=== STEP5: 작업카드(JSON) 생성 ===")
    card = rag.step5_generate_workcard(STEP5_QUERY, top_k=TOPK_CARD)
    os.makedirs(os.path.dirname(OUTPUT_CARD_PATH), exist_ok=True)
    with open(OUTPUT_CARD_PATH, "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=2)
    print(f"[STEP5] saved card: {OUTPUT_CARD_PATH}")
    show_workcard(card)  # 🔵 콘솔 카드 출력    
    save_card_md(card, "outputs/cards/tm_card.md")       # 🟢 Markdown 저장 (VS Code에서 미리보기 좋음)
    save_card_html(card, "outputs/cards/tm_card.html")   # 🟢 HTML 저장 (브라우저로 열기)


    print("\n✅ Done. (5차 실행 완료)")

if __name__ == "__main__":
    main() 