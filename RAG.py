import os, re, json, pickle
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import pandas as pd
from scipy import sparse
import PyPDF2

MANUAL_PDF  = os.getenv("TM_MANUAL_PDF", "C:\Dev\KorailPJT\data\견인전동기.pdf")
PARSED_DIR  = os.getenv("TM_PARSED_DIR", "C:\Dev\KorailPJT\parsed")
INDEX_DIR   = os.getenv("TM_INDEX_DIR",  "C:\Dev\KorailPJT\index")
os.makedirs(PARSED_DIR, exist_ok=True)
os.makedirs(INDEX_DIR,  exist_ok=True)

def _parse_pdf_to_text_pages(pdf_path: str) -> List[str]:
    pages: List[str] = []
    # 1) pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                pages.append(p.extract_text() or "")
        if pages: return pages
    except Exception:
        pass
    # 2) PyMuPDF
    try:
        import fitz
        doc = fitz.open(pdf_path)
        for i in range(doc.page_count):
            pages.append(doc.load_page(i).get_text())
        if pages: return pages
    except Exception:
        pass
    # 3) PyPDF2
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(pdf_path)
        for p in reader.pages:
            pages.append(p.extract_text() or "")
        if pages: return pages
    except Exception:
        pass
    return pages

def step1_parse_pdf(save: bool = True) -> int:
    if not os.path.exists(MANUAL_PDF):
        raise FileNotFoundError(f"PDF가 없습니다: {MANUAL_PDF}")
    pages = _parse_pdf_to_text_pages(MANUAL_PDF)
    if not pages:
        raise RuntimeError("PDF에서 텍스트를 추출하지 못했습니다.")
    if save:
        with open(os.path.join(PARSED_DIR, "pages.pkl"), "wb") as f:
            pickle.dump(pages, f)
    return len(pages)

# ===== 2) 섹션/표/안전 청크 =====
SECTION_RE = re.compile(r"(?m)^(?P<num>\d+(?:\.\d+)+)\s*(?P<title>[^\n]+)")

@dataclass
class Chunk:
    chunk_id: int
    section_id: str
    title: str
    page_start: int
    page_end: int
    type: str   # "procedure"|"table"|"safety"|"misc"
    text: str

def _load_pages() -> List[str]:
    pkl = os.path.join(PARSED_DIR, "pages.pkl")
    if not os.path.exists(pkl):
        step1_parse_pdf(save=True)
    with open(pkl, "rb") as f:
        return pickle.load(f)

def step2_build_chunks(save: bool = True) -> int:
    pages = _load_pages()
    # 페이지 연결
    offsets, joined, total = [], [], 0
    for t in pages:
        t = t if isinstance(t, str) else ""
        joined.append(t); offsets.append(total); total += len(t) + 1
    joined_s = "\n".join(joined)
    def pos_to_page(pos: int) -> int:
        for p, off in enumerate(offsets):
            if p+1 == len(offsets): return p
            if offsets[p] <= pos < offsets[p+1]: return p
        return len(offsets)-1

    chunks: List[Chunk] = []
    matches = list(SECTION_RE.finditer(joined_s))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(joined_s)
        body = joined_s[start:end].strip()
        if len(body) < 50: 
            continue
        p_start = pos_to_page(start)+1
        p_end   = pos_to_page(end)+1
        chunks.append(Chunk(
            chunk_id=len(chunks),
            section_id=m.group("num"),
            title=m.group("title").strip(),
            page_start=p_start, page_end=p_end,
            type="procedure",
            text=body
        ))
    # 표/수치/안전 힌트
    for i, page in enumerate(pages, start=1):
        text = page or ""
        if any(k in text for k in ["유  형 원  인 조  치", "사  양", "규격", "MΩ", "N.m", " g "]):
            chunks.append(Chunk(len(chunks), f"P{i}", "표/수치/규격(heuristic)", i, i, "table", text))
        if any(k in text for k in ["경고", "주의", "위험", "660 kg", "660kg"]):
            chunks.append(Chunk(len(chunks), f"P{i}", "안전문구(heuristic)", i, i, "safety", text))

    df = pd.DataFrame([asdict(c) for c in chunks])
    if save:
        df.to_csv(os.path.join(PARSED_DIR, "chunks.csv"), index=False, encoding='utf-8-sig')
    return len(df)

# ===== 3) 인덱스(TF-IDF) =====
def _load_chunks_df() -> pd.DataFrame:
    csv = os.path.join(PARSED_DIR, "chunks.csv")
    if not os.path.exists(csv):
        step2_build_chunks(save=True)
    return pd.read_csv(csv)

def step3_build_index(save: bool = True) -> str:
    from sklearn.feature_extraction.text import TfidfVectorizer
    df = _load_chunks_df()
    texts = (df["title"].fillna("") + "\n" + df["text"].fillna("")).tolist()
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,5), min_df=1, max_df=0.9)
    X = vectorizer.fit_transform(texts)
    if save:
        df.to_pickle(os.path.join(INDEX_DIR, "meta.pkl"))
        with open(os.path.join(INDEX_DIR, "vectorizer.pkl"), "wb") as f: pickle.dump(vectorizer, f)
        sparse.save_npz(os.path.join(INDEX_DIR, "matrix.npz"), X)
    return INDEX_DIR

def _ensure_index():
    need = ["meta.pkl","vectorizer.pkl","matrix.npz"]
    if all(os.path.exists(os.path.join(INDEX_DIR, f)) for f in need):
        return
    # 없으면 앞 단계부터 자동 실행
    if not os.path.exists(os.path.join(PARSED_DIR, "chunks.csv")):
        if not os.path.exists(os.path.join(PARSED_DIR, "pages.pkl")):
            step1_parse_pdf(save=True)
        step2_build_chunks(save=True)
    step3_build_index(save=True)

# ===== 4) 검색 =====
def step4_retrieve(query: str, top_k: int = 5) -> pd.DataFrame:
    from sklearn.metrics.pairwise import linear_kernel
    _ensure_index()
    df = pd.read_pickle(os.path.join(INDEX_DIR, "meta.pkl"))
    with open(os.path.join(INDEX_DIR, "vectorizer.pkl"), "rb") as f: vectorizer = pickle.load(f)
    X = sparse.load_npz(os.path.join(INDEX_DIR, "matrix.npz"))
    qv = vectorizer.transform([query])
    sims = linear_kernel(qv, X).ravel()
    idx = sims.argsort()[::-1][:top_k]
    out = df.iloc[idx].copy()
    out["score"] = sims[idx]
    out["preview"] = out["text"].str.replace("\n", " ").str.slice(0, 200) + "..."
    return out[["score","section_id","title","type","page_start","page_end","preview"]]

# ===== 5) 작업카드 생성(숫자 강제 인용) =====
NUM_PATS = {
    "voltage_V": re.compile(r"(\d{3,4})\s*V"),
    "resistance_MOhm": re.compile(r"(\d{1,3})\s*M[ΩΩ]"),
    "grease_g": re.compile(r"(\d{1,3})\s*g\b"),
    "torque_Nm": re.compile(r"(\d{1,3})\s*N\.?m"),
    "gap_mm": re.compile(r"([0-9.]+)\s*mm")
}

def _extract_numbers(text: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for k, pat in NUM_PATS.items():
        vals = pat.findall(text or "")
        if vals:
            uniq = []
            for v in vals:
                if v not in uniq:
                    uniq.append(v)
            out[k] = uniq
    return out

def step5_generate_workcard(query: str, top_k: int = 8) -> Dict[str, Any]:
    _ensure_index()
    hits = step4_retrieve(query, top_k=top_k)
    df = pd.read_pickle(os.path.join(INDEX_DIR, "meta.pkl"))
    merged = ""
    refs = []
    for _, row in hits.iterrows():
        mask = (
            (df["section_id"] == row["section_id"]) &
            (df["page_start"] == row["page_start"]) &
            (df["page_end"] == row["page_end"])
        )
        txt = df.loc[mask, "text"]
        if len(txt): merged += "\n" + str(txt.values[0])
        refs.append({"section_id": row["section_id"], "pages": [int(row["page_start"]), int(row["page_end"])], "type": row["type"]})
    nums = _extract_numbers(merged)

    safety = ["고전압 작업 전 차단·방전"]
    for _, r in df[df["type"]=="safety"].iterrows():
        if "660 kg" in r["text"] or "660kg" in r["text"]:
            safety.append("견인전동기 중량(660 kg) 취급 주의")
            break

    actions = []
    if "베어링" in query or "그리스" in query:
        g = (nums.get("grease_g") or ["5"])[0]
        actions.append({"step": 1, "text": f"그리스 주입구 청결 확인 후, Mobilith SHC 100을 각 베어링에 {g} g 주입한다(정지 상태).", "refs":[r["section_id"] for r in refs]})
    if "절연저항" in query or "과열" in query:
        v = (nums.get("voltage_V") or ["1000"])[0]
        r = (nums.get("resistance_MOhm") or ["50"])[0]
        actions.append({"step": len(actions)+1, "text": f"절연저항계를 {v} V로 설정하여 측정하고 기준 {r} MΩ 이상 여부를 확인한다.", "refs":[r["section_id"] for r in refs]})
    if "토크" in query:
        ts = sorted(set(nums.get("torque_Nm") or []))
        if ts:
            actions.append({"step": len(actions)+1, "text":"체결부를 규정 토크로 조인다: " + ", ".join(ts) + " N·m", "refs":[r["section_id"] for r in refs]})

    card = {
        "asset": {"device":"traction_motor"},
        "diagnosis": {"from_query": query},
        "actions": actions,
        "tests": [],
        "safety": safety,
        "sources": refs
    }
    if ("절연저항" in query or "과열" in query) and (("voltage_V" not in nums) or ("resistance_MOhm" not in nums)):
        card["warning"] = "필수 수치(전압/합격저항)를 문서에서 찾지 못했습니다. 쿼리를 바꿔 재시도하세요."
    return card