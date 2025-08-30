import os, re, json, pickle, warnings, argparse, multiprocessing as mp
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import pandas as pd
from scipy import sparse
from time import time

# ======================
# 0) PATHS & CONFIG
# ======================
DATA_DIR   = Path(os.getenv("TM_DATA_DIR",   r"C:\Dev\KorailPJT\data"))
PARSED_DIR = Path(os.getenv("TM_PARSED_DIR", r"C:\Dev\KorailPJT\parsed"))
INDEX_DIR  = Path(os.getenv("TM_INDEX_DIR",  r"C:\Dev\KorailPJT\index"))
REPORTS_DIR= Path(os.getenv("TM_REPORTS_DIR",r"C:\Dev\KorailPJT\reports"))
for d in (PARSED_DIR, INDEX_DIR, REPORTS_DIR): d.mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore", category=UserWarning)
CPU_WORKERS = max(2, mp.cpu_count()//2)

# 검색 가중치/옵션
TFIDF_WEIGHT = 0.6
BM25_WEIGHT  = 0.4
EMB_WEIGHT   = 0.25     # 임베딩 인덱스 사용 시 추가 가중치 (옵션)
RERANK_TOPN  = 50       # 크로스엔코더 재랭크 후보 수(옵션)

# 파일명 기반 초기 장치 매핑(프로젝트에 맞춰 자유 확장)
DEVICE_MAP_SEEDS = {
    "견인전동기": "traction_motor", "모터": "traction_motor", "냉각팬": "traction_motor",
    "주공기압축기":"main_compressor","콤프레서":"main_compressor",
    "팬터그래프":"pantograph","집전장치":"pantograph",
    "차축베어링":"axle_bearing","윤축":"axle_bearing",
    "출입문":"door_system","승강문":"door_system","도어":"door_system",
    "제동":"brake_system","캘리퍼":"brake_system","WSP":"brake_system",
    "냉난방":"hvac","공기조화":"hvac",
    "접지브러쉬":"ground_brush","CCTV":"cctv","무선통신":"radio",
}

# 동의어/표기 변형(인덱싱/검색 양쪽에 적용)
SYNONYMS = {
    "주공기압축기": ["콤프레서","공기압축기","메인콤프","컴프레서"],
    "견인전동기": ["트랙션모터","트랙션 모터","모터"],
    "출입문": ["승강문","도어"],
    "팬터그래프": ["집전장치","팬터"],
    "차축베어링": ["윤축베어링","베어링(차축)"],
    "절연저항": ["메거","절연 계측","절연 시험"],
    "그리스": ["윤활 그리스","윤활제"],
}

# ======================
# 0-1) TEXT NORMALIZER
# ======================
def _norm_text(s: str) -> str:
    if not isinstance(s, str): s = str(s or "")
    t = s.replace('\u00A0',' ').replace('\u200b',' ').replace('\u2060',' ')
    t = re.sub(r'[ \t]+', ' ', t)
    t = re.sub(r'\r\n?', '\n', t)
    # 단위 통일
    t = re.sub(r'N[\s\.\-·]*m', 'N·m', t, flags=re.IGNORECASE)      # N m / N.m / N·m
    t = t.replace('MΩ','MΩ').replace('MOhm','MΩ').replace('M OHM','MΩ').replace('㎜','mm')
    t = t.replace('㎏','kg').replace('℃','°C').replace('㎐','Hz')
    # 표 줄바꿈 보정 (숫자\n단위)
    t = re.sub(r'(\d)\s*\n\s*(g|mm|V|A|Hz|kHz|N·m|MΩ|bar|MPa|kPa|°C)\b', r'\1 \2', t, flags=re.IGNORECASE)
    return t.strip()

def _apply_synonyms(s: str) -> str:
    t = s
    for base, alts in SYNONYMS.items():
        for a in alts:
            t = re.sub(fr'\b{re.escape(a)}\b', base, t)
    return t

# ======================
# 1) PDF PARSING (+OCR 폴백)
# ======================
def _ocr_pages(pdf_path: Path, page_indices: List[int]) -> List[str]:
    try:
        from pdf2image import convert_from_path
        import pytesseract
        images = convert_from_path(str(pdf_path), dpi=300,
                                   first_page=min(page_indices)+1, last_page=max(page_indices)+1)
        out = []
        base = min(page_indices)
        for idx in page_indices:
            img = images[idx - base]
            txt = pytesseract.image_to_string(img, lang="kor+eng")
            out.append(_norm_text(txt))
        return out
    except Exception as e:
        print(f"[ocr] skipped {pdf_path.name}: {e}")
        return ["" for _ in page_indices]

def _maybe_ocr_fix(pdf_path: Path, pages: List[str], min_chars=200) -> List[str]:
    short_idxs = [i for i,t in enumerate(pages) if len((t or "").strip()) < min_chars]
    if not short_idxs: return pages
    ocr_txts = _ocr_pages(pdf_path, short_idxs)
    for i, txt in zip(short_idxs, ocr_txts):
        if txt and len(txt.strip()) > len(pages[i]):
            pages[i] = txt
    return pages

def parse_one_pdf(pdf_path: Path) -> Tuple[str, List[str], str]:
    err = ""
    pages: List[str] = []
    # 1) pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(str(pdf_path)) as pdf:
            for p in pdf.pages: pages.append(_norm_text(p.extract_text() or ""))
        if any(pages): return (pdf_path.name, _maybe_ocr_fix(pdf_path, pages), err)
    except Exception as e:
        err += f"pdfplumber:{e};"
    # 2) PyMuPDF
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        for i in range(doc.page_count):
            pages.append(_norm_text(doc.load_page(i).get_text() or ""))
        if any(pages): return (pdf_path.name, _maybe_ocr_fix(pdf_path, pages), err)
    except Exception as e:
        err += f"fitz:{e};"
    # 3) PyPDF2
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(str(pdf_path))
        for p in reader.pages: pages.append(_norm_text(p.extract_text() or ""))
        if any(pages): return (pdf_path.name, _maybe_ocr_fix(pdf_path, pages), err)
    except Exception as e:
        err += f"PyPDF2:{e};"
    # 4) 전면 OCR(최후)
    try:
        pages = _ocr_pages(pdf_path, list(range(0, 9999)))  # first/last는 내부에서 안전 처리
    except Exception as e:
        err += f"fullOCR:{e};"
    return (pdf_path.name, pages, err)

def step1_parse_pdf(save: bool=True) -> Dict[str, int]:
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in: {DATA_DIR}")
    results = []
    with mp.Pool(CPU_WORKERS) as pool:
        for r in pool.imap_unordered(parse_one_pdf, pdf_files):
            results.append(r)
    failed = []
    stats = {}
    for fname, pages, err in results:
        if not pages:
            failed.append((fname, err))
            continue
        stats[fname] = len(pages)
        if save:
            with open(PARSED_DIR / f"{fname}_pages.pkl", "wb") as f:
                pickle.dump(pages, f)
    if failed:
        (REPORTS_DIR / "step1_failed.json").write_text(json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[STEP1] failed: {len(failed)} → reports/step1_failed.json")
    if not stats: raise RuntimeError("No text extracted from any PDFs.")
    print(f"[STEP1] parsed: {len(stats)} files")
    return stats

# ======================
# 2) 장치 인식 + 섹션/표/안전 청크
# ======================
SECTION_RE = re.compile(r"(?m)^(?P<num>(?:\d+(?:[\.\-]\d+)*|\[\d+\]|\d+\)))[\s\-–—]*(?P<title>[^\n]{2,})")

def _device_from_filename(fname: str) -> Optional[str]:
    for k, v in DEVICE_MAP_SEEDS.items():
        if k in fname: return v
    if "모터" in fname: return "traction_motor"
    return None

@dataclass
class Chunk:
    chunk_id: int
    pdf_name: str
    section_id: str
    title: str
    page_start: int
    page_end: int
    type: str    # procedure|table|safety|misc
    text: str
    device_id: str

def _load_pages() -> Dict[str, List[str]]:
    pkl_files = list(PARSED_DIR.glob("*_pages.pkl"))
    if not pkl_files:
        step1_parse_pdf(save=True)
        pkl_files = list(PARSED_DIR.glob("*_pages.pkl"))
    return {p.name.replace("_pages.pkl",""): pickle.load(open(p,"rb")) for p in pkl_files}

def step2_build_chunks(save: bool=True) -> int:
    pdf_pages = _load_pages()
    chunks: List[Chunk] = []
    cid = 0
    for pdf_name, pages in pdf_pages.items():
        dev_hint = _device_from_filename(pdf_name) or "unknown"
        # 페이지 연결
        offsets, joined, total = [], [], 0
        for t in pages:
            t = _apply_synonyms(_norm_text(t))
            joined.append(t); offsets.append(total); total += len(t) + 1
        joined_s = "\n".join(joined)

        def pos_to_page(pos: int) -> int:
            lo, hi = 0, len(offsets)-1
            while lo < hi:
                mid = (lo+hi+1)//2
                if offsets[mid] <= pos: lo = mid
                else: hi = mid-1
            return lo

        # 섹션 추출
        matches = list(SECTION_RE.finditer(joined_s))
        for i, m in enumerate(matches):
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(joined_s)
            body = joined_s[start:end].strip()
            if len(body) < 50: continue
            p_start = pos_to_page(start) + 1
            p_end   = pos_to_page(end) + 1
            chunks.append(Chunk(cid, pdf_name, m.group("num"),
                                _norm_text(m.group("title").strip()),
                                p_start, p_end, "procedure", body, dev_hint))
            cid += 1

        # 표/수치/안전 힌트(페이지 단위)
        for i, page in enumerate(pages, start=1):
            text = _apply_synonyms(_norm_text(page or ""))
            # 표/수치: 단위/숫자 밀집 + 키워드
            if (len(re.findall(r'\b\d+(\.\d+)?\b', text)) >= 8) or any(k in text for k in
               ["사양","규격","정격","유  형 원  인 조  치","치수","토크","절연","그리스","유량","압력","간극","rpm","Hz","°C","MΩ","N·m","mm","g","V","A","bar","MPa","kPa"]):
                chunks.append(Chunk(cid, pdf_name, f"P{i}", "표/수치/규격(heuristic)", i, i, "table", text, dev_hint)); cid += 1
            # 안전
            if any(k in text for k in ["경고","주의","위험","금지","PPE","안전","660 kg","660kg","고전압","감전"]):
                chunks.append(Chunk(cid, pdf_name, f"P{i}", "안전문구(heuristic)", i, i, "safety", text, dev_hint)); cid += 1

    df = pd.DataFrame([asdict(c) for c in chunks])
    if save:
        df.to_csv(PARSED_DIR / "chunks.csv", index=False, encoding="utf-8-sig")
    print(f"[STEP2] chunks: {len(df)}")
    return len(df)

# ======================
# 3) 인덱스: TF-IDF + BM25 (+옵션 임베딩)
# ======================
def _load_chunks_df() -> pd.DataFrame:
    csv = PARSED_DIR / "chunks.csv"
    if not csv.exists():
        step2_build_chunks(save=True)
    return pd.read_csv(csv)

def _tokenize_for_bm25(text: str) -> List[str]:
    t = _apply_synonyms(_norm_text(text))
    return re.findall(r"[가-힣A-Za-z0-9]+", t)

def step3_build_index(save: bool=True, build_embedding: bool=False) -> str:
    from sklearn.feature_extraction.text import TfidfVectorizer
    df = _load_chunks_df()
    texts = (df["title"].fillna("") + "\n" + df["text"].fillna("")).tolist()

    # TF-IDF(한글 강력) — char_wb 2~5
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,5), min_df=1, max_df=0.95)
    X = vectorizer.fit_transform(texts)

    # BM25
    bm25_tokens = [_tokenize_for_bm25(t) for t in texts]
    bm25_obj = None
    try:
        from rank_bm25 import BM25Okapi
        bm25_obj = BM25Okapi(bm25_tokens)
    except Exception:
        pass

    # (옵션) 임베딩 인덱스
    emb_mat = None; emb_model_name = None
    if build_embedding:
        try:
            from sentence_transformers import SentenceTransformer
            emb_model_name = os.getenv("TM_EMB_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
            model = SentenceTransformer(emb_model_name)
            emb_mat = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
        except Exception as e:
            print(f"[INDEX] embedding skipped: {e}")

    if save:
        pickle.dump(df, open(INDEX_DIR / "meta.pkl", "wb"))
        pickle.dump(vectorizer, open(INDEX_DIR / "vectorizer.pkl","wb"))
        sparse.save_npz(INDEX_DIR / "matrix.npz", X)
        pickle.dump(bm25_tokens, open(INDEX_DIR / "bm25_tokens.pkl","wb"))
        if bm25_obj is not None:
            pickle.dump(bm25_obj, open(INDEX_DIR / "bm25.pkl","wb"))
        if emb_mat is not None:
            pickle.dump({"emb": emb_mat, "model": emb_model_name}, open(INDEX_DIR / "emb.pkl","wb"))
    print(f"[STEP3] index built (TF-IDF{' + BM25' if bm25_obj else ''}{' + EMB' if emb_mat is not None else ''})")
    return str(INDEX_DIR)

def _ensure_index():
    need = ["meta.pkl","vectorizer.pkl","matrix.npz","bm25_tokens.pkl"]
    if all((INDEX_DIR / f).exists() for f in need): return
    if not (PARSED_DIR / "chunks.csv").exists():
        if not list(PARSED_DIR.glob("*_pages.pkl")): step1_parse_pdf(save=True)
        step2_build_chunks(save=True)
    step3_build_index(save=True)

# ======================
# 4) 장치별 용어 사전(자동 확장) + 장치 감지
# ======================
def build_device_lexicons(top_n: int=80) -> Dict[str, List[str]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    df = _load_chunks_df()
    if "device_id" not in df.columns:
        df["device_id"] = "unknown"
    by_dev = df.groupby("device_id")["text"].apply(lambda s: "\n".join(s.fillna("")))
    all_txt = "\n".join(df["text"].fillna(""))

    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=3, max_df=0.9)
    V_all = vectorizer.fit_transform([_apply_synonyms(_norm_text(all_txt))])
    vocab = vectorizer.get_feature_names_out()

    lex = {}
    for dev, txt in by_dev.items():
        Vi = vectorizer.transform([_apply_synonyms(_norm_text(txt))])
        scores = (Vi - V_all).toarray().ravel()
        top_idx = scores.argsort()[::-1][:top_n]
        cand = [vocab[i] for i in top_idx]
        cand = [c for c in cand if not re.fullmatch(r"[0-9\.\-·]+", c)]
        # 단위/불용어 제외
        ban = {"mm","nm","g","mω","n·m","v","kv","khz","hz","°c","bar","mpa","kpa","rpm"}
        cand = [c for c in cand if c.lower() not in ban]
        seeds = [k for k,v in DEVICE_MAP_SEEDS.items() if v==dev]
        lex[dev] = sorted(set(seeds + cand))
    pickle.dump(lex, open(INDEX_DIR / "device_lexicons.pkl","wb"))
    print(f"[LEX] devices: {len(lex)} built")
    return lex

def _load_device_lexicons() -> Dict[str, List[str]]:
    p = INDEX_DIR / "device_lexicons.pkl"
    if p.exists(): return pickle.load(open(p,"rb"))
    return build_device_lexicons()

def infer_device_from_query(q: str, lexicons: Dict[str, List[str]]) -> List[Tuple[str, float]]:
    qn = _apply_synonyms(_norm_text(q)).lower()
    scores = []
    for dev, words in lexicons.items():
        sc = 0.0
        for w in words[:50]:
            if w and w.lower() in qn: sc += 1.0
        scores.append((dev, sc))
    scores.sort(key=lambda x: x[1], reverse=True)
    out = [(d,s) for d,s in scores[:2] if s>0]
    return out or [("unknown", 0.0)]

def expand_query_with_device(q: str, dev: str, lexicons: Dict[str, List[str]], max_terms=12) -> str:
    base = _apply_synonyms(_norm_text(q))
    extra = [w for w in lexicons.get(dev, [])[:max_terms] if len(w)>=2]
    return base if not extra else base + " " + " ".join(sorted(set(extra)))

# ======================
# 5) 검색(이중/삼중 앙상블 + 라우팅)
# ======================
def _retrieve_core(queries: List[str], top_k: int=8, tfidf_w=TFIDF_WEIGHT, bm25_w=BM25_WEIGHT, emb_w=EMB_WEIGHT) -> pd.DataFrame:
    from sklearn.metrics.pairwise import linear_kernel
    _ensure_index()
    df_meta: pd.DataFrame = pickle.load(open(INDEX_DIR / "meta.pkl","rb"))
    vectorizer = pickle.load(open(INDEX_DIR / "vectorizer.pkl","rb"))
    X = sparse.load_npz(INDEX_DIR / "matrix.npz")
    bm25_tokens = pickle.load(open(INDEX_DIR / "bm25_tokens.pkl","rb"))
    try:
        bm25_obj = pickle.load(open(INDEX_DIR / "bm25.pkl","rb"))
    except Exception:
        bm25_obj = None

    # TF-IDF
    qv = vectorizer.transform(queries)
    sims = linear_kernel(qv, X).toarray()  # (Q,N)

    # BM25
    if bm25_obj is not None:
        bm25_scores = []
        for q in queries:
            toks = _tokenize_for_bm25(q)
            bm25_scores.append(bm25_obj.get_scores(toks))
        bm25_scores = pd.DataFrame(bm25_scores)  # (Q,N)
        # 정규화
        bm25_scores = (bm25_scores - bm25_scores.min(axis=1).values.reshape(-1,1)) / \
                      (bm25_scores.max(axis=1).values.reshape(-1,1) - bm25_scores.min(axis=1).values.reshape(-1,1) + 1e-9)
        tfidf_scores = pd.DataFrame(sims)
        tfidf_scores = (tfidf_scores - tfidf_scores.min(axis=1).values.reshape(-1,1)) / \
                       (tfidf_scores.max(axis=1).values.reshape(-1,1) - tfidf_scores.min(axis=1).values.reshape(-1,1) + 1e-9)
        combo = tfidf_w*tfidf_scores + bm25_w*bm25_scores
        final = combo.values
    else:
        final = sims

    # (옵션) 임베딩 인덱스 앙상블
    try:
        emb_pack = pickle.load(open(INDEX_DIR / "emb.pkl","rb"))
        emb_mat = emb_pack["emb"]     # (N,D)
        emb_model_name = emb_pack["model"]
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer(emb_model_name)
        q_emb = model.encode(queries, convert_to_numpy=True, normalize_embeddings=True)  # (Q,D)
        emb_sim = (q_emb @ emb_mat.T)  # cosine (Q,N)
        emb_sim = (emb_sim - emb_sim.min(axis=1, keepdims=True)) / (emb_sim.max(axis=1, keepdims=True) - emb_sim.min(axis=1, keepdims=True) + 1e-9)
        final = (1-emb_w)*final + emb_w*emb_sim
    except Exception:
        pass

    # 여러 확장 쿼리 합산(맥스)
    agg = final.max(axis=0)
    idx = agg.argsort()[::-1][:top_k]
    out = df_meta.iloc[idx].copy()
    out["score"] = agg[idx]
    out["preview"] = out["text"].str.replace("\n", " ").str.slice(0, 200) + "..."
    return out[["score","pdf_name","section_id","title","type","page_start","page_end","device_id","text","preview"]]

def step4_retrieve(query: str, top_k: int=8, device_hint: Optional[str]=None) -> pd.DataFrame:
    lex = _load_device_lexicons()
    dev_cands = [device_hint] if device_hint else [d for d,_ in infer_device_from_query(query, lex)]
    dev_cands = [d for d in dev_cands if d] or ["unknown"]
    # 장치별 확장 쿼리
    queries = [ _apply_synonyms(_norm_text(query)) ] + [ expand_query_with_device(query, d, lex) for d in dev_cands ]
    out = _retrieve_core(queries, top_k=top_k)
    # 장치 가중치
    if dev_cands and "device_id" in out.columns:
        out.loc[out["device_id"].isin(dev_cands), "score"] *= 1.1
        out = out.sort_values("score", ascending=False).head(top_k)
    # (옵션) 크로스엔코더 재랭크
    try:
        from sentence_transformers import CrossEncoder
        ce_model = os.getenv("TM_CE_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        ce = CrossEncoder(ce_model)
        pairs = [(query, t) for t in out["text"].tolist()]
        ce_scores = ce.predict(pairs, show_progress_bar=False)
        out.loc[:,"score"] = 0.7*out["score"].values + 0.3*(ce_scores - ce_scores.min())/(ce_scores.max()-ce_scores.min()+1e-9)
        out = out.sort_values("score", ascending=False).head(top_k)
    except Exception:
        pass
    return out.drop(columns=["text"])

# ======================
# 6) 수치/단위 추출(문맥 검증 포함)
# ======================
NUM_PATS = {
    "voltage_V":       re.compile(r"\b(\d{2,4})(?:\s*(?:kV|KV))?\s*V\b", re.IGNORECASE),
    "current_A":       re.compile(r"\b(\d{1,3}(?:\.\d+)?)\s*A\b", re.IGNORECASE),
    "resistance_MOhm": re.compile(r"\b(\d{1,4}(?:\.\d+)?)\s*MΩ\b", re.IGNORECASE),
    "resistance_kOhm": re.compile(r"\b(\d{1,4}(?:\.\d+)?)\s*kΩ\b", re.IGNORECASE),
    "resistance_Ohm":  re.compile(r"\b(\d{1,5}(?:\.\d+)?)\s*Ω\b"),
    "cap_uF":          re.compile(r"\b(\d{1,4}(?:\.\d+)?)\s*(?:µF|uF)\b"),
    "grease_g":        re.compile(r"\b(\d{1,3})\s*g\b"),
    "torque_Nm":       re.compile(r"\b(\d{1,3}(?:\.\d+)?)\s*N·m\b", re.IGNORECASE),
    "torque_range":    re.compile(r"\b(\d{1,3})\s*[~\-–]\s*(\d{1,3})\s*N·m\b", re.IGNORECASE),
    "gap_mm":          re.compile(r"\b([0-9]+(?:\.[0-9]+)?)\s*mm\b", re.IGNORECASE),
    "press_bar":       re.compile(r"\b(\d{1,3}(?:\.\d+)?)\s*bar\b", re.IGNORECASE),
    "press_MPa":       re.compile(r"\b(\d(?:\.\d+)?)\s*MPa\b", re.IGNORECASE),
    "temp_C":          re.compile(r"\b(-?\d{1,3}(?:\.\d+)?)\s*°C\b"),
    "freq_Hz":         re.compile(r"\b(\d{1,5})\s*Hz\b", re.IGNORECASE),
    "freq_kHz":        re.compile(r"\b(\d{1,4})\s*kHz\b", re.IGNORECASE),
    "rpm":             re.compile(r"\b(\d{2,6})\s*rpm\b", re.IGNORECASE),
}

CONTEXT_HINTS = {
    "torque_Nm": ["규정 토크","체결 토크","조임 토크","체결부","볼트","커플링","플랜지"],
    "resistance_MOhm": ["절연","합격 기준","절연저항","절연 측정","메거","메가옴"],
    "voltage_V": ["절연","시험 전압","메거","측정 전압","테스트"],
    "grease_g": ["그리스","윤활","주입","주입량","도포"],
    "gap_mm": ["간극","갭","틈","클리어런스"],
    "press_bar": ["압력","공압","라인 압력","설정 압력","토출 압력"],
    "temp_C": ["온도","과열","열화","냉각","주위온도","권선","베어링"],
    "freq_Hz": ["주파수","센서","진동","측정","분석","필터"],
}

def _number_norm(key: str, val: str) -> Tuple[str, str]:
    """정규화: kV→V, kΩ→Ω 등(표시/내부)"""
    if key=="voltage_V":
        # 정규식에서 kV는 V 뒤에 붙이므로 별도 변환은 생략(원문 그대로 표기)
        return val, val
    if key=="resistance_kOhm":
        try: ohm = float(val)*1e3; return f"{val} kΩ", f"{ohm:.0f} Ω"
        except: return f"{val} kΩ", val
    return val, val

def _extract_numbers_with_context(text: str) -> Dict[str, List[str]]:
    T = _apply_synonyms(_norm_text(text))
    out: Dict[str, List[str]] = {}
    for k, pat in NUM_PATS.items():
        vals = []
        for m in pat.finditer(T):
            val = "–".join(m.groups()) if k=="torque_range" else m.group(1)
            s, e = m.span()
            ctx = T[max(0,s-80):min(len(T),e+80)]
            base_k = "torque_Nm" if k=="torque_range" else k
            hints = CONTEXT_HINTS.get(base_k, [])
            if hints and not any(h in ctx for h in hints): 
                continue
            disp, norm = _number_norm(k, val)
            vals.append(disp)
        if vals:
            out[k] = sorted(set(vals), key=lambda x: (len(x), x))
    return out

# ======================
# 7) 작업카드 생성
# ======================
def step5_generate_workcard(query: str, top_k: int=8, device_hint: Optional[str]=None) -> Dict[str, Any]:
    _ensure_index()
    hits = step4_retrieve(query, top_k=top_k, device_hint=device_hint)
    df = pickle.load(open(INDEX_DIR / "meta.pkl","rb"))

    merged, refs = "", []
    for _, row in hits.iterrows():
        mask = (
            (df["pdf_name"] == row["pdf_name"]) &
            (df["section_id"] == row["section_id"]) &
            (df["page_start"] == row["page_start"]) &
            (df["page_end"] == row["page_end"])
        )
        txts = df.loc[mask, "text"]
        if len(txts): merged += "\n" + str(txts.values[0])
        refs.append({
            "pdf_name": row["pdf_name"],
            "section_id": row["section_id"],
            "pages": [int(row["page_start"]), int(row["page_end"])],
            "type": row["type"],
            "device_id": row.get("device_id","unknown")
        })

    nums = _extract_numbers_with_context(merged)

    # 안전문구 수집
    safety = ["고전압 작업 전 차단·방전"]
    try:
        for _, r in df[df["type"]=="safety"].head(200).iterrows():
            if any(k in str(r["text"]) for k in ["660 kg","660kg","고전압","감전"]):
                safety.append(f"안전 주의: {r['pdf_name']} P{r['page_start']}")
                break
    except Exception:
        pass

    actions = []
    dev_cands = infer_device_from_query(query, _load_device_lexicons())
    asset_dev = (device_hint or (dev_cands[0][0] if dev_cands else "unknown"))
    qn = _apply_synonyms(_norm_text(query))

    # 의도별 액션 템플릿
    if any(k in qn for k in ["베어링","그리스","윤활"]):
        g = (nums.get("grease_g") or ["5"])[0]
        actions.append({"step": len(actions)+1,
                        "text": f"그리스 주입구 청결 확인 후 권장 그리스를 각 베어링에 {g} g 주입(정지 상태, 과주입 금지).",
                        "refs": [f"{r['pdf_name']}:{r['section_id']}" for r in refs]})
    if any(k in qn for k in ["절연","절연저항","과열","메거"]):
        v = (nums.get("voltage_V") or ["1000"])[0]
        rmin = (nums.get("resistance_MOhm") or ["50"])[0]
        actions.append({"step": len(actions)+1,
                        "text": f"절연저항계 {v} V로 측정하여 합격 기준 {rmin} MΩ 이상 확인.",
                        "refs": [f"{r['pdf_name']}:{r['section_id']}" for r in refs]})
    if any(k in qn for k in ["토크","체결","조임"]):
        ts = sorted(set(nums.get("torque_Nm") or []))
        if "torque_range" in nums:
            ts += [f"{rng} (범위)" for rng in nums["torque_range"]]
        if ts:
            actions.append({"step": len(actions)+1,
                            "text": "체결부 규정 토크로 조임: " + ", ".join(ts) + " N·m",
                            "refs": [f"{r['pdf_name']}:{r['section_id']}" for r in refs]})
    if any(k in qn for k in ["간극","갭","클리어런스"]):
        gs = sorted(set(nums.get("gap_mm") or []))
        if gs:
            actions.append({"step": len(actions)+1,
                            "text": "규정 간극 확인/조정: " + ", ".join(gs) + " mm",
                            "refs": [f"{r['pdf_name']}:{r['section_id']}" for r in refs]})

    card = {
        "asset": {"device": asset_dev},
        "diagnosis": {"from_query": query},
        "actions": actions,
        "tests": [],
        "safety": list(dict.fromkeys(safety)),
        "sources": refs
    }
    if (any(k in qn for k in ["절연","절연저항","과열"])) and (("voltage_V" not in nums) or ("resistance_MOhm" not in nums)):
        card["warning"] = "필수 수치(시험 전압/합격저항)를 명확히 찾지 못했습니다. 결과 근거 페이지를 확인하세요."
    return card

# ======================
# 8) 평가/로그
# ======================
def evaluate(golden_path: Path, top_k=8) -> Dict[str, Any]:
    """
    golden.json 형식 예:
    [
      {"query":"절연저항 기준", "must_include":["MΩ","절연"], "device_hint":"traction_motor"},
      {"query":"차축베어링 그리스 주입", "must_numbers":["grease_g"], "device_hint":"axle_bearing"}
    ]
    """
    gold = json.loads(Path(golden_path).read_text(encoding="utf-8"))
    _ensure_index()
    ok_hit, total = 0, 0
    num_ok, num_total = 0, 0
    logs = []
    for g in gold:
        query = g["query"]; hint = g.get("device_hint")
        res = step4_retrieve(query, top_k=top_k, device_hint=hint)
        text_join = " ".join(res["preview"].tolist())
        must = g.get("must_include", [])
        total += 1
        hit = all(m in text_join for m in must) if must else (len(res)>0)
        ok_hit += int(hit)
        # 수치
        wc = step5_generate_workcard(query, top_k=top_k, device_hint=hint)
        nums = set(re.findall(r"\b(N·m|MΩ|mm|g|V|A|bar|MPa|kPa|°C|Hz|rpm)\b", json.dumps(wc, ensure_ascii=False)))
        for n in g.get("must_numbers", []):
            num_total += 1
            num_ok += int(any(k.startswith(n) for k in wc.get("actions", []) or []) or (n in json.dumps(wc, ensure_ascii=False)))
        logs.append({"query":query,"hit":hit,"res_top1":res.head(1).to_dict(orient="records") if len(res) else []})
    report = {
        "topk_hit_rate": round(ok_hit/max(total,1), 3),
        "number_ok_rate": round(num_ok/max(num_total,1), 3),
        "cases": total
    }
    (REPORTS_DIR / "eval_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORTS_DIR / "eval_logs.json").write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[EVAL]", report)
    return report

# ======================
# 9) 빌드/CLI
# ======================
def rebuild_all(build_embedding: bool=False):
    t0 = time()
    step1_parse_pdf(save=True)
    step2_build_chunks(save=True)
    step3_build_index(save=True, build_embedding=build_embedding)
    build_device_lexicons(top_n=100)
    print(f"[DONE] rebuild_all in {time()-t0:.1f}s")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="모든 인덱스 재생성")
    ap.add_argument("--embed", action="store_true", help="임베딩 인덱스까지 생성(선택)")
    ap.add_argument("--query", type=str, help="검색 질의")
    ap.add_argument("--workcard", type=str, help="작업카드 생성 질의")
    ap.add_argument("--device", type=str, help="장치 힌트(device_id)")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--evaluate", type=str, help="골든셋 JSON 경로")
    args = ap.parse_args()

    if args.rebuild:
        rebuild_all(build_embedding=args.embed)
    elif args.evaluate:
        _ensure_index(); evaluate(Path(args.evaluate), top_k=args.topk)
    elif args.query:
        _ensure_index()
        df = step4_retrieve(args.query, top_k=args.topk, device_hint=args.device)
        print(df.to_string(index=False))
    elif args.workcard:
        _ensure_index()
        card = step5_generate_workcard(args.workcard, top_k=args.topk, device_hint=args.device)
        print(json.dumps(card, ensure_ascii=False, indent=2))
    else:
        ap.print_help()
