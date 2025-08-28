# 01_build_dataset.py (상단 경로/수집 부분만 교체)
from pathlib import Path
import re, os, json, csv

# 1) PDF 폴더 (윈도우에서는 raw string 추천)
DATA_DIR = Path(r"C:\Dev\KorailPJT\data")

# 2) 파일명: 1.pdf ~ 4.pdf 고정 수집
PDFS = [DATA_DIR / f"{i}.pdf" for i in range(1, 5)]

# 존재 확인
missing = [str(p) for p in PDFS if not p.exists()]
if missing:
    print("[ERROR] 아래 파일이 존재하지 않습니다:")
    for m in missing: print("  -", m)
    # 필요하면 여기서 exit()로 중단해도 됨

# ========= 텍스트 추출기: pdfplumber → PyPDF2 순차 시도 =========
def pdf_to_text(path):
    # 1) pdfplumber (권장: 표/한글 비교적 잘 나옴)
    try:
        import pdfplumber
        txt_parts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                txt_parts.append(t)
        if any(txt_parts):
            return "\n".join(txt_parts)
    except Exception:
        pass
    # 2) PyPDF2 (대체)
    try:
        import PyPDF2
        text = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for p in reader.pages:
                try:
                    text.append(p.extract_text() or "")
                except Exception:
                    text.append("")
        return "\n".join(text)
    except Exception:
        return ""

# ========= 코드/제목 라인 잡는 정규식 보강 =========
# 예시 패턴들: "D1-17-03: 활주방지 고장", "91-23-08 : 주변환장치(CI) 견인력제한", "CODE-XXX – 제목"
PATTERNS = [
    re.compile(r"(^|\n)\s*([A-Z]{1,3}\d?[-–]\d{1,3}[-–]\d{2,})\s*[:\-–]\s*([^\n]+)"),
    re.compile(r"(^|\n)\s*(\d{2}[-–]\d{2}[-–]\d{2})\s*[:\-–]\s*([^\n]+)"),
    re.compile(r"(^|\n)\s*([A-Z0-9]{2,}[-–][A-Z0-9\-]{2,})\s*[:\-–]\s*([^\n]+)"),
]

def find_code_lines(tx):
    hits = []
    for pat in PATTERNS:
        for m in pat.finditer(tx):
            code_id = m.group(2).strip()
            title   = m.group(3).strip()
            hits.append((m.start(), code_id, title))
    # 중복/겹침 제거: code_id 기준으로 앞선 것 우선
    seen, uniq = set(), []
    for pos, cid, ttl in sorted(hits, key=lambda x:x[0]):
        if cid in seen: 
            continue
        seen.add(cid)
        uniq.append((pos, cid, ttl))
    return uniq

# ========= 약라벨 규칙(기존 그대로 사용해도 됨, 필요 시 단어 추가) =========
DEVICE_RULES = {
    r"(주변환장치|CI|인버터|컨버터|SIV|CON|보조전원)": "AUX_POWER",
    r"(견인|구동|VCU|Traction|Drive)": "TRACTION",
    r"(제동|WSP|ECU|활주방지)": "BRAKE",
    r"(승강문|도어)": "DOOR",
    r"(팬터그래프|팬터|가선|MCB|지붕라인|25kV)": "PANTO",
    r"(주공기|압축기|COMP|AIR COMP)": "AIR_COMP",
    r"(현수|서스펜션|SUSP)": "SUSPENSION",
    r"(신호장치|ATP|BTM|EVC|PSTK)": "SIGNAL",
    r"(VCU|CCU)": "VCU/CCU",
    r"(축전지|배터리|BMS|충전기)": "BATTERY",
    r"(차축|휠|비회전|발열|대차 불안정)": "WHEELSET",
}
FAIL_RULES = {
    r"(과열|열검지|Overheat|Temp)": "OVERHEAT",
    r"(과전류|Overcurrent)": "OVERCURRENT",
    r"(접지.?고장|접지.?단락|Ground|접지)": "GROUND_FAULT",
    r"(통신고장|Communication|Comm)": "COMM_FAULT",
    r"(센서|속도센서|엔코더|검지.*고장|Sensor)": "SENSOR_FAULT",
    r"(CB ?트립|차단기|브레이커)": "CB_TRIP",
    r"(누설|상실|Loss|Leak)": "LEAK/LOSS",
    r"(활주방지|WSP|슬립|Slip)": "SLIP/WSP",
    r"(불안정|Unstable|대차 불안정)": "UNBALANCE/UNSTABLE",
    r"(충전기.*고장|충전율 부족|저전압)": "CHARGER_FAULT",
    r"(절연|메가|Insulation)": "INSULATION",
    r"(전원 차단|단전|차단)": "POWER_LOSS",
}

def make_labels(text):
    import re
    device, failure = set(), set()
    for pat, lab in DEVICE_RULES.items():
        if re.search(pat, text, re.IGNORECASE): device.add(lab)
    for pat, lab in FAIL_RULES.items():
        if re.search(pat, text, re.IGNORECASE): failure.add(lab)
    return sorted(device), sorted(failure)

# ========= 메인 빌드 =========
OUT_DIR = Path("./out"); OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_CSV = OUT_DIR / "faultcodes_dataset.csv"

def build():
    rows = []
    for pdf in PDFS:
        if not pdf.exists():
            print("[WARN] Not found:", pdf)
            continue
        tx = pdf_to_text(pdf)
        if not tx.strip():
            print("[WARN] Empty text (OCR 필요할 수 있음):", pdf)
            continue
        hits = find_code_lines(tx)
        if not hits:
            print("[WARN] 코드 패턴 미검출:", pdf)
        for pos, code_id, title in hits:
            ctx = tx[pos:pos+1200]  # 코드 주변 문맥
            y_dev, y_fail = make_labels(title + " " + ctx)
            rows.append({
                "code_id": code_id,
                "title": title,
                "context": ctx[:1000],
                "y_device": "|".join(y_dev),
                "y_failure": "|".join(y_fail),
                "source_pdf": pdf.name
            })
    # 저장
    with open(DATA_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["code_id","title","context","y_device","y_failure","source_pdf"])
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"saved: {DATA_CSV} ({len(rows)} rows)")

if __name__ == "__main__":
    build()
