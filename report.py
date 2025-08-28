# src/pretty.py
from typing import Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule
from rich.markdown import Markdown
from datetime import datetime
import html
from jinja2 import Template
import os

console = Console(soft_wrap=True)

# ========== 콘솔 출력 ==========
def show_retrieval(df):
    """step4 결과 DataFrame을 예쁜 테이블로 출력"""
    t = Table(title="🔎 RAG Retrieval Top-k", show_lines=False, header_style="bold cyan")
    for col in ["score","section_id","title","type","page_start","page_end","preview"]:
        t.add_column(col)
    for _, r in df.iterrows():
        t.add_row(
            f"{r['score']:.4f}",
            str(r["section_id"]),
            str(r["title"]),
            str(r["type"]),
            str(int(r["page_start"])),
            str(int(r["page_end"])),
            str(r["preview"])[:180] + ("…" if len(str(r["preview"]))>180 else "")
        )
    console.print(t)

def show_workcard(card: Dict[str, Any]):
    """step5 작업카드 JSON을 예쁜 카드 형태로 출력"""
    a = card.get("asset", {})
    d = card.get("diagnosis", {})
    actions = card.get("actions", [])
    tests = card.get("tests", [])
    safety = card.get("safety", [])
    sources = card.get("sources", [])
    warn = card.get("warning")

    header = f"[bold cyan]🧰 작업 카드[/]  |  [bold]Device:[/] {a.get('device','-')}  [bold]Component:[/] {a.get('component','-')}"
    console.print(Rule(header))
    console.print(f"🎯 [bold]Diagnosis[/]: {d.get('mode','-')}  "
                  f"(sev={d.get('severity','-')}, conf={d.get('confidence','-')})")
    if d.get("evidence"):
        console.print("🧩 [bold]Evidence[/]: " + ", ".join(map(str, d["evidence"])))

    # Actions
    if actions:
        t = Table(title="🛠️ Actions", header_style="bold green")
        t.add_column("Step", justify="right", style="bold")
        t.add_column("Instruction")
        t.add_column("Refs")
        for s in actions:
            refs = ", ".join(s.get("refs", [])[:3])
            t.add_row(str(s.get("step","")), s.get("text",""), refs)
        console.print(t)

    # Tests
    if tests:
        t = Table(title="🧪 Tests", header_style="bold green")
        t.add_column("Name")
        t.add_column("Method")
        t.add_column("Pass")
        t.add_column("Refs")
        for ts in tests:
            t.add_row(ts.get("name",""), ts.get("method",""), ts.get("pass",""), ", ".join(ts.get("refs",[])[:3]))
        console.print(t)

    # Safety
    if safety:
        s = "\n".join([f"• {x}" for x in safety])
        console.print(Panel(s, title="⚠️ Safety", border_style="red"))

    # Sources
    if sources:
        t = Table(title="📄 Sources", header_style="bold green")
        t.add_column("Section")
        t.add_column("Pages")
        t.add_column("Type")
        for src in sources[:6]:
            t.add_row(str(src.get("section_id","")), str(src.get("pages","")), str(src.get("type","")))
        console.print(t)

    if warn:
        console.print(Panel(f"⚠️ {warn}", border_style="yellow"))


# ========== 파일 저장(Markdown/HTML) ==========
def card_to_markdown(card: Dict[str, Any]) -> str:
    a = card.get("asset", {})
    d = card.get("diagnosis", {})
    actions = card.get("actions", [])
    tests = card.get("tests", [])
    safety = card.get("safety", [])
    sources = card.get("sources", [])

    lines = []
    lines.append(f"# 🧰 작업 카드 ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
    lines.append(f"- **Device**: {a.get('device','-')}  |  **Component**: {a.get('component','-')}")
    lines.append(f"- **Diagnosis**: {d.get('mode','-')} (sev={d.get('severity','-')}, conf={d.get('confidence','-')})")
    if d.get("evidence"):
        lines.append(f"- **Evidence**: {', '.join(map(str, d['evidence']))}")
    lines.append("\n## 🛠️ Actions")
    if actions:
        for s in actions:
            refs = ", ".join(s.get("refs", []))
            lines.append(f"- **Step {s.get('step','')}**: {s.get('text','')}  _(refs: {refs})_")
    else:
        lines.append("- (none)")
    lines.append("\n## 🧪 Tests")
    if tests:
        for ts in tests:
            refs = ", ".join(ts.get("refs", []))
            lines.append(f"- **{ts.get('name','')}** — **Method**: {ts.get('method','')} / **Pass**: {ts.get('pass','')}  _(refs: {refs})_")
    else:
        lines.append("- (none)")
    lines.append("\n## ⚠️ Safety")
    if safety:
        for s in safety:
            lines.append(f"- {s}")
    else:
        lines.append("- (none)")
    lines.append("\n## 📄 Sources")
    if sources:
        for s in sources:
            lines.append(f"- Sec. {s.get('section_id','')}  p{s.get('pages','')}  ({s.get('type','')})")
    else:
        lines.append("- (none)")
    if card.get("warning"):
        lines.append(f"\n> ⚠️ **Warning**: {card['warning']}")
    return "\n".join(lines)

def save_card_md(card: Dict[str, Any], path: str):
    md = card_to_markdown(card)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)

HTML_TMPL = Template("""
<!doctype html><html lang="ko"><meta charset="utf-8">
<title>작업 카드</title>
<style>
body{font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Pretendard,Apple SD Gothic Neo,Malgun Gothic,sans-serif;
     margin:24px; color:#111; line-height:1.45}
h1{font-size:24px} h2{font-size:18px; margin-top:20px}
.card{border:1px solid #e5e7eb; border-radius:12px; padding:16px; background:#fff; box-shadow:0 2px 8px rgba(0,0,0,.04)}
.badge{display:inline-block; padding:2px 8px; border-radius:12px; background:#eef2ff; color:#3730a3; font-size:12px}
.section{margin-top:16px}
ul{margin:8px 0 0 20px}
.code{font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace}
small{color:#6b7280}
</style>
<div class="card">
<h1>🧰 작업 카드 <small>{{ now }}</small></h1>
<p><span class="badge">Device</span> {{ device }} &nbsp; <span class="badge">Component</span> {{ component }}</p>
<p><b>Diagnosis</b>: {{ diag.mode }} (sev={{ diag.sev }}, conf={{ diag.conf }})</p>
{% if diag.evidence %}<p><b>Evidence</b>: {{ diag.evidence }}</p>{% endif %}

<div class="section">
<h2>🛠️ Actions</h2>
<ul>
{% for s in actions %}
<li><b>Step {{ s.step }}</b>: {{ s.text }} <small>(refs: {{ ", ".join(s.refs) }})</small></li>
{% endfor %}
{% if actions|length == 0 %}<li>(none)</li>{% endif %}
</ul>
</div>

<div class="section">
<h2>🧪 Tests</h2>
<ul>
{% for t in tests %}
<li><b>{{ t.name }}</b> — Method: {{ t.method }} / Pass: {{ t["pass"] }} <small>(refs: {{ ", ".join(t.refs) }})</small></li>
{% endfor %}
{% if tests|length == 0 %}<li>(none)</li>{% endif %}
</ul>
</div>

<div class="section">
<h2>⚠️ Safety</h2>
<ul>
{% for s in safety %}<li>{{ s }}</li>{% endfor %}
{% if safety|length == 0 %}<li>(none)</li>{% endif %}
</ul>
</div>

<div class="section">
<h2>📄 Sources</h2>
<ul>
{% for s in sources %}
<li>Sec. {{ s.section_id }} p{{ s.pages }} ({{ s.type }})</li>
{% endfor %}
{% if sources|length == 0 %}<li>(none)</li>{% endif %}
</ul>
</div>

{% if warning %}
<div class="section"><h2>⚠️ Warning</h2><p>{{ warning }}</p></div>
{% endif %}
</div>
""".strip())

def save_card_html(card: Dict[str, Any], path: str):
    from datetime import datetime
    a = card.get("asset", {})
    d = card.get("diagnosis", {})
    ctx = {
        "now": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "device": a.get("device","-"),
        "component": a.get("component","-"),
        "diag": {"mode": d.get("mode","-"), "sev": d.get("severity","-"), "conf": d.get("confidence","-"),
                 "evidence": ", ".join(d.get("evidence",[])) if d.get("evidence") else ""},
        "actions": card.get("actions", []),
        "tests": card.get("tests", []),
        "safety": card.get("safety", []),
        "sources": card.get("sources", []),
        "warning": card.get("warning", "")
    }
    html_out = HTML_TMPL.render(**ctx)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_out)
