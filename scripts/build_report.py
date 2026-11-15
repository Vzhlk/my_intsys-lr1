"""Сборка отчёта ЛР1: report/REPORT.md и docx с титульным листом ТПУ (5 страниц с титулом).

    python scripts/build_report.py [межстрочный_интервал]

Титульный лист и стили берутся из отчёта ЛР0 (../Лаба 0/Лаба_0_отчет.docx), содержимое — из scripts/report_content.py.
"""
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import docx  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: E402
from docx.oxml import OxmlElement  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.shared import Cm, Pt  # noqa: E402

from report_content import blocks, to_markdown  # noqa: E402

TEMPLATE = ROOT.parent.parent / "Лаба 0" / "Лаба_0_отчет.docx"
DST = ROOT.parent / "Лаба_1_отчет.docx"
LS = float(sys.argv[1]) if len(sys.argv) > 1 else 1.1
THEME = "Инференс и развёртывание языковой модели: от одного запроса к серверу под нагрузкой"
DATE = "15.11.2026"


def retitle(body_elements):
    """Титульный лист ЛР0 -> ЛР1: номер работы, тема, дата (правка текста в w:t)."""
    for el in body_elements:
        for p in el.iter(qn("w:p")):
            ts = list(p.iter(qn("w:t")))
            joined = "".join(t.text or "" for t in ts).strip()
            new = THEME if joined == "Воспроизводимый эксперимент: подбор гиперпараметров нейросети" else (DATE if joined == "04.11.2026" else None)
            if new and ts:
                ts[0].text = new
                for t in ts[1:]:
                    t.text = ""
        for t in el.iter(qn("w:t")):
            if t.text is None:
                continue
            if "РАБОТЕ 0" in t.text:
                t.text = t.text.replace("РАБОТЕ 0", "РАБОТЕ 1")
            elif t.text.strip() == "0" and t.getparent().getprevious() is None:
                pass
            if "Воспроизводимый эксперимент" in t.text or "подбор гиперпараметров нейросети" in t.text:
                t.text = t.text.replace("Воспроизводимый эксперимент: подбор гиперпараметров нейросети", THEME)
            if t.text.strip() == "04.11.2026":
                t.text = DATE


def main():
    bl = blocks()
    (ROOT / "report" / "REPORT.md").write_text(to_markdown(bl), encoding="utf-8")
    d = docx.Document(TEMPLATE)
    body = d.element.body
    els = list(body)
    sect = els[-1]
    title = els[:39]
    retitle(title)
    # тема в две строки: убираем два пустых абзаца перед «Томск – 2026 г.», чтобы титул остался на одной странице
    empties = [el for el in els[27:38] if el.tag == qn("w:p") and not "".join(el.itertext()).strip()]
    for el in empties[:2]:
        body.remove(el)
    for el in els[39:-1]:
        body.remove(el)
    st = d.styles["ТекстОснова"]
    st.paragraph_format.line_spacing = LS
    st.paragraph_format.space_after = Pt(0)
    st.paragraph_format.space_before = Pt(0)

    def add_runs(p, text, size=None, bold_all=False):
        """Текст с `кодом` (моноширинный) и **жирным** внутри абзаца."""
        for part in re.split(r"(`[^`]+`)", text):
            if not part:
                continue
            if part.startswith("`"):
                r = p.add_run(part.strip("`")); r.font.name = "Consolas"
                r._r.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Consolas")
                r.font.size = Pt((size or 12) - 1)
            else:
                for sub in re.split(r"(\*\*[^*]+\*\*)", part):
                    if not sub:
                        continue
                    r = p.add_run(sub.strip("*") if sub.startswith("**") else sub)
                    r.bold = bold_all or sub.startswith("**")
                    if size:
                        r.font.size = Pt(size)

    def add_p(text="", lead=None, before=0, after=0, keep=False, indent=True, align=None, size=None, page_break=False, bold=False):
        p = d.add_paragraph(style="ТекстОснова")
        pf = p.paragraph_format
        pf.space_before = Pt(before); pf.space_after = Pt(after)
        if keep:
            pf.keep_with_next = True
        if page_break:
            pf.page_break_before = True
        if not indent:
            pf.first_line_indent = Cm(0)
        if align is not None:
            p.alignment = align
        if lead:
            r = p.add_run(lead); r.bold = True
            if size:
                r.font.size = Pt(size)
        if text:
            add_runs(p, text, size, bold_all=bold)
        return p

    def table(rows, widths, size=9):
        t = d.add_table(rows=len(rows), cols=len(rows[0]))
        t.style = d.styles["Table Grid"]
        t.alignment = 1
        t.autofit = False
        for i, row in enumerate(rows):
            for j, v in enumerate(row):
                cell = t.cell(i, j)
                cell.width = Cm(widths[j])
                p = cell.paragraphs[0]
                p.paragraph_format.space_before = Pt(0); p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1.0
                p.paragraph_format.first_line_indent = Cm(0)
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if (j == 0 and widths[0] > 5) else WD_ALIGN_PARAGRAPH.CENTER
                add_runs(p, str(v), size, bold_all=(i == 0))
                for r in p.runs:
                    r.font.name = r.font.name or "Times New Roman"
                if i == 0:
                    tcPr = cell._tc.get_or_add_tcPr()
                    sh = OxmlElement("w:shd"); sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), "D9D9D9")
                    tcPr.append(sh)
            t.rows[i]._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        return t

    for blk in bl:
        k = blk[0]
        if k == "h1":
            add_p(lead=blk[1], before=12 if blk[2] else 5, keep=True, page_break=bool(blk[2]) if len(blk) > 2 else False)
        elif k == "h2":
            add_p(lead=blk[1], before=3, keep=True)
        elif k == "p":
            add_p(blk[1], lead=blk[2], before=0)
        elif k == "table":
            add_p(blk[1], keep=True, before=3, after=1, indent=False, align=WD_ALIGN_PARAGRAPH.LEFT, size=11)
            table(blk[2], blk[3])
            add_p(indent=False).paragraph_format.line_spacing = Pt(3)
        elif k == "fig":
            p = add_p(indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, keep=True, before=2)
            p.add_run().add_picture(str(ROOT / blk[2]), width=Cm(blk[3]))
            add_p(blk[1], indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, size=10, after=2)
        elif k == "figrow":
            ft = d.add_table(rows=2, cols=len(blk[1]))
            ft.alignment = 1
            ft.autofit = False
            for j, (cap, path, w) in enumerate(blk[1]):
                c0, c1 = ft.cell(0, j), ft.cell(1, j)
                c0.width = c1.width = Cm(w + 0.2)
                p0 = c0.paragraphs[0]
                p0.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p0.paragraph_format.first_line_indent = Cm(0)
                p0.paragraph_format.keep_with_next = True
                p0.add_run().add_picture(str(ROOT / path), width=Cm(w))
                p1 = c1.paragraphs[0]
                p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p1.paragraph_format.first_line_indent = Cm(0)
                r = p1.add_run(cap); r.font.size = Pt(10); r.font.name = "Times New Roman"
            ft.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        elif k == "sources":
            for i, s in enumerate(blk[1], 1):
                p = add_p(f"{i}. {s}", indent=False, size=10)
                p.paragraph_format.line_spacing = 1.0
    body.remove(sect)
    body.append(sect)
    d.save(DST)
    print(f"saved {DST}")


if __name__ == "__main__":
    main()
