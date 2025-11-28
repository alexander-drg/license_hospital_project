import os
from typing import List
from pypdf import PdfReader
import mammoth

try:
    import textract
except Exception:
    textract = None

def extract_text_from_pdf(pdf_path: str) -> str:
    out = []
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        for p in reader.pages:
            out.append(p.extract_text() or "")
    return "\n".join(out)

def extract_text_from_docx(docx_path: str) -> str:
    with open(docx_path, "rb") as docx_file:
        return mammoth.extract_raw_text(docx_file).value or ""

def _extract_text_from_doc(doc_path: str) -> str:
    if textract is None: return ""
    try:
        return textract.process(doc_path).decode("utf-8", errors="ignore")
    except Exception:
        return ""

def read_files(path: str) -> List[str]:
    texts: List[str] = []
    if os.path.isfile(path):
        base, files = os.path.dirname(path), [os.path.basename(path)]
    else:
        base = path
        try: files = os.listdir(base)
        except FileNotFoundError: return []

    for fn in files:
        fp = os.path.join(base, fn)
        name = fn.lower()
        try:
            if name.endswith(".pdf"): texts.append(extract_text_from_pdf(fp))
            elif name.endswith(".docx"): texts.append(extract_text_from_docx(fp))
            elif name.endswith(".doc"):
                t = _extract_text_from_doc(fp)
                if t: texts.append(t)
        except Exception as e:
            print(f"Error reading {fn}: {e}")
    return texts
