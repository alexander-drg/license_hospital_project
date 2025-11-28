# extract_txt.py  — modernized to avoid pdfminer3; keeps same API
import os
from typing import List

from pypdf import PdfReader
import mammoth

# Optional: we try to use textract ONLY for legacy .doc if available.
try:
    import textract  # type: ignore
except Exception:
    textract = None


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Returns text from a PDF file using pypdf.
    """
    parts: List[str] = []
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)


def extract_text_from_docx(docx_path: str) -> str:
    """
    Returns text from a DOCX file using mammoth (same as original).
    """
    with open(docx_path, "rb") as docx_file:
        result = mammoth.extract_raw_text(docx_file)
        return result.value or ""


def _extract_text_from_doc(doc_path: str) -> str:
    """
    Best-effort for legacy .doc:
    - If textract is installed, use it (like the original).
    - Otherwise, return empty string (and let caller continue).
    """
    if textract is None:
        # No hard crash if textract isn't present.
        return ""
    try:
        return textract.process(doc_path).decode("utf-8", errors="ignore")
    except Exception:
        return ""


def read_files(file_path: str) -> List[str]:
    """
    Returns a list of texts from multiple files in a directory (PDF, DOCX, DOC).
    Keeps original behavior/signature. Ignores unreadable files gracefully.
    """
    texts: List[str] = []

    # Accept either a directory OR a single file path
    if os.path.isfile(file_path):
        dirname, filename = os.path.split(file_path)
        files = [filename]
        base = dirname
    else:
        base = file_path
        try:
            files = os.listdir(base)
        except FileNotFoundError:
            return []

    for filename in files:
        path = os.path.join(base, filename)
        name = filename.lower()
        try:
            if name.endswith(".pdf"):
                texts.append(extract_text_from_pdf(path))
            elif name.endswith(".docx"):
                texts.append(extract_text_from_docx(path))
            elif name.endswith(".doc"):
                txt = _extract_text_from_doc(path)
                if txt:
                    texts.append(txt)
                else:
                    print(f"Warning: Skipping legacy .doc without textract: {filename}")
            # ignore other extensions silently
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    return texts


if __name__ == "__main__":
    # Example usage: point to a folder or a single file
    sample_path = "./resumes"
    print(read_files(sample_path))
