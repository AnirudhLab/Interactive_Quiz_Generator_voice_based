import pdfplumber
from docx import Document

def extract_text(file):
    if file.type == "application/pdf":
        with pdfplumber.open(file) as pdf:
            return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    elif file.type in ["text/plain", "text/csv"]:
        return file.read().decode("utf-8")
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    return None
