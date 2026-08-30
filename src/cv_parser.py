import re
import fitz


SECTION_HEADINGS = [
    "profile", "summary", "education", "experience", "work experience",
    "professional experience", "projects", "publications", "skills",
    "technical skills", "languages", "certificates", "certifications"
]


def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from a Streamlit-uploaded PDF file using PyMuPDF."""
    file_bytes = uploaded_file.read()
    document = fitz.open(stream=file_bytes, filetype="pdf")

    text_parts = []
    for page in document:
        text_parts.append(page.get_text())

    document.close()
    return "\n".join(text_parts)


def clean_text(text: str) -> str:
    """Basic text cleanup."""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def detect_sections(text: str) -> dict:
    """
    Lightweight CV section extraction.
    It looks for common headings and groups text under them.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    sections = {}
    current_section = "general"
    sections[current_section] = []

    for line in lines:
        normalized = line.lower().strip(":").strip()

        if normalized in SECTION_HEADINGS:
            current_section = normalized
            sections.setdefault(current_section, [])
        else:
            sections.setdefault(current_section, []).append(line)

    return {
        section: "\n".join(content).strip()
        for section, content in sections.items()
        if "\n".join(content).strip()
    }


def parse_uploaded_cvs(uploaded_files) -> list[dict]:
    """Parse multiple uploaded CV PDFs."""
    parsed_cvs = []

    for uploaded_file in uploaded_files:
        raw_text = extract_text_from_pdf(uploaded_file)
        cleaned = clean_text(raw_text)
        sections = detect_sections(cleaned)

        parsed_cvs.append({
            "filename": uploaded_file.name,
            "text": cleaned,
            "sections": sections
        })

    return parsed_cvs