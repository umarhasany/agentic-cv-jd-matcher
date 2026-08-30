import re


SKILL_KEYWORDS = [
    "python", "pyspark", "spark", "spark sql", "sql", "pandas", "numpy",
    "scikit-learn", "sklearn", "tensorflow", "keras", "pytorch",
    "machine learning", "deep learning", "computer vision", "nlp",
    "natural language processing", "llm", "llms", "generative ai",
    "rag", "embeddings", "semantic search", "vector search",
    "classification", "regression", "clustering", "anomaly detection",
    "data analysis", "data visualization", "power bi", "excel",
    "api", "apis", "streamlit", "docker", "git"
]

DOMAIN_KEYWORDS = {
    "data science": ["data science", "data scientist", "data analysis", "analytics"],
    "machine learning": ["machine learning", "ml", "predictive model", "scoring"],
    "deep learning": ["deep learning", "neural network", "cnn", "transformer"],
    "computer vision": ["computer vision", "image", "video", "visual", "inspection", "ndt"],
    "nlp": ["nlp", "natural language processing", "text", "language", "llm"],
    "generative ai": ["generative ai", "genai", "llm", "agentic ai", "ai agent"],
    "data engineering": ["data engineering", "etl", "pipeline", "spark", "pyspark"],
    "business intelligence": ["dashboard", "power bi", "reporting", "visualization"],
    "risk / finance": ["risk", "finance", "banking", "credit", "insurance"],
    "industrial ai": ["industrial", "inspection", "quality assurance", "anomaly detection", "ndt"]
}

CONTRACT_KEYWORDS = {
    "internship": ["internship", "intern", "stage", "stagiaire"],
    "work-study": ["work-study", "alternance", "apprenticeship", "apprenti"],
    "thesis": ["master thesis", "master's thesis", "thesis", "mémoire"]
}

LANGUAGE_KEYWORDS = {
    "English": ["english", "anglais"],
    "French": ["french", "français"],
    "German": ["german", "allemand"]
}


def normalize_text(text: str) -> str:
    """Lowercase and normalize spacing."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def find_keywords(text: str, keywords: list[str]) -> list[str]:
    """Return keywords found in the text."""
    normalized = normalize_text(text)
    found = []

    for keyword in keywords:
        pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
        if re.search(pattern, normalized):
            found.append(keyword)

    return sorted(set(found))


def find_domains(text: str) -> list[str]:
    """Detect broad domains from a job description."""
    normalized = normalize_text(text)
    domains = []

    for domain, keywords in DOMAIN_KEYWORDS.items():
        for keyword in keywords:
            if keyword in normalized:
                domains.append(domain)
                break

    return sorted(set(domains))


def find_contract_type(text: str) -> list[str]:
    """Detect contract type such as internship, work-study, or thesis."""
    normalized = normalize_text(text)
    contract_types = []

    for contract_type, keywords in CONTRACT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in normalized:
                contract_types.append(contract_type)
                break

    return sorted(set(contract_types))


def find_languages(text: str) -> list[str]:
    """Detect requested languages."""
    normalized = normalize_text(text)
    languages = []

    for language, keywords in LANGUAGE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in normalized:
                languages.append(language)
                break

    return sorted(set(languages))


def extract_role_title(text: str) -> str:
    """
    Basic role title extraction.
    This is intentionally simple for MVP.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for line in lines[:10]:
        lowered = line.lower()
        if any(word in lowered for word in ["intern", "internship", "alternance", "thesis", "data scientist", "machine learning", "ai"]):
            return line[:120]

    return "Unknown role"


def parse_jd(job_description: str) -> dict:
    """Parse a job description into structured information."""
    return {
        "role_title": extract_role_title(job_description),
        "skills": find_keywords(job_description, SKILL_KEYWORDS),
        "domains": find_domains(job_description),
        "contract_type": find_contract_type(job_description),
        "languages": find_languages(job_description),
        "raw_text": job_description
    }