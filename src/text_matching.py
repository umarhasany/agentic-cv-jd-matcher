import re


ALIASES = {
    "python": ["python"],
    "pytorch": ["pytorch", "py torch"],
    "tensorflow": ["tensorflow", "tensor flow", "keras"],
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning", "neural network", "neural networks", "cnn", "transformer"],
    "computer vision": ["computer vision", "image classification", "image processing", "visual", "opencv"],
    "anomaly detection": ["anomaly detection", "defect detection", "outlier detection", "abnormality detection"],
    "industrial inspection": ["industrial inspection", "visual inspection", "quality inspection", "quality classification"],
    "quality assurance": ["quality assurance", "qa", "quality evaluation"],
    "self-supervised learning": ["self-supervised learning", "self supervised learning"],
    "model evaluation": ["model evaluation", "evaluation metrics", "precision", "recall", "f1"],
    "nlp": ["nlp", "natural language processing"],
    "llm": ["llm", "llms", "large language model"],
    "rag": ["rag", "retrieval augmented generation"],
    "pyspark": ["pyspark", "spark"],
    "spark sql": ["spark sql"],
    "sql": ["sql"],
    "power bi": ["power bi", "powerbi"],
}


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("/", " ")
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9+#. ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def term_found(term: str, text: str) -> bool:
    normalized_text = normalize_text(text)
    normalized_term = normalize_text(term)

    candidates = ALIASES.get(normalized_term, [normalized_term])

    for candidate in candidates:
        candidate = normalize_text(candidate)
        if candidate and candidate in normalized_text:
            return True

    return False


def find_supported_terms(terms: list[str], text: str) -> list[str]:
    supported = []

    for term in terms:
        if term_found(term, text):
            supported.append(term.lower())

    return sorted(set(supported))


def find_missing_terms(terms: list[str], text: str) -> list[str]:
    missing = []

    for term in terms:
        if not term_found(term, text):
            missing.append(term.lower())

    return sorted(set(missing))