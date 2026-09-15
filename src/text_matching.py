import re


ALIASES = {
    "python": [
        "python"
    ],

    "pytorch": [
        "pytorch",
        "py torch"
    ],

    "tensorflow": [
        "tensorflow",
        "tensor flow",
        "keras"
    ],

    "machine learning": [
        "machine learning",
        "ml"
    ],

    "deep learning": [
        "deep learning",
        "neural network",
        "neural networks",
        "cnn",
        "transformer"
    ],

    "computer vision": [
        "computer vision",
        "image classification",
        "image processing",
        "visual",
        "opencv"
    ],

    "anomaly detection": [
        "anomaly detection",
        "defect detection",
        "outlier detection",
        "abnormality detection"
    ],

    "industrial inspection": [
        "industrial inspection",
        "visual inspection",
        "quality inspection",
        "quality classification"
    ],

    "quality assurance": [
        "quality assurance",
        "qa",
        "quality evaluation"
    ],

    "self-supervised learning": [
        "self-supervised learning",
        "self supervised learning"
    ],

    "model evaluation": [
        "model evaluation",
        "evaluation metrics",
        "precision",
        "recall",
        "f1"
    ],

    "nlp": [
        "nlp",
        "natural language processing"
    ],

    "llm": [
        "llm",
        "llms",
        "large language model",
        "large language models"
    ],

    "rag": [
        "rag",
        "retrieval augmented generation"
    ],

    "pyspark": [
        "pyspark",
        "py spark"
    ],

    "spark sql": [
        "spark sql"
    ],

    "sql": [
        "sql"
    ],

    "power bi": [
        "power bi",
        "powerbi"
    ],

    "git": [
        "git"
    ],

    "streamlit": [
        "streamlit"
    ]
}


def normalize_text(text: str) -> str:
    """
    Normalize text for reliable term matching.
    """

    text = text.lower()

    # Treat separators as spaces.
    text = text.replace("/", " ")
    text = text.replace("-", " ")

    # Keep letters, numbers, +, # and dots.
    text = re.sub(
        r"[^a-z0-9+#. ]",
        " ",
        text
    )

    # Collapse repeated whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def build_term_pattern(candidate: str) -> str:
    """
    Build a regex that matches a complete word or phrase
    rather than a substring inside another word.

    Example:
        'git' matches:
            Git
            experience with Git

        but does NOT match:
            GitHub
    """

    normalized_candidate = normalize_text(
        candidate
    )

    words = normalized_candidate.split()

    if not words:
        return ""

    phrase_pattern = r"\s+".join(
        re.escape(word)
        for word in words
    )

    return (
        r"(?<![a-z0-9])"
        + phrase_pattern
        + r"(?![a-z0-9])"
    )


def term_found(
    term: str,
    text: str
) -> bool:
    """
    Check whether a term or one of its aliases
    appears as a complete word/phrase in the text.
    """

    normalized_text = normalize_text(
        text
    )

    normalized_term = normalize_text(
        term
    )

    candidates = ALIASES.get(
        normalized_term,
        [normalized_term]
    )

    for candidate in candidates:

        pattern = build_term_pattern(
            candidate
        )

        if (
            pattern
            and re.search(
                pattern,
                normalized_text
            )
        ):
            return True

    return False


def find_supported_terms(
    terms: list[str],
    text: str
) -> list[str]:
    """
    Return terms that are supported by the text.
    """

    supported = []

    for term in terms:

        if term_found(
            term,
            text
        ):
            supported.append(
                term.lower()
            )

    return sorted(
        set(supported)
    )


def find_missing_terms(
    terms: list[str],
    text: str
) -> list[str]:
    """
    Return terms that are not clearly supported
    by the text.
    """

    missing = []

    for term in terms:

        if not term_found(
            term,
            text
        ):
            missing.append(
                term.lower()
            )

    return sorted(
        set(missing)
    )