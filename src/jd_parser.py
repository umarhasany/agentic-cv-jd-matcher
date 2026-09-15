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
    "data science": [
        "data science",
        "data scientist",
        "data analysis",
        "analytics"
    ],

    "machine learning": [
        "machine learning",
        "ml",
        "predictive model",
        "scoring"
    ],

    "deep learning": [
        "deep learning",
        "neural network",
        "cnn",
        "transformer"
    ],

    "computer vision": [
        "computer vision",
        "image",
        "video",
        "visual",
        "inspection",
        "ndt"
    ],

    "nlp": [
        "nlp",
        "natural language processing",
        "text",
        "language",
        "llm"
    ],

    "generative ai": [
        "generative ai",
        "genai",
        "llm",
        "agentic ai",
        "ai agent"
    ],

    "data engineering": [
        "data engineering",
        "etl",
        "pipeline",
        "spark",
        "pyspark"
    ],

    "business intelligence": [
        "dashboard",
        "power bi",
        "reporting",
        "visualization"
    ],

    "risk / finance": [
        "risk",
        "finance",
        "banking",
        "credit",
        "insurance"
    ],

    "industrial ai": [
        "industrial",
        "inspection",
        "quality assurance",
        "anomaly detection",
        "ndt"
    ]
}


CONTRACT_KEYWORDS = {
    "internship": [
        "internship",
        "intern",
        "stage",
        "stagiaire"
    ],

    "work-study": [
        "work-study",
        "alternance",
        "apprenticeship",
        "apprenti"
    ],

    "thesis": [
        "master thesis",
        "master's thesis",
        "thesis",
        "mémoire"
    ]
}


LANGUAGE_KEYWORDS = {
    "English": [
        "english",
        "anglais"
    ],

    "French": [
        "french",
        "français"
    ],

    "German": [
        "german",
        "allemand"
    ]
}


def normalize_text(text: str) -> str:
    """
    Lowercase text and normalize whitespace.
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def find_keywords(text: str, keywords: list[str]) -> list[str]:
    """
    Return skill keywords explicitly found in the job description.

    Word boundaries prevent partial-word matches.
    """
    normalized = normalize_text(text)
    found = []

    for keyword in keywords:
        pattern = r"\b" + re.escape(keyword.lower()) + r"\b"

        if re.search(pattern, normalized):
            found.append(keyword)

    return sorted(set(found))


def find_domains(text: str) -> list[str]:
    """
    Detect broad domains from the job description.
    """
    normalized = normalize_text(text)
    domains = []

    for domain, keywords in DOMAIN_KEYWORDS.items():

        for keyword in keywords:
            pattern = r"\b" + re.escape(keyword.lower()) + r"\b"

            if re.search(pattern, normalized):
                domains.append(domain)
                break

    return sorted(set(domains))


def find_contract_type(text: str) -> list[str]:
    """
    Detect contract type such as internship,
    work-study, or thesis.
    """
    normalized = normalize_text(text)
    contract_types = []

    for contract_type, keywords in CONTRACT_KEYWORDS.items():

        for keyword in keywords:
            pattern = r"\b" + re.escape(keyword.lower()) + r"\b"

            if re.search(pattern, normalized):
                contract_types.append(contract_type)
                break

    return sorted(set(contract_types))


def find_languages(text: str) -> list[str]:
    """
    Detect explicitly mentioned languages.

    Word boundaries prevent false matches such as:
        German -> Germany
    """
    normalized = normalize_text(text)
    languages = []

    for language, keywords in LANGUAGE_KEYWORDS.items():

        for keyword in keywords:
            pattern = r"\b" + re.escape(keyword.lower()) + r"\b"

            if re.search(pattern, normalized):
                languages.append(language)
                break

    return sorted(set(languages))


def extract_role_title(text: str) -> str:
    """
    Extract a likely role title from the beginning
    of the job description.

    Also handles cases where the first paragraph of
    the JD has accidentally been pasted onto the
    same line as the title.
    """
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    role_indicators = [
        "intern",
        "internship",
        "alternance",
        "thesis",
        "data scientist",
        "machine learning",
        "ai"
    ]

    for line in lines[:10]:
        lowered = line.lower()

        if any(
            indicator in lowered
            for indicator in role_indicators
        ):

            # Remove common beginning-of-description text
            # if it appears on the same line as the title.
            line = re.split(
                r"\b(?:"
                r"we are looking|"
                r"we're looking|"
                r"we are seeking|"
                r"we seek|"
                r"your role|"
                r"your mission|"
                r"about the role|"
                r"about this role|"
                r"the role involves|"
                r"you will"
                r")\b",
                line,
                maxsplit=1,
                flags=re.IGNORECASE
            )[0]

            title = line.strip(" :-")

            if title:
                return title[:120]

    return "Unknown role"


def parse_jd(job_description: str) -> dict:
    """
    Parse a job description into structured information.
    """
    return {
        "role_title": extract_role_title(
            job_description
        ),

        "skills": find_keywords(
            job_description,
            SKILL_KEYWORDS
        ),

        "domains": find_domains(
            job_description
        ),

        "contract_type": find_contract_type(
            job_description
        ),

        "languages": find_languages(
            job_description
        ),

        "raw_text": job_description
    }