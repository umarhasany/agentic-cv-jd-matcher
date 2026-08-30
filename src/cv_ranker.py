from functools import lru_cache

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.text_matching import find_supported_terms


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer(MODEL_NAME)


def semantic_similarity(text_a: str, text_b: str) -> float:
    model = get_embedding_model()

    embeddings = model.encode(
        [text_a, text_b],
        normalize_embeddings=True
    )

    score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return float(score)


def calculate_component_scores(job_description: str, parsed_jd: dict, cv_text: str) -> dict:
    jd_skills = parsed_jd.get("skills", [])
    jd_domains = parsed_jd.get("domains", [])

    semantic = semantic_similarity(job_description, cv_text)

    supported_skills = find_supported_terms(jd_skills, cv_text)
    supported_domains = find_supported_terms(jd_domains, cv_text)

    skill_score = len(supported_skills) / len(jd_skills) if jd_skills else 0
    domain_score = len(supported_domains) / len(jd_domains) if jd_domains else 0

    final_score = (
        0.55 * semantic +
        0.35 * skill_score +
        0.10 * domain_score
    )

    return {
        "semantic_score": round(semantic * 100),
        "skill_score": round(skill_score * 100),
        "domain_score": round(domain_score * 100),
        "final_score": round(final_score * 100),
        "supported_skills": supported_skills,
        "supported_domains": supported_domains
    }


def rank_cvs_against_jd(job_description: str, parsed_jd: dict, parsed_cvs: list[dict]) -> list[dict]:
    if not parsed_cvs:
        return []

    ranked = []

    for cv in parsed_cvs:
        scores = calculate_component_scores(
            job_description=job_description,
            parsed_jd=parsed_jd,
            cv_text=cv["text"]
        )

        ranked.append({
            "filename": cv["filename"],
            "final_score": scores["final_score"],
            "semantic_score": scores["semantic_score"],
            "skill_score": scores["skill_score"],
            "domain_score": scores["domain_score"],
            "supported_skills": scores["supported_skills"],
            "supported_domains": scores["supported_domains"],
            "sections": cv["sections"],
            "preview": cv["text"][:700]
        })

    ranked.sort(key=lambda x: x["final_score"], reverse=True)
    return ranked