from sklearn.metrics.pairwise import cosine_similarity

from src.cv_ranker import get_embedding_model
from src.text_matching import find_supported_terms


def chunk_text(text: str, max_words: int = 120, overlap: int = 30) -> list[str]:
    words = text.split()

    if len(words) <= max_words:
        return [" ".join(words)]

    chunks = []
    start = 0

    while start < len(words):
        end = start + max_words
        chunks.append(" ".join(words[start:end]))

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def retrieve_evidence_from_best_cv(
    job_description: str,
    parsed_jd: dict,
    best_cv: dict,
    top_k: int = 6
) -> list[dict]:
    chunks = chunk_text(best_cv["text"])

    jd_terms = []
    jd_terms.extend(parsed_jd.get("skills", []))
    jd_terms.extend(parsed_jd.get("domains", []))

    model = get_embedding_model()

    embeddings = model.encode(
        chunks + [job_description],
        normalize_embeddings=True
    )

    jd_vector = embeddings[-1]
    chunk_vectors = embeddings[:-1]

    scores = cosine_similarity([jd_vector], chunk_vectors)[0]

    evidence = []

    for index, (chunk, score) in enumerate(zip(chunks, scores), start=1):
        matched_terms = find_supported_terms(jd_terms, chunk)

        evidence.append({
            "filename": best_cv["filename"],
            "chunk_id": index,
            "semantic_score": round(float(score) * 100),
            "matched_terms": matched_terms,
            "evidence_text": chunk,
            "why_this_matched": (
                f"Matched because this CV section semantically relates to the JD and mentions: {', '.join(matched_terms[:8])}."
                if matched_terms
                else "Matched based on semantic similarity with the job description."
            )
        })

    evidence.sort(key=lambda x: x["semantic_score"], reverse=True)
    return evidence[:top_k]