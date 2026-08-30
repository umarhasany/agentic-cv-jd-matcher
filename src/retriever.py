import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_profile(profile_path: str = "data/profile.json") -> dict:
    with open(profile_path, "r", encoding="utf-8") as file:
        return json.load(file)


def flatten_profile_items(profile: dict) -> list[dict]:
    items = []

    for exp in profile.get("experience", []):
        items.append({
            "name": exp["name"],
            "type": exp.get("type", "experience"),
            "description": exp["description"],
            "skills": exp.get("skills", [])
        })

    for project in profile.get("projects", []):
        items.append({
            "name": project["name"],
            "type": project.get("type", "project"),
            "description": project["description"],
            "skills": project.get("skills", [])
        })

    return items


def item_to_text(item: dict) -> str:
    skills = " ".join(item.get("skills", []))
    return f"{item['name']} {item['description']} {skills}"


def normalize(text: str) -> str:
    text = text.lower()
    text = text.replace("/", " ")
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9+#. ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_matched_skills(item: dict, job_description: str) -> list[str]:
    jd_text = normalize(job_description)
    matched = []

    for skill in item.get("skills", []):
        skill_text = normalize(skill)

        if skill_text and skill_text in jd_text:
            matched.append(skill)

    return sorted(set(matched))


def build_match_reason(matched_skills: list[str]) -> str:
    if matched_skills:
        return f"Matched because it shares: {', '.join(matched_skills[:8])}."

    return "Matched based on overall text similarity between the job description and this profile item."


def retrieve_relevant_items(job_description: str, top_k: int = 5) -> list[dict]:
    profile = load_profile()
    items = flatten_profile_items(profile)

    documents = [item_to_text(item) for item in items]
    documents.append(job_description)

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(documents)

    jd_vector = matrix[-1]
    item_vectors = matrix[:-1]

    scores = cosine_similarity(jd_vector, item_vectors).flatten()

    ranked_items = []

    for item, score in zip(items, scores):
        matched_skills = get_matched_skills(item, job_description)

        ranked_items.append({
            **item,
            "similarity_score": round(float(score), 3),
            "matched_skills": matched_skills,
            "match_reason": build_match_reason(matched_skills)
        })

    ranked_items.sort(key=lambda x: x["similarity_score"], reverse=True)
    return ranked_items[:top_k]