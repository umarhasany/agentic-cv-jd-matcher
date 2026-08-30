def calculate_skill_overlap(parsed_jd: dict, relevant_items: list[dict]) -> dict:
    """Calculate overlap between JD skills and retrieved profile item skills."""
    jd_skills = set(skill.lower() for skill in parsed_jd.get("skills", []))

    candidate_skills = set()
    for item in relevant_items:
        for skill in item.get("skills", []):
            candidate_skills.add(skill.lower())

    matched_skills = sorted(jd_skills.intersection(candidate_skills))
    missing_skills = sorted(jd_skills.difference(candidate_skills))

    if not jd_skills:
        skill_score = 0
    else:
        skill_score = round((len(matched_skills) / len(jd_skills)) * 100)

    return {
        "skill_score": skill_score,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills
    }


def calculate_overall_score(relevant_items: list[dict], skill_score: int) -> int:
    """Combine retrieval similarity and skill overlap into one simple score."""
    if not relevant_items:
        return 0

    avg_similarity = sum(item["similarity_score"] for item in relevant_items[:3]) / min(3, len(relevant_items))
    retrieval_score = avg_similarity * 100

    overall = round((0.6 * retrieval_score) + (0.4 * skill_score))
    return min(100, max(0, overall))