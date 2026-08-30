def generate_application_message(parsed_jd: dict, relevant_items: list[dict], score_data: dict) -> str:
    """Generate a short, honest application message using retrieved evidence."""
    role = parsed_jd.get("role_title", "the position")

    top_items = relevant_items[:2]
    evidence = []

    for item in top_items:
        evidence.append(item["name"])

    evidence_text = ", ".join(evidence) if evidence else "my academic and project experience"

    matched_skills = score_data.get("matched_skills", [])
    matched_text = ", ".join(matched_skills[:6]) if matched_skills else "Python, machine learning, and data analysis"

    message = f"""Dear Hiring Team,

I am currently an M2 student in Data and Intelligence for Smart Systems at Université Claude Bernard Lyon 1, and I am interested in {role}.

My profile matches the role through experience with {matched_text}. In particular, my work on {evidence_text} is relevant to the technical and analytical requirements of this position.

I would be happy to discuss how my background can contribute to your team.

Kind regards,
Syed Umar Hasany"""

    return message