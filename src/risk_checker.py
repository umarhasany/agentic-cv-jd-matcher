def check_overclaim_risks(parsed_jd: dict, missing_skills: list[str]) -> list[str]:
    """
    Identify skills from the JD that are not strongly supported by the profile database.
    This helps keep generated messages honest.
    """
    risks = []

    high_risk_terms = [
        "production", "kubernetes", "aws", "azure", "gcp",
        "mlops", "langchain", "llamaindex", "agentic ai",
        "ai agent", "agents", "rag", "docker", "ci/cd"
    ]

    for skill in missing_skills:
        if skill.lower() in high_risk_terms:
            risks.append(
                f"Be careful claiming strong hands-on experience with '{skill}' unless you can support it."
            )

    if not risks and missing_skills:
        risks.append(
            "Some JD skills are not strongly supported by the current profile database. Mention willingness to learn instead of claiming expertise."
        )

    if not risks:
        risks.append("No major overclaim risks detected based on the current profile database.")

    return risks