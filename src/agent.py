from src.jd_parser import parse_jd
from src.cv_ranker import rank_cvs_against_jd
from src.cv_evidence_retriever import retrieve_evidence_from_best_cv
from src.text_matching import find_supported_terms, find_missing_terms


def calculate_cv_skill_overlap(parsed_jd: dict, best_cv: dict) -> dict:
    jd_skills = parsed_jd.get("skills", [])
    cv_text = best_cv["text"]

    supported_skills = find_supported_terms(jd_skills, cv_text)
    missing_skills = find_missing_terms(jd_skills, cv_text)

    if not jd_skills:
        skill_score = 0
    else:
        skill_score = round((len(supported_skills) / len(jd_skills)) * 100)

    return {
        "skill_score": skill_score,
        "supported_skills": supported_skills,
        "missing_skills": missing_skills
    }


def build_risk_notes(missing_skills: list[str]) -> list[str]:
    if not missing_skills:
        return ["No major overclaim risks detected based on the selected CV."]

    return [
        f"'{skill}' was found in the job description but not clearly found in the selected CV. Do not claim it directly."
        for skill in missing_skills
    ]


def generate_message(parsed_jd: dict, best_cv_name: str, evidence: list[dict], score_data: dict) -> str:
    role = parsed_jd.get("role_title", "this position")

    supported = score_data.get("supported_skills", [])
    supported_text = ", ".join(supported[:6]) if supported else "the relevant experience shown in my CV"

    evidence_terms = []
    for item in evidence[:3]:
        matched = ", ".join(item.get("matched_terms", []))
        if matched:
            evidence_terms.append(matched)

    evidence_text = "; ".join(evidence_terms) if evidence_terms else best_cv_name

    return f"""Dear Hiring Team,

I am currently an M2 student in Data and Intelligence for Smart Systems at Université Claude Bernard Lyon 1, and I am interested in {role}.

Based on the selected CV, my profile matches the role through {supported_text}. The most relevant evidence comes from CV sections related to {evidence_text}.

I would be happy to discuss how my background can contribute to your team.

Kind regards,
Syed Umar Hasany"""


def build_workflow_trace(parsed_jd, ranked_cvs, score_data, risks) -> list[dict]:
    best_cv = ranked_cvs[0]["filename"] if ranked_cvs else "No CV selected"

    return [
        {
            "Step": "1",
            "Agent tool": "JD Parser",
            "Action": "Extract role, domains, skills, contract type, and language requirements.",
            "Output": f"{len(parsed_jd.get('skills', []))} skills, {len(parsed_jd.get('domains', []))} domains detected"
        },
        {
            "Step": "2",
            "Agent tool": "CV Parser",
            "Action": "Read and extract text from uploaded CV PDFs.",
            "Output": f"{len(ranked_cvs)} CV(s) processed"
        },
        {
            "Step": "3",
            "Agent tool": "Hybrid CV Ranker",
            "Action": "Rank each CV using semantic similarity, skill overlap, and domain match.",
            "Output": f"Best CV selected: {best_cv}"
        },
        {
            "Step": "4",
            "Agent tool": "Semantic Evidence Retriever",
            "Action": "Retrieve the most relevant evidence chunks from the selected CV.",
            "Output": "Evidence chunks retrieved from selected CV"
        },
        {
            "Step": "5",
            "Agent tool": "Fit + Risk Checker",
            "Action": "Check which JD skills are supported or missing in the selected CV.",
            "Output": f"{score_data.get('skill_score', 0)}% supported skill overlap, {len(risks)} risk note(s)"
        },
        {
            "Step": "6",
            "Agent tool": "Message Generator",
            "Action": "Generate a short message using only selected-CV evidence.",
            "Output": "Draft message generated"
        }
    ]


def run_agent(job_description: str, parsed_cvs: list[dict]) -> dict:
    parsed_jd = parse_jd(job_description)
    ranked_cvs = rank_cvs_against_jd(
        job_description=job_description,
        parsed_jd=parsed_jd,
        parsed_cvs=parsed_cvs
    )

    if not ranked_cvs:
        return {
            "error": "No CV PDFs were uploaded. Please upload at least one CV."
        }

    best_cv_filename = ranked_cvs[0]["filename"]

    best_cv = next(
        cv for cv in parsed_cvs
        if cv["filename"] == best_cv_filename
    )

    evidence = retrieve_evidence_from_best_cv(
        job_description=job_description,
        parsed_jd=parsed_jd,
        best_cv=best_cv,
        top_k=6
    )

    score_data = calculate_cv_skill_overlap(parsed_jd, best_cv)
    risks = build_risk_notes(score_data["missing_skills"])

    message = generate_message(
        parsed_jd=parsed_jd,
        best_cv_name=best_cv_filename,
        evidence=evidence,
        score_data=score_data
    )

    workflow_trace = build_workflow_trace(
        parsed_jd=parsed_jd,
        ranked_cvs=ranked_cvs,
        score_data=score_data,
        risks=risks
    )

    return {
        "parsed_jd": parsed_jd,
        "ranked_cvs": ranked_cvs,
        "best_cv": best_cv_filename,
        "cv_evidence": evidence,
        "score_data": score_data,
        "risks": risks,
        "application_message": message,
        "workflow_trace": workflow_trace
    }