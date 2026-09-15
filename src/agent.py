from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from src.jd_parser import parse_jd
from src.cv_ranker import rank_cvs_against_jd
from src.cv_evidence_retriever import retrieve_evidence_from_best_cv
from src.text_matching import find_supported_terms, find_missing_terms


# ---------------------------------------------------------
# 1. LANGGRAPH STATE
# ---------------------------------------------------------

class AgentState(TypedDict, total=False):
    job_description: str
    parsed_cvs: list[dict]

    parsed_jd: dict
    ranked_cvs: list[dict]

    best_cv_filename: str
    best_cv: dict

    cv_evidence: list[dict]

    evidence_attempts: int
    evidence_quality: dict

    score_data: dict
    risks: list[str]

    application_message: str
    workflow_trace: list[dict]

    error: str


# ---------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------

def calculate_cv_skill_overlap(
    parsed_jd: dict,
    best_cv: dict
) -> dict:

    jd_skills = parsed_jd.get(
        "skills",
        []
    )

    cv_text = best_cv["text"]

    supported_skills = find_supported_terms(
        jd_skills,
        cv_text
    )

    missing_skills = find_missing_terms(
        jd_skills,
        cv_text
    )

    if not jd_skills:
        skill_score = 0
    else:
        skill_score = round(
            (
                len(supported_skills)
                / len(jd_skills)
            )
            * 100
        )

    return {
        "skill_score": skill_score,
        "supported_skills": supported_skills,
        "missing_skills": missing_skills
    }


def build_risk_notes(
    missing_skills: list[str]
) -> list[str]:

    if not missing_skills:
        return [
            "No major overclaim risks detected "
            "based on the selected CV."
        ]

    return [
        (
            f"'{skill}' was found in the job description "
            f"but not clearly found in the selected CV. "
            f"Do not claim it directly."
        )
        for skill in missing_skills
    ]


def evaluate_evidence_strength(
    evidence: list[dict],
    parsed_jd: dict
) -> dict:
    """
    Evaluate whether retrieved CV evidence is strong enough.

    Uses:
    - strongest semantic match
    - average of top evidence chunks
    - percentage of JD skills represented in evidence
    """

    if not evidence:
        return {
            "is_strong": False,
            "top_semantic_score": 0,
            "average_top_score": 0,
            "skill_coverage": 0,
            "matched_skills": [],
            "reason": "No evidence chunks were retrieved."
        }

    semantic_scores = [
        item.get("semantic_score", 0)
        for item in evidence
    ]

    top_semantic_score = max(
        semantic_scores
    )

    top_three_scores = sorted(
        semantic_scores,
        reverse=True
    )[:3]

    average_top_score = round(
        sum(top_three_scores)
        / len(top_three_scores)
    )

    jd_skills = set(
        parsed_jd.get(
            "skills",
            []
        )
    )

    evidence_terms = set()

    for item in evidence:
        evidence_terms.update(
            item.get(
                "matched_terms",
                []
            )
        )

    matched_skills = sorted(
        jd_skills.intersection(
            evidence_terms
        )
    )

    if jd_skills:
        skill_coverage = round(
            (
                len(matched_skills)
                / len(jd_skills)
            )
            * 100
        )
    else:
        skill_coverage = 100

    strong_semantic_match = (
        top_semantic_score >= 35
        and average_top_score >= 25
    )

    strong_skill_coverage = (
        skill_coverage >= 50
    )

    is_strong = (
        strong_semantic_match
        and strong_skill_coverage
    )

    if is_strong:
        reason = (
            "Retrieved evidence has sufficient "
            "semantic relevance and JD skill coverage."
        )
    else:
        reason = (
            "Retrieved evidence is weak or incomplete, "
            "so broader retrieval should be attempted."
        )

    return {
        "is_strong": is_strong,
        "top_semantic_score": top_semantic_score,
        "average_top_score": average_top_score,
        "skill_coverage": skill_coverage,
        "matched_skills": matched_skills,
        "reason": reason
    }


def generate_message(
    parsed_jd: dict,
    best_cv_name: str,
    evidence: list[dict],
    score_data: dict
) -> str:

    role = parsed_jd.get(
        "role_title",
        "this position"
    )

    supported = score_data.get(
        "supported_skills",
        []
    )

    if supported:
        supported_text = ", ".join(
            supported[:6]
        )
    else:
        supported_text = (
            "the relevant experience shown in my CV"
        )

    evidence_terms = []

    for item in evidence[:3]:

        matched = ", ".join(
            item.get(
                "matched_terms",
                []
            )
        )

        if matched:
            evidence_terms.append(
                matched
            )

    if evidence_terms:
        evidence_text = "; ".join(
            evidence_terms
        )
    else:
        evidence_text = best_cv_name

    return f"""Dear Hiring Team,

I am currently an M2 student in Data and Intelligence for Smart Systems at Université Claude Bernard Lyon 1, and I am interested in {role}.

Based on the selected CV, my profile matches the role through {supported_text}. The most relevant evidence comes from CV sections related to {evidence_text}.

I would be happy to discuss how my background can contribute to your team.

Kind regards,
Syed Umar Hasany"""


def build_workflow_trace(
    state: AgentState
) -> list[dict]:

    parsed_jd = state["parsed_jd"]
    ranked_cvs = state["ranked_cvs"]
    score_data = state["score_data"]
    risks = state["risks"]

    evidence_attempts = state.get(
        "evidence_attempts",
        0
    )

    evidence_quality = state.get(
        "evidence_quality",
        {}
    )

    best_cv = (
        ranked_cvs[0]["filename"]
        if ranked_cvs
        else "No CV selected"
    )

    trace = [
        {
            "Step": "1",
            "Agent tool": "JD Parser",
            "Action": (
                "Extract role, domains, skills, "
                "contract type, and language requirements."
            ),
            "Output": (
                f"{len(parsed_jd.get('skills', []))} skills, "
                f"{len(parsed_jd.get('domains', []))} domains detected"
            )
        },

        {
            "Step": "2",
            "Agent tool": "Hybrid CV Ranker",
            "Action": (
                "Rank uploaded CVs using semantic similarity, "
                "skill overlap, and domain match."
            ),
            "Output": (
                f"{len(ranked_cvs)} CV(s) ranked"
            )
        },

        {
            "Step": "3",
            "Agent tool": "Best CV Selector",
            "Action": (
                "Select the highest-ranking CV "
                "for the job description."
            ),
            "Output": (
                f"Best CV selected: {best_cv}"
            )
        },

        {
            "Step": "4",
            "Agent tool": "Semantic Evidence Retriever",
            "Action": (
                "Retrieve relevant evidence chunks "
                "from the selected CV."
            ),
            "Output": (
                f"{len(state.get('cv_evidence', []))} "
                f"evidence chunk(s) retained"
            )
        },

        {
            "Step": "5",
            "Agent tool": "Evidence Quality Evaluator",
            "Action": (
                "Judge whether retrieved evidence "
                "is strong enough to continue."
            ),
            "Output": (
                f"Top semantic score: "
                f"{evidence_quality.get('top_semantic_score', 0)}, "
                f"skill coverage: "
                f"{evidence_quality.get('skill_coverage', 0)}%, "
                f"retry count: {evidence_attempts}"
            )
        }
    ]

    if evidence_attempts > 0:
        trace.append({
            "Step": "6",
            "Agent tool": "Evidence Retrieval Retry",
            "Action": (
                "Broaden retrieval because the first "
                "evidence search was too weak."
            ),
            "Output": (
                f"Evidence retrieval retried "
                f"{evidence_attempts} time(s)"
            )
        })

        fit_step = "7"
        message_step = "8"

    else:
        fit_step = "6"
        message_step = "7"

    trace.append({
        "Step": fit_step,
        "Agent tool": "Fit + Risk Checker",
        "Action": (
            "Check which JD skills are supported "
            "or missing in the selected CV."
        ),
        "Output": (
            f"{score_data.get('skill_score', 0)}% "
            f"supported skill overlap, "
            f"{len(risks)} risk note(s)"
        )
    })

    trace.append({
        "Step": message_step,
        "Agent tool": "Message Generator",
        "Action": (
            "Generate a short message using "
            "selected-CV evidence."
        ),
        "Output": "Draft message generated"
    })

    return trace


# ---------------------------------------------------------
# 3. LANGGRAPH NODES
# ---------------------------------------------------------

def parse_jd_node(
    state: AgentState
) -> dict:

    parsed_jd = parse_jd(
        state["job_description"]
    )

    return {
        "parsed_jd": parsed_jd
    }


def rank_cvs_node(
    state: AgentState
) -> dict:

    ranked_cvs = rank_cvs_against_jd(
        job_description=state[
            "job_description"
        ],
        parsed_jd=state[
            "parsed_jd"
        ],
        parsed_cvs=state[
            "parsed_cvs"
        ]
    )

    return {
        "ranked_cvs": ranked_cvs
    }


def select_best_cv_node(
    state: AgentState
) -> dict:

    ranked_cvs = state[
        "ranked_cvs"
    ]

    if not ranked_cvs:
        return {
            "error": (
                "No CV PDFs were uploaded. "
                "Please upload at least one CV."
            )
        }

    best_cv_filename = (
        ranked_cvs[0]["filename"]
    )

    best_cv = next(
        cv
        for cv in state["parsed_cvs"]
        if cv["filename"]
        == best_cv_filename
    )

    return {
        "best_cv_filename":
            best_cv_filename,

        "best_cv":
            best_cv,

        "evidence_attempts":
            0
    }


def retrieve_evidence_node(
    state: AgentState
) -> dict:
    """
    Initial retrieval uses normal settings.

    Retry retrieval uses:
    - more evidence chunks
    - smaller chunks
    - greater overlap
    """

    attempts = state.get(
        "evidence_attempts",
        0
    )

    if attempts == 0:

        top_k = 6
        max_words = 120
        overlap = 30

    else:

        top_k = 10
        max_words = 90
        overlap = 40

    evidence = retrieve_evidence_from_best_cv(
        job_description=state[
            "job_description"
        ],

        parsed_jd=state[
            "parsed_jd"
        ],

        best_cv=state[
            "best_cv"
        ],

        top_k=top_k,
        max_words=max_words,
        overlap=overlap
    )

    return {
        "cv_evidence": evidence
    }


def evaluate_evidence_node(
    state: AgentState
) -> dict:

    evidence_quality = (
        evaluate_evidence_strength(
            evidence=state[
                "cv_evidence"
            ],

            parsed_jd=state[
                "parsed_jd"
            ]
        )
    )

    return {
        "evidence_quality":
            evidence_quality
    }


def increment_retry_node(
    state: AgentState
) -> dict:

    current_attempts = state.get(
        "evidence_attempts",
        0
    )

    return {
        "evidence_attempts":
            current_attempts + 1
    }


def fit_and_risk_check_node(
    state: AgentState
) -> dict:

    score_data = (
        calculate_cv_skill_overlap(
            state["parsed_jd"],
            state["best_cv"]
        )
    )

    risks = build_risk_notes(
        score_data[
            "missing_skills"
        ]
    )

    return {
        "score_data": score_data,
        "risks": risks
    }


def generate_message_node(
    state: AgentState
) -> dict:

    message = generate_message(
        parsed_jd=state[
            "parsed_jd"
        ],

        best_cv_name=state[
            "best_cv_filename"
        ],

        evidence=state[
            "cv_evidence"
        ],

        score_data=state[
            "score_data"
        ]
    )

    return {
        "application_message":
            message
    }


def build_trace_node(
    state: AgentState
) -> dict:

    workflow_trace = (
        build_workflow_trace(
            state
        )
    )

    return {
        "workflow_trace":
            workflow_trace
    }


# ---------------------------------------------------------
# 4. CONDITIONAL ROUTING
# ---------------------------------------------------------

def route_after_cv_selection(
    state: AgentState
) -> str:

    if state.get("error"):
        return "stop"

    return "continue"


def route_after_evidence_evaluation(
    state: AgentState
) -> str:
    """
    Agent decision:

    Strong evidence:
        continue.

    Weak evidence on first attempt:
        retry using broader retrieval.

    Weak evidence after retry:
        continue, but do not retry forever.
    """

    evidence_quality = state.get(
        "evidence_quality",
        {}
    )

    attempts = state.get(
        "evidence_attempts",
        0
    )

    if evidence_quality.get(
        "is_strong",
        False
    ):
        return "continue"

    if attempts < 1:
        return "retry"

    return "continue"


# ---------------------------------------------------------
# 5. BUILD LANGGRAPH
# ---------------------------------------------------------

workflow = StateGraph(
    AgentState
)


workflow.add_node(
    "parse_jd",
    parse_jd_node
)

workflow.add_node(
    "rank_cvs",
    rank_cvs_node
)

workflow.add_node(
    "select_best_cv",
    select_best_cv_node
)

workflow.add_node(
    "retrieve_evidence",
    retrieve_evidence_node
)

workflow.add_node(
    "evaluate_evidence",
    evaluate_evidence_node
)

workflow.add_node(
    "increment_retry",
    increment_retry_node
)

workflow.add_node(
    "fit_and_risk_check",
    fit_and_risk_check_node
)

workflow.add_node(
    "generate_message",
    generate_message_node
)

workflow.add_node(
    "build_trace",
    build_trace_node
)


# Main path

workflow.add_edge(
    START,
    "parse_jd"
)

workflow.add_edge(
    "parse_jd",
    "rank_cvs"
)

workflow.add_edge(
    "rank_cvs",
    "select_best_cv"
)


# CV-selection decision

workflow.add_conditional_edges(
    "select_best_cv",
    route_after_cv_selection,
    {
        "continue":
            "retrieve_evidence",

        "stop":
            END
    }
)


# Evidence retrieval and evaluation

workflow.add_edge(
    "retrieve_evidence",
    "evaluate_evidence"
)


# Evidence-quality decision

workflow.add_conditional_edges(
    "evaluate_evidence",
    route_after_evidence_evaluation,
    {
        "continue":
            "fit_and_risk_check",

        "retry":
            "increment_retry"
    }
)


# Retry loop

workflow.add_edge(
    "increment_retry",
    "retrieve_evidence"
)


# Final stages

workflow.add_edge(
    "fit_and_risk_check",
    "generate_message"
)

workflow.add_edge(
    "generate_message",
    "build_trace"
)

workflow.add_edge(
    "build_trace",
    END
)


agent_graph = workflow.compile()


# ---------------------------------------------------------
# 6. PUBLIC ENTRY POINT
# ---------------------------------------------------------

def run_agent(
    job_description: str,
    parsed_cvs: list[dict]
) -> dict:

    initial_state: AgentState = {
        "job_description":
            job_description,

        "parsed_cvs":
            parsed_cvs,

        "evidence_attempts":
            0
    }

    result = agent_graph.invoke(
        initial_state
    )

    if result.get("error"):
        return {
            "error":
                result["error"]
        }

    return {
        "parsed_jd":
            result["parsed_jd"],

        "ranked_cvs":
            result["ranked_cvs"],

        "best_cv":
            result[
                "best_cv_filename"
            ],

        "cv_evidence":
            result[
                "cv_evidence"
            ],

        "evidence_quality":
            result[
                "evidence_quality"
            ],

        "evidence_attempts":
            result[
                "evidence_attempts"
            ],

        "score_data":
            result[
                "score_data"
            ],

        "risks":
            result[
                "risks"
            ],

        "application_message":
            result[
                "application_message"
            ],

        "workflow_trace":
            result[
                "workflow_trace"
            ]
    }