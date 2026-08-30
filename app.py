import streamlit as st
import pandas as pd

from src.agent import run_agent
from src.cv_parser import parse_uploaded_cvs


st.set_page_config(
    page_title="Agentic CV-JD Matcher",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Agentic AI CV-JD Matcher")

st.write(
    "Upload multiple CV PDFs and paste a job description. The agent parses the CVs, "
    "compares them against the job description, selects the best CV, retrieves evidence "
    "from that CV, checks unsupported claims, and generates a human-reviewable message."
)

with st.expander("What makes this agentic?"):
    st.write(
        "The system uses the uploaded CVs as its context. It runs a tool-based workflow: "
        "JD parsing, CV parsing, hybrid CV ranking, semantic evidence retrieval, "
        "risk checking, and message generation."
    )

uploaded_cvs = st.file_uploader(
    "Upload one or more CV PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

job_description = st.text_area(
    "Paste job description here",
    height=300,
    placeholder="Paste the full job description..."
)

if st.button("Analyze job description"):
    if not uploaded_cvs:
        st.warning("Please upload at least one CV PDF.")
    elif not job_description.strip():
        st.warning("Please paste a job description first.")
    else:
        with st.spinner("Parsing CVs and running agent workflow..."):
            parsed_cvs = parse_uploaded_cvs(uploaded_cvs)
            result = run_agent(job_description, parsed_cvs=parsed_cvs)

        if "error" in result:
            st.error(result["error"])
        else:
            parsed = result["parsed_jd"]
            score_data = result["score_data"]

            st.subheader("1. Agent workflow trace")

            workflow_df = pd.DataFrame(result["workflow_trace"])
            st.dataframe(workflow_df, width="stretch", hide_index=True)

            st.subheader("2. Parsed job description")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Role title:**", parsed["role_title"])
                st.write("**Domains:**", ", ".join(parsed["domains"]) or "Not detected")
                st.write("**Contract type:**", ", ".join(parsed["contract_type"]) or "Not detected")

            with col2:
                st.write("**Languages:**", ", ".join(parsed["languages"]) or "Not detected")
                st.write("**Extracted skills:**", ", ".join(parsed["skills"]) or "Not detected")

            st.subheader("3. Uploaded CV ranking")

            cv_rank_df = pd.DataFrame([
                {
                    "CV filename": cv["filename"],
                    "Final hybrid score": cv["final_score"],
                    "Semantic score": cv["semantic_score"],
                    "Skill score": cv["skill_score"],
                    "Domain score": cv["domain_score"],
                    "Supported skills": ", ".join(cv["supported_skills"]) or "None",
                    "Supported domains": ", ".join(cv["supported_domains"]) or "None"
                }
                for cv in result["ranked_cvs"]
            ])

            st.dataframe(cv_rank_df, width="stretch", hide_index=True)

            st.subheader("4. Selected best CV")

            st.success(result["best_cv"])

            best_ranked_cv = result["ranked_cvs"][0]

            st.info(
                f"This CV was selected because it had the highest final hybrid score "
                f"({best_ranked_cv['final_score']}). The score combines semantic similarity "
                f"({best_ranked_cv['semantic_score']}), supported skill overlap "
                f"({best_ranked_cv['skill_score']}), and domain match "
                f"({best_ranked_cv['domain_score']})."
            )

            st.write("**Why this CV is a strong match:**")
            st.write(
                f"- It supports these JD skills: "
                f"{', '.join(best_ranked_cv['supported_skills']) or 'None detected'}"
            )
            st.write(
                f"- It matches these JD domains: "
                f"{', '.join(best_ranked_cv['supported_domains']) or 'None detected'}"
            )

            st.subheader("5. Fit and gap analysis")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Supported skill overlap", f"{score_data['skill_score']}%")

            with col2:
                st.metric("Missing skills", len(score_data["missing_skills"]))

            st.write("**Supported skills found in selected CV:**")
            st.write(", ".join(score_data["supported_skills"]) or "None")

            st.write("**Missing or weakly supported skills:**")
            st.write(", ".join(score_data["missing_skills"]) or "None")

            st.subheader("6. Evidence from selected CV")

            evidence_df = pd.DataFrame([
                {
                    "CV filename": item["filename"],
                    "Chunk": item["chunk_id"],
                    "Semantic score": item["semantic_score"],
                    "Matched terms": ", ".join(item.get("matched_terms", [])) or "Semantic similarity only",
                    "Why this matched": item["why_this_matched"]
                }
                for item in result["cv_evidence"]
            ])

            st.dataframe(evidence_df, width="stretch", hide_index=True)

            for item in result["cv_evidence"]:
                with st.expander(f"Evidence - chunk {item['chunk_id']}"):
                    st.write(f"**Why this matched:** {item['why_this_matched']}")
                    st.write(f"**Matched terms:** {', '.join(item['matched_terms']) or 'Semantic similarity only'}")
                    st.write(item["evidence_text"])

            st.subheader("7. Overclaim risk check")

            for risk in result["risks"]:
                st.write(f"- {risk}")

            st.subheader("8. Generated application message")

            st.text_area(
                "Draft message for human review",
                value=result["application_message"],
                height=260
            )