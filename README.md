# Agentic CV-JD Matcher

An agentic AI application that compares multiple uploaded CVs against a job description, selects the best-fit CV, retrieves supporting evidence, identifies missing or weakly supported skills, and generates a human-reviewable application message.

## What the app does

- Upload multiple CV PDFs
- Paste a job description
- Parse the job description
- Parse uploaded CVs
- Rank CVs using hybrid semantic scoring
- Select the best CV
- Retrieve evidence from the selected CV
- Identify supported and missing skills
- Flag overclaim risks
- Generate a short application message

## Why this is agentic

The app follows a tool-based workflow instead of giving a single direct answer.

Workflow:

1. JD Parser
2. CV Parser
3. Hybrid CV Ranker
4. Semantic Evidence Retriever
5. Fit and Risk Checker
6. Message Generator

Each step produces an intermediate result that is shown to the user for review.

## Tech Stack

- Python
- Streamlit
- PyMuPDF
- sentence-transformers
- scikit-learn
- Pandas

## How to run

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py