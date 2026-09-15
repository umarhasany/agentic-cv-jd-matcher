# Agentic CV-JD Matcher

An agentic AI application that compares multiple uploaded CVs against a job description, ranks them using hybrid semantic scoring, retrieves supporting evidence, identifies unsupported skills, and generates a human-reviewable application message.

The project uses **LangGraph** to manage workflow state, conditional routing, evidence-quality evaluation, and adaptive retrieval.

## What the App Does

- Upload multiple CV PDFs
- Paste a job description
- Parse the job description into structured information
- Parse uploaded CVs
- Rank CVs using hybrid semantic scoring
- Select the highest-ranked CV
- Retrieve semantically relevant evidence from the selected CV
- Evaluate whether the retrieved evidence is strong enough
- Retry retrieval automatically when evidence is weak
- Identify supported and missing skills
- Flag overclaim risks
- Generate a short human-reviewable application message
- Display the workflow trace and intermediate decisions

## CV Ranking Logic

Each CV receives a hybrid score based on three signals:

- **55% Semantic Similarity**
- **35% Skill Overlap**
- **10% Domain Match**

The system uses `sentence-transformers/all-MiniLM-L6-v2` to generate embeddings and cosine similarity to compare the meaning of the job description with each CV.

```text
Hybrid Score
=
0.55 × Semantic Similarity
+
0.35 × Skill Overlap
+
0.10 × Domain Match
```

This combines semantic understanding with explicit evidence instead of relying only on keywords.

## Semantic Evidence Retrieval

After selecting the best-ranked CV, the system divides it into overlapping text chunks.

Each chunk is embedded and compared against the job description.

Initial retrieval uses:

```text
top_k = 6
120 words/chunk
30-word overlap
```

If the evidence is judged weak, the agent changes its retrieval strategy:

```text
top_k = 10
90 words/chunk
40-word overlap
```

The smaller chunks provide more focused semantic matches, while greater overlap reduces the chance of important information being split across chunk boundaries.

## Why This Is Agentic

The first version of the project followed a fixed procedural pipeline.

The current version uses a LangGraph `StateGraph` with explicit state, conditional routing, and an adaptive retry loop.

Current workflow:

```text
START
  ↓
JD Parser
  ↓
Hybrid CV Ranker
  ↓
Best CV Selector
  ↓
Semantic Evidence Retriever
  ↓
Evidence Quality Evaluator
  ↓
Is evidence strong enough?
  ├── Yes → Fit + Risk Checker
  │
  └── No → Change retrieval strategy
              ↓
          Retrieve again
              ↓
          Evaluate again
              ↓
          Fit + Risk Checker
  ↓
Message Generator
  ↓
END
```

The agent therefore:

1. Maintains state across multiple processing stages.
2. Observes the quality of retrieved evidence.
3. Makes a decision based on that evidence.
4. Changes its retrieval strategy when necessary.
5. Retries automatically.
6. Prevents unsupported claims from being used in the generated message.

The project does not yet claim to be a fully autonomous LLM agent. Instead, agentic capabilities are being added incrementally through explicit state, decisions, branching, adaptation, and guardrails.

## Evidence Quality Evaluation

Retrieved evidence is evaluated using:

- strongest semantic similarity score,
- average score of the strongest evidence chunks,
- JD skill coverage.

If evidence is sufficiently strong, the workflow continues.

If it is weak on the first attempt, LangGraph routes execution back through the evidence retriever with different retrieval parameters.

This creates a real:

```text
observe → decide → adapt → retry
```

loop.

## Overclaim Protection

The application includes an explicit risk checker.

If a job description requires a skill that is not clearly supported by the selected CV, the system warns against claiming it.

Example:

```text
'anomaly detection' was found in the job description
but not clearly found in the selected CV.
Do not claim it directly.
```

The text matcher uses whole-word and whole-phrase matching to reduce false positives.

For example:

```text
Git      → matches Git
GitHub   → does not automatically imply Git experience
```

## Example Validation

### Strong-Match Test

A ROSEN Master Thesis job description focused on:

- Machine Learning
- Deep Learning
- Computer Vision
- Python
- PyTorch
- TensorFlow
- Anomaly Detection

produced:

```text
Best CV: CV_latest.pdf
Hybrid Score: 65
Semantic Score: 52
Skill Score: 86
Domain Score: 60
```

Evidence quality was strong enough on the first attempt:

```text
Skill coverage: 86%
Retry count: 0
```

The system correctly identified `anomaly detection` as unsupported.

### Weak-Match Test

A deliberately mismatched JD requiring:

- Streamlit
- Git
- Anomaly Detection

produced:

```text
Skill overlap: 0%
Skill coverage: 0%
Retry count: 1
```

The LangGraph workflow automatically triggered the broader evidence-retrieval strategy.

This validated the adaptive retry branch.

## Tech Stack

- Python
- Streamlit
- LangGraph
- PyMuPDF
- sentence-transformers
- scikit-learn
- Pandas
- NumPy

## Project Structure

```text
agentic-cv-jd-matcher/
│
├── app.py
├── requirements.txt
├── README.md
├── PROJECT_REPORT.md
│
└── src/
    ├── __init__.py
    ├── agent.py
    ├── jd_parser.py
    ├── cv_parser.py
    ├── cv_ranker.py
    ├── cv_evidence_retriever.py
    └── text_matching.py
```

## How to Run

Create a virtual environment:

```powershell
python -m venv .venv
```

Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the Streamlit application:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Project Report

For a detailed explanation of the architecture, ranking logic, semantic retrieval, LangGraph workflow, testing, bugs fixed, limitations, and planned improvements, see:

[PROJECT_REPORT.md](PROJECT_REPORT.md)

## Current Limitations

The current version still has several intentional limitations:

- JD skill extraction partly relies on manually defined keyword dictionaries.
- Evidence-quality thresholds are heuristic.
- The workflow currently retries evidence retrieval only once.
- If evidence remains weak after retry, the system does not yet reconsider the next-ranked CV.
- Application-message generation is currently template-based.
- LLM-based autonomous tool selection has not yet been added.

## Next Planned Upgrade

The next major improvement is **CV reconsideration**.

If evidence remains weak after the retrieval retry, the agent will evaluate the next-ranked CV rather than blindly continuing with the first selection.

Planned logic:

```text
Select best-ranked CV
        ↓
Retrieve evidence
        ↓
Evidence strong?
     /       \
   Yes        No
    ↓          ↓
Continue     Retry
               ↓
          Still weak?
             /    \
           No      Yes
           ↓        ↓
       Continue   Try next CV
                    ↓
               Strong match?
                 /      \
               Yes       No
                ↓         ↓
             Use it    Report weak fit
```

This will allow downstream evidence to challenge the original ranking decision.
