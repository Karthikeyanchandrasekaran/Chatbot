import streamlit as st
import pandas as pd

st.set_page_config(page_title="DE Engineering Assistant", page_icon="🤖", layout="wide")

df = pd.read_csv("engineering_assets.csv")

st.title("🤖 Data Engineering Assistant")
st.caption("Ask about models, Synapse notebooks, ADF pipelines, Git repos, data paths, dashboards, and wiki pages.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me something like: Where is the scoring notebook for Customer Conversion?"}
    ]

def find_project(question: str):
    q = question.lower()
    for _, row in df.iterrows():
        if row["project_name"].lower() in q or row["model_name"].lower() in q:
            return row
    return None

def answer_question(question: str):
    q = question.lower()
    row = find_project(question)

    if row is None:
        projects = "\n".join([f"- {p}" for p in df["project_name"].unique()])
        return f"I could not identify the project/model. Available projects:\n{projects}"

    if "training notebook" in q or "train notebook" in q:
        return f"Training notebook for **{row['project_name']}** is **{row['training_notebook']}**."

    if "scoring notebook" in q or "score notebook" in q:
        return f"Scoring notebook for **{row['project_name']}** is **{row['scoring_notebook']}**."

    if "pipeline" in q or "adf" in q:
        return f"ADF pipeline for **{row['project_name']}** is **{row['adf_pipeline']}**."

    if "training data" in q:
        return f"Training data path for **{row['project_name']}** is:\n\n`{row['training_data_path']}`"

    if "scoring data" in q:
        return f"Scoring data path for **{row['project_name']}** is:\n\n`{row['scoring_data_path']}`"

    if "model artifact" in q or "model path" in q:
        return f"Model artifact path for **{row['project_name']}** is:\n\n`{row['model_artifact_path']}`"

    if "git" in q or "repo" in q:
        return f"Git repo for **{row['project_name']}** is **{row['git_repo']}**."

    if "dashboard" in q or "power bi" in q:
        return f"Dashboard for **{row['project_name']}** is **{row['dashboard']}**."

    if "wiki" in q or "document" in q:
        return f"Wiki/documentation for **{row['project_name']}** is **{row['wiki']}**."

    return f"""
### {row['project_name']} - Project 360

| Area | Details |
|---|---|
| Model | {row['model_name']} |
| Training Notebook | {row['training_notebook']} |
| Scoring Notebook | {row['scoring_notebook']} |
| ADF Pipeline | {row['adf_pipeline']} |
| Training Data | `{row['training_data_path']}` |
| Scoring Data | `{row['scoring_data_path']}` |
| Model Artifact | `{row['model_artifact_path']}` |
| Git Repo | {row['git_repo']} |
| Dashboard | {row['dashboard']} |
| Wiki | {row['wiki']} |
| Owner | {row['owner']} |
"""

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask about a model, notebook, pipeline, Git repo, data path, dashboard, or wiki...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    response = answer_question(prompt)

    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)
