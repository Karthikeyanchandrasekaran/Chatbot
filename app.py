import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Data Engineering Assistant",
    page_icon="🤖",
    layout="wide"
)

projects = pd.read_csv("de_project_master.csv")
models = pd.read_csv("de_model_master.csv")
notebooks = pd.read_csv("de_synapse_notebook.csv")
pipelines = pd.read_csv("de_pipeline_master.csv")
scripts = pd.read_csv("de_git_script_master.csv")
data_assets = pd.read_csv("de_data_asset_master.csv")
dashboards = pd.read_csv("de_dashboard_master.csv")
documents = pd.read_csv("de_document_master.csv")
relationships = pd.read_csv("de_asset_relationship.csv")

st.title("🤖 Data Engineering Assistant")
st.caption("Ask about projects, models, Synapse notebooks, ADF pipelines, Git scripts, data assets, dashboards, wiki, and lineage.")

def find_project(question):
    q = question.lower()

    for _, row in projects.iterrows():
        if row["project_name"].lower() in q:
            return row

    for _, row in models.iterrows():
        if row["model_name"].lower() in q:
            project_id = row["project_id"]
            return projects[projects["project_id"] == project_id].iloc[0]

    return None

def format_table(df):
    if df.empty:
        return "No records found."
    return df.to_markdown(index=False)

def project_360(project_id):
    project = projects[projects["project_id"] == project_id].iloc[0]

    project_models = models[models["project_id"] == project_id]
    project_notebooks = notebooks[notebooks["project_id"] == project_id]
    project_pipelines = pipelines[pipelines["project_id"] == project_id]
    project_scripts = scripts[scripts["project_id"] == project_id]
    project_data = data_assets[data_assets["project_id"] == project_id]
    project_dashboards = dashboards[dashboards["project_id"] == project_id]
    project_docs = documents[documents["project_id"] == project_id]

    return f"""
## Project 360: {project['project_name']}

### Project Overview

| Field | Value |
|---|---|
| Business Objective | {project['business_objective']} |
| Business Owner | {project['business_owner']} |
| Engineering Owner | {project['engineering_owner']} |
| Status | {project['status']} |
| Created Date | {project['created_date']} |

### Models
{format_table(project_models)}

### Synapse Notebooks
{format_table(project_notebooks)}

### ADF Pipelines
{format_table(project_pipelines)}

### Git Scripts
{format_table(project_scripts)}

### Data Assets
{format_table(project_data)}

### Dashboards
{format_table(project_dashboards)}

### Documentation
{format_table(project_docs)}
"""

def answer_question(question):
    q = question.lower()
    project = find_project(question)

    if project is None:
        available = "\n".join([f"- {p}" for p in projects["project_name"].tolist()])
        return f"I could not identify the project/model. Available projects:\n\n{available}"

    project_id = project["project_id"]
    project_name = project["project_name"]

    if "360" in q or "complete" in q or "overview" in q or "details" in q:
        return project_360(project_id)

    if "model" in q:
        return f"### Models for {project_name}\n\n{format_table(models[models['project_id'] == project_id])}"

    if "notebook" in q or "synapse" in q:
        return f"### Synapse Notebooks for {project_name}\n\n{format_table(notebooks[notebooks['project_id'] == project_id])}"

    if "pipeline" in q or "adf" in q:
        return f"### ADF Pipelines for {project_name}\n\n{format_table(pipelines[pipelines['project_id'] == project_id])}"

    if "git" in q or "script" in q or "repo" in q or "code" in q:
        return f"### Git Scripts for {project_name}\n\n{format_table(scripts[scripts['project_id'] == project_id])}"

    if "data" in q or "path" in q or "gen1" in q or "gen2" in q:
        return f"### Data Assets for {project_name}\n\n{format_table(data_assets[data_assets['project_id'] == project_id])}"

    if "dashboard" in q or "power bi" in q:
        return f"### Dashboards for {project_name}\n\n{format_table(dashboards[dashboards['project_id'] == project_id])}"

    if "wiki" in q or "document" in q or "documentation" in q:
        return f"### Documentation for {project_name}\n\n{format_table(documents[documents['project_id'] == project_id])}"

    if "lineage" in q or "relationship" in q or "graph" in q:
        return f"### Engineering Relationships for {project_name}\n\n{format_table(relationships[relationships['project_id'] == project_id])}"

    return project_360(project_id)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! Ask me: Show Customer Conversion project 360, Which ADF pipeline runs conversion, or Show Git scripts for Customer Conversion."
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask about projects, models, notebooks, ADF, Git, data paths, dashboard, wiki, or lineage...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    response = answer_question(prompt)

    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)
