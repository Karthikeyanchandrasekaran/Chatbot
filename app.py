import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Data Engineering Assistant",
    page_icon="🤖",
    layout="wide"
)

# -----------------------------
# Load CSV files
# -----------------------------
projects = pd.read_csv("de_project_master.csv")
models = pd.read_csv("de_model_master.csv")
notebooks = pd.read_csv("de_synapse_notebook.csv")
pipelines = pd.read_csv("de_pipeline_master.csv")
scripts = pd.read_csv("de_git_script_master.csv")
data_assets = pd.read_csv("de_data_asset_master.csv")
dashboards = pd.read_csv("de_dashboard_master.csv")
documents = pd.read_csv("de_document_master.csv")
relationships = pd.read_csv("de_asset_relationship.csv")

pipeline_metrics = pd.read_csv("de_engineering_metrics.csv")
model_metrics = pd.read_csv("de_model_metric_history.csv")

pipeline_metrics["metric_date"] = pd.to_datetime(pipeline_metrics["metric_date"])
model_metrics["metric_date"] = pd.to_datetime(model_metrics["metric_date"])


# -----------------------------
# Helper functions
# -----------------------------
def format_table(df):
    if df.empty:
        return "No records found."
    return df.to_markdown(index=False)


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


def project_360(project_id):
    project = projects[projects["project_id"] == project_id].iloc[0]

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
{format_table(models[models['project_id'] == project_id])}

### Synapse Notebooks
{format_table(notebooks[notebooks['project_id'] == project_id])}

### ADF Pipelines
{format_table(pipelines[pipelines['project_id'] == project_id])}

### Git Scripts
{format_table(scripts[scripts['project_id'] == project_id])}

### Data Assets
{format_table(data_assets[data_assets['project_id'] == project_id])}

### Dashboards
{format_table(dashboards[dashboards['project_id'] == project_id])}

### Documentation
{format_table(documents[documents['project_id'] == project_id])}
"""


# -----------------------------
# Pipeline Metrics
# -----------------------------
def show_pipeline_metrics(project_id):
    df = pipeline_metrics[pipeline_metrics["project_id"] == project_id]

    if df.empty:
        st.warning("No pipeline metrics found for this project.")
        return

    latest_date = df["metric_date"].max()
    latest = df[df["metric_date"] == latest_date]
    metric_map = latest.set_index("metric_name")["metric_value"].to_dict()

    st.subheader("ADF / Engineering Pipeline Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        success_rate = metric_map.get("pipeline_success_rate", None)
        st.metric(
            "Pipeline Success Rate",
            f"{success_rate}%" if success_rate is not None else "NA"
        )

    with col2:
        st.metric(
            "Total Pipelines",
            len(pipelines[pipelines["project_id"] == project_id])
        )

    with col3:
        st.metric(
            "Total Notebooks",
            len(notebooks[notebooks["project_id"] == project_id])
        )

    fig = px.line(
        df,
        x="metric_date",
        y="metric_value",
        color="metric_name",
        markers=True,
        title="Pipeline / Engineering Metrics Trend"
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Metric Value",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View pipeline metrics data"):
        st.dataframe(df, use_container_width=True)


# -----------------------------
# Model Metrics
# -----------------------------
def show_model_metrics(project_id):
    df = model_metrics[model_metrics["project_id"] == project_id]

    if df.empty:
        st.warning("No model metrics found for this project.")
        return

    latest_date = df["metric_date"].max()
    latest = df[df["metric_date"] == latest_date]
    metric_map = latest.set_index("metric_name")["metric_value"].to_dict()

    st.subheader("Model Performance Metrics")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Accuracy", f"{metric_map.get('accuracy', 0):.2f}")

    with col2:
        st.metric("Precision", f"{metric_map.get('precision', 0):.2f}")

    with col3:
        st.metric("Recall", f"{metric_map.get('recall', 0):.2f}")

    with col4:
        st.metric("F1 Score", f"{metric_map.get('f1_score', 0):.2f}")

    with col5:
        st.metric("AUC", f"{metric_map.get('auc', 0):.2f}")

    fig = px.line(
        df,
        x="metric_date",
        y="metric_value",
        color="metric_name",
        markers=True,
        title="Model Metrics Trend"
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Metric Value",
        yaxis_range=[0, 1],
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View model metrics data"):
        st.dataframe(df, use_container_width=True)


# -----------------------------
# Combined Metrics
# -----------------------------
def show_all_metrics(project_id):
    show_pipeline_metrics(project_id)
    st.divider()
    show_model_metrics(project_id)


# -----------------------------
# Text Answers
# -----------------------------
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


# -----------------------------
# UI
# -----------------------------
st.title("🤖 Data Engineering Assistant")
st.caption(
    "Ask about projects, models, Synapse notebooks, ADF pipelines, Git scripts, data paths, dashboards, wiki, lineage, and metrics."
)

with st.sidebar:
    st.header("Try these prompts")
    st.markdown("""
- Show Customer Conversion project 360
- Show ADF pipeline for Customer Conversion
- Show Git scripts for Customer Conversion
- Show data paths for Customer Conversion
- Show lineage for Customer Conversion
- Show pipeline metrics for Customer Conversion
- Show model metrics for Customer Conversion
- Show all metrics for Customer Conversion
- Show model performance chart for Customer Conversion
- Show AUC trend for Customer Conversion
""")


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! Ask me about projects, ADF, Synapse, Git, data paths, lineage, or metrics."
        }
    ]


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


prompt = st.chat_input(
    "Ask about projects, models, notebooks, ADF, Git, data paths, dashboards, wiki, lineage, or metrics..."
)


if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    q = prompt.lower()
    project = find_project(prompt)

    metric_keywords = [
        "metric",
        "metrics",
        "chart",
        "trend",
        "kpi",
        "performance",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "success rate"
    ]

    pipeline_keywords = [
        "pipeline metric",
        "pipeline metrics",
        "adf metric",
        "adf metrics",
        "engineering metric",
        "engineering metrics",
        "success rate"
    ]

    model_metric_keywords = [
        "model metric",
        "model metrics",
        "model performance",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc"
    ]

    with st.chat_message("assistant"):
        if project is not None and any(word in q for word in metric_keywords):

            if any(word in q for word in model_metric_keywords):
                st.markdown(f"Showing model performance metrics for **{project['project_name']}**")
                show_model_metrics(project["project_id"])

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Showing model performance metrics for {project['project_name']}"
                })

            elif any(word in q for word in pipeline_keywords):
                st.markdown(f"Showing pipeline engineering metrics for **{project['project_name']}**")
                show_pipeline_metrics(project["project_id"])

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Showing pipeline engineering metrics for {project['project_name']}"
                })

            else:
                st.markdown(f"Showing all engineering and model metrics for **{project['project_name']}**")
                show_all_metrics(project["project_id"])

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Showing all metrics for {project['project_name']}"
                })

        else:
            response = answer_question(prompt)
            st.markdown(response)

            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })
