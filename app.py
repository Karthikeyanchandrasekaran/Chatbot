import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Data Engineering Assistant",
    page_icon="🤖",
    layout="wide"
)

# --------------------------------------------------
# Load CSV files
# --------------------------------------------------
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


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def format_table(df):
    if df.empty:
        return "No records found."
    return df.to_markdown(index=False)


def build_search_index():
    rows = []

    tables = {
        "Project": projects,
        "Model": models,
        "Synapse Notebook": notebooks,
        "ADF Pipeline": pipelines,
        "Git Script": scripts,
        "Data Asset": data_assets,
        "Dashboard": dashboards,
        "Document": documents,
        "Relationship": relationships
    }

    for table_name, df in tables.items():
        for _, row in df.iterrows():
            text = " ".join([str(v) for v in row.values])
            project_id = row["project_id"] if "project_id" in row.index else None

            rows.append({
                "table_name": table_name,
                "project_id": project_id,
                "search_text": text,
                "record": row.to_dict()
            })

    return pd.DataFrame(rows)


search_index = build_search_index()
vectorizer = TfidfVectorizer(stop_words="english")
search_matrix = vectorizer.fit_transform(search_index["search_text"])


def dynamic_search(question, top_n=5):
    query_vector = vectorizer.transform([question])
    scores = cosine_similarity(query_vector, search_matrix).flatten()

    results = search_index.copy()
    results["score"] = scores
    results = results.sort_values("score", ascending=False).head(top_n)

    return results[results["score"] > 0.05]


def find_project(question):
    q = question.lower()

    for _, row in projects.iterrows():
        if str(row["project_name"]).lower() in q:
            return row

    for _, row in models.iterrows():
        if str(row["model_name"]).lower() in q:
            project_id = row["project_id"]
            match = projects[projects["project_id"] == project_id]
            if not match.empty:
                return match.iloc[0]

    results = dynamic_search(question, top_n=1)
    if not results.empty:
        project_id = results.iloc[0]["project_id"]
        if pd.notna(project_id):
            match = projects[projects["project_id"] == int(project_id)]
            if not match.empty:
                return match.iloc[0]

    return None


# --------------------------------------------------
# Project 360
# --------------------------------------------------
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

### Relationships / Lineage
{format_table(relationships[relationships['project_id'] == project_id])}
"""


# --------------------------------------------------
# Metrics
# --------------------------------------------------
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
        success_rate = metric_map.get("pipeline_success_rate")
        st.metric(
            "Pipeline Success Rate",
            f"{success_rate}%" if success_rate is not None else "NA"
        )

    with col2:
        st.metric("Total Pipelines", len(pipelines[pipelines["project_id"] == project_id]))

    with col3:
        st.metric("Total Notebooks", len(notebooks[notebooks["project_id"] == project_id]))

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


def show_all_metrics(project_id):
    show_pipeline_metrics(project_id)
    st.divider()
    show_model_metrics(project_id)


# --------------------------------------------------
# Intent Detection
# --------------------------------------------------
def is_metric_question(q):
    keywords = [
        "metric", "metrics", "chart", "trend", "kpi",
        "performance", "accuracy", "precision", "recall",
        "f1", "auc", "success rate", "health"
    ]
    return any(word in q for word in keywords)


def is_model_metric_question(q):
    keywords = [
        "model metric", "model metrics", "model performance",
        "accuracy", "precision", "recall", "f1", "auc"
    ]
    return any(word in q for word in keywords)


def is_pipeline_metric_question(q):
    keywords = [
        "pipeline metric", "pipeline metrics", "adf metric",
        "adf metrics", "engineering metric", "engineering metrics",
        "success rate", "pipeline health", "adf health"
    ]
    return any(word in q for word in keywords)


def is_project_360_question(q):
    keywords = [
        "360", "complete", "overview", "full details",
        "project details", "everything", "all details"
    ]
    return any(word in q for word in keywords)


# --------------------------------------------------
# Smart targeted answers
# --------------------------------------------------
def smart_answer(question):
    q = question.lower()
    project = find_project(question)

    if project is None:
        available = "\n".join([f"- {p}" for p in projects["project_name"].tolist()])
        return f"""
I could not identify the project/model.

Available projects:

{available}
"""

    project_id = project["project_id"]
    project_name = project["project_name"]

    # Model path / artifact
    if (
        "model path" in q
        or "model artifact" in q
        or "artifact path" in q
        or "where is the model" in q
        or "model stored" in q
        or "model location" in q
    ):
        df = models[models["project_id"] == project_id]

        if df.empty:
            return f"No model details found for **{project_name}**."

        row = df.iloc[0]

        return f"""
### Model Path for {project_name}

| Field | Value |
|---|---|
| Model Name | {row['model_name']} |
| Model Type | {row['model_type']} |
| Latest Version | {row['latest_version']} |
| Status | {row['status']} |
| Model Artifact Path | `{row['model_artifact_path']}` |
| Owner | {row['owner']} |
"""

    # Training notebook
    if "training notebook" in q or "train notebook" in q:
        df = notebooks[
            (notebooks["project_id"] == project_id)
            & (notebooks["notebook_type"].str.lower() == "training")
        ]

        if df.empty:
            return f"No training notebook found for **{project_name}**."

        row = df.iloc[0]

        return f"""
### Training Notebook for {project_name}

| Field | Value |
|---|---|
| Notebook Name | {row['notebook_name']} |
| Workspace | {row['workspace']} |
| Notebook Path | `{row['notebook_path']}` |
| Purpose | {row['purpose']} |
| Owner | {row['owner']} |
"""

    # Scoring notebook
    if (
        "scoring notebook" in q
        or "score notebook" in q
        or "scoring code" in q
        or "score code" in q
    ):
        df = notebooks[
            (notebooks["project_id"] == project_id)
            & (notebooks["notebook_type"].str.lower() == "scoring")
        ]

        if df.empty:
            return f"No scoring notebook found for **{project_name}**."

        row = df.iloc[0]

        return f"""
### Scoring Notebook for {project_name}

| Field | Value |
|---|---|
| Notebook Name | {row['notebook_name']} |
| Workspace | {row['workspace']} |
| Notebook Path | `{row['notebook_path']}` |
| Purpose | {row['purpose']} |
| Owner | {row['owner']} |
"""

    # ADF pipeline
    if "adf" in q or "pipeline" in q or "trigger" in q or "schedule" in q:
        df = pipelines[pipelines["project_id"] == project_id]

        if df.empty:
            return f"No pipeline found for **{project_name}**."

        return f"""
### ADF Pipeline Details for {project_name}

{format_table(df)}
"""

    # Git scripts
    if "git" in q or "repo" in q or "script" in q or "code" in q:
        df = scripts[scripts["project_id"] == project_id]

        if df.empty:
            return f"No Git script details found for **{project_name}**."

        return f"""
### Git / Script Details for {project_name}

{format_table(df)}
"""

    # Data paths
    if (
        "data path" in q
        or "training data" in q
        or "scoring data" in q
        or "prediction" in q
        or "output data" in q
        or "gen1" in q
        or "gen2" in q
    ):
        df = data_assets[data_assets["project_id"] == project_id]

        if "training data" in q:
            df = df[df["purpose"].str.lower().str.contains("training", na=False)]

        elif "scoring data" in q:
            df = df[df["purpose"].str.lower().str.contains("scoring", na=False)]

        elif "prediction" in q or "output" in q:
            df = df[df["purpose"].str.lower().str.contains("prediction|output", na=False)]

        if df.empty:
            return f"No matching data path found for **{project_name}**."

        return f"""
### Data Path Details for {project_name}

{format_table(df)}
"""

    # Dashboard
    if "dashboard" in q or "power bi" in q or "report" in q:
        df = dashboards[dashboards["project_id"] == project_id]

        if df.empty:
            return f"No dashboard found for **{project_name}**."

        return f"""
### Dashboard Details for {project_name}

{format_table(df)}
"""

    # Wiki / documentation
    if "wiki" in q or "document" in q or "documentation" in q:
        df = documents[documents["project_id"] == project_id]

        if df.empty:
            return f"No documentation found for **{project_name}**."

        return f"""
### Documentation for {project_name}

{format_table(df)}
"""

    # Lineage / relationship
    if "lineage" in q or "relationship" in q or "graph" in q or "dependency" in q:
        df = relationships[relationships["project_id"] == project_id]

        if df.empty:
            return f"No relationship data found for **{project_name}**."

        return f"""
### Engineering Lineage for {project_name}

{format_table(df)}
"""

    # Fallback
    return project_360(project_id)


# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("🤖 Data Engineering Assistant")
st.caption(
    "Ask about projects, models, Synapse notebooks, ADF pipelines, Git scripts, data paths, dashboards, wiki, lineage, and metrics."
)

with st.sidebar:
    st.header("Example questions")
    st.markdown("""
- Give me the model path for Customer Conversion
- Where is the scoring notebook for Customer Conversion?
- Show training notebook for Customer Conversion
- Show ADF pipeline for Customer Conversion
- Show Git scripts for Customer Conversion
- Show data paths for Customer Conversion
- Show model metrics for Customer Conversion
- Show pipeline metrics for Customer Conversion
- Show all metrics for Customer Conversion
- Show Customer Conversion project 360
""")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! Ask me about model paths, notebooks, ADF, Git, data paths, lineage, or metrics."
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

    with st.chat_message("assistant"):
        if project is not None and is_metric_question(q):
            project_id = project["project_id"]
            project_name = project["project_name"]

            if is_model_metric_question(q):
                st.markdown(f"Showing model performance metrics for **{project_name}**")
                show_model_metrics(project_id)
                assistant_text = f"Showing model performance metrics for {project_name}"

            elif is_pipeline_metric_question(q):
                st.markdown(f"Showing pipeline engineering metrics for **{project_name}**")
                show_pipeline_metrics(project_id)
                assistant_text = f"Showing pipeline engineering metrics for {project_name}"

            else:
                st.markdown(f"Showing all engineering and model metrics for **{project_name}**")
                show_all_metrics(project_id)
                assistant_text = f"Showing all metrics for {project_name}"

        elif project is not None and is_project_360_question(q):
            assistant_text = project_360(project["project_id"])
            st.markdown(assistant_text)

        else:
            assistant_text = smart_answer(prompt)
            st.markdown(assistant_text)

    st.session_state.messages.append({
        "role": "assistant",
        "content": assistant_text
    })
