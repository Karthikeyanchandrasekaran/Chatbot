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
# Basic helpers
# -----------------------------
def format_table(df):
    if df.empty:
        return "No records found."
    return df.to_markdown(index=False)

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

# -----------------------------
# Dynamic search index
# -----------------------------
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

def dynamic_answer(question):
    results = dynamic_search(question, top_n=6)

    if results.empty:
        available = "\n".join([f"- {p}" for p in projects["project_name"].tolist()])
        return f"""
I could not find a strong match.

Available projects:

{available}

Try asking:
- Where is conversion scoring?
- Show ADF pipeline for Customer Conversion
- Which Git scripts are used for churn?
- Show model metrics for conversion model
"""

    response = "### Best matching engineering assets\n\n"

    for _, result in results.iterrows():
        record = result["record"]
        table_name = result["table_name"]
        score = round(result["score"], 2)

        response += f"#### {table_name}  \n"
        response += f"Match score: `{score}`\n\n"

        for key, value in record.items():
            response += f"- **{key}**: {value}\n"

        response += "\n---\n\n"

    return response

# -----------------------------
# Project 360
# -----------------------------
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

# -----------------------------
# Pipeline metrics
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
# Model metrics
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

def show_all_metrics(project_id):
    show_pipeline_metrics(project_id)
    st.divider()
    show_model_metrics(project_id)

# -----------------------------
# Intent detection
# -----------------------------
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
    keywords = ["360", "complete", "overview", "full details", "project details", "everything"]
    return any(word in q for word in keywords)

# -----------------------------
# UI
# -----------------------------
st.title("🤖 Data Engineering Assistant")
st.caption(
    "Dynamic free POV chatbot for projects, models, Synapse notebooks, ADF pipelines, Git scripts, data paths, dashboards, wiki, lineage, and metrics."
)

with st.sidebar:
    st.header("Example questions")
    st.markdown("""
- Show Customer Conversion project 360
- Where is conversion scoring happening?
- Which Git code is used for churn?
- Show data paths for Customer Conversion
- Show lineage for Customer Conversion
- Show model metrics for Customer Conversion
- Show pipeline health for Customer Conversion
- Show all metrics for Customer Conversion
- Show AUC trend for conversion model
""")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! Ask me anything about your engineering assets or metrics."
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
            assistant_text = dynamic_answer(prompt)
            st.markdown(assistant_text)

    st.session_state.messages.append({
        "role": "assistant",
        "content": assistant_text
    })
