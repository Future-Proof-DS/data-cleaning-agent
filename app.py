"""Streamlit interface for the Data Cleaning Agent."""

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from data_cleaning_agent import (
    CLEANING_STEPS,
    LightweightDataCleaningAgent,
    build_cleaning_instructions,
)
from data_cleaning_agent.utils import get_cleaning_changes, get_data_quality_report

load_dotenv()


def show_quality_metrics(report, before=None):
    """Show the headline numbers, with the change from `before` when given."""
    metrics = {
        "Rows": "rows",
        "Missing Cells": "missing_cells",
        "Duplicate Rows": "duplicate_rows",
        "Outliers": "outliers",
    }
    for column, (label, key) in zip(st.columns(len(metrics)), metrics.items()):
        delta = report[key] - before[key] if before else None
        column.metric(label, report[key], delta=delta, delta_color="off")


st.title("🧹 Data Cleaning Agent")

# Choose what the agent does. These toggles are the agent's only instructions
with st.sidebar:
    st.header("Cleaning Steps")
    selected_steps = [
        step for step, description in CLEANING_STEPS.items()
        if st.checkbox(step.replace("_", " ").capitalize(), value=True, help=description)
    ]
    custom_instructions = st.text_area(
        "Custom instructions (optional)",
        placeholder="e.g. Keep rows with a missing employee_id",
    )

user_instructions = build_cleaning_instructions(selected_steps, custom_instructions)

# Upload file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    # Load data
    try:
        df_raw = pd.read_csv(uploaded_file)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as e:
        st.error(f"This file could not be read as a CSV: {e}")
        st.stop()

    # A new file starts clean, without the results of the last one
    if st.session_state.get("file_id") != uploaded_file.file_id:
        st.session_state["file_id"] = uploaded_file.file_id
        st.session_state.pop("result", None)

    # Data quality before cleaning
    report_raw = get_data_quality_report(df_raw)
    column_details = report_raw["column_details"]

    st.subheader("Data Quality")
    st.write(f"Shape: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
    show_quality_metrics(report_raw)

    if report_raw["missing_cells"] > 0:
        st.caption("Missing values by column (%)")
        st.bar_chart(column_details.loc[column_details["Missing"] > 0, "Missing %"])

    with st.expander("Column Details"):
        st.dataframe(column_details)

    with st.expander("Original Data"):
        st.dataframe(df_raw.head())

    # Clean button
    if not user_instructions:
        st.warning("Select a cleaning step or add custom instructions to continue.")

    if st.button("Clean Data", disabled=not user_instructions):
        st.session_state.pop("result", None)
        with st.spinner("Cleaning..."):
            try:
                # Hard limits, so one run can never hang or spend more than expected
                llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0,
                    max_tokens=2000,
                    max_retries=0,
                    timeout=90,
                )
                agent = LightweightDataCleaningAgent(model=llm, log=True)
                agent.invoke_agent(data_raw=df_raw, user_instructions=user_instructions)
            except Exception as e:
                st.error(f"The agent could not finish: {e}")
                st.stop()

        # Saved so the results survive the rerun Streamlit does on every click
        st.session_state["result"] = {
            "instructions": user_instructions,
            "code": agent.get_data_cleaner_function(),
            "error": agent.get_data_cleaner_error(),
            "df_cleaned": agent.get_data_cleaned(),
        }

    result = st.session_state.get("result")

    if result and result["error"]:
        st.error(f"The agent could not fix its code after 3 retries. {result['error']}")
        with st.expander("Last Generated Code"):
            st.code(result["code"], language="python")

    elif result:
        df_cleaned = result["df_cleaned"]

        st.success("Done!")

        st.subheader("Cleaned Data")
        st.write(f"Shape: {df_cleaned.shape[0]} rows × {df_cleaned.shape[1]} columns")
        show_quality_metrics(get_data_quality_report(df_cleaned), before=report_raw)
        st.dataframe(df_cleaned.head())

        # What the cleaning actually did to this data
        changes = get_cleaning_changes(df_raw, df_cleaned)
        rows_removed = changes["rows_removed"]
        columns_removed = changes["columns_removed"]
        cells_changed = changes["cells_changed"]

        st.subheader("What Changed")
        st.write(
            f"Rows removed: {len(rows_removed)} | Columns removed: {len(columns_removed)} | "
            f"Cells changed: {len(cells_changed)}"
        )

        if columns_removed:
            st.write(f"Columns removed: {', '.join(columns_removed)}")

        if not cells_changed.empty:
            st.caption("Cells changed by column")
            st.bar_chart(cells_changed["Column"].value_counts())
            with st.expander("Cells Changed"):
                st.dataframe(cells_changed, hide_index=True)

        if not rows_removed.empty:
            with st.expander("Rows Removed"):
                st.dataframe(rows_removed)

        with st.expander("Generated Code"):
            st.text(result["instructions"])
            st.code(result["code"], language="python")

        # Download
        csv = df_cleaned.to_csv(index=False)
        st.download_button(
            "Download Cleaned Data",
            data=csv,
            file_name="cleaned_data.csv",
            mime="text/csv"
        )
