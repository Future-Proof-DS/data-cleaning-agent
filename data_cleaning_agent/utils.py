# Utility functions for lightweight data cleaning agent

import re
import logging
import pandas as pd
from langchain_core.output_parsers import BaseOutputParser

logger = logging.getLogger(__name__)


class PythonOutputParser(BaseOutputParser):
    """Extract Python code from LLM responses."""

    def parse(self, text: str):
        """Extract code from ```python``` blocks or return text as-is."""
        python_code_match = re.search(r'```python(.*?)```', text, re.DOTALL)
        if python_code_match:
            return python_code_match.group(1).strip()
        return text


def get_dataframe_summary(df: pd.DataFrame, n_sample: int = 30) -> str:
    """
    Generate a simple summary of a DataFrame for the LLM.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to summarize.
    n_sample : int, optional
        Number of rows to include in the sample. Defaults to 30.

    Returns
    -------
    str
        A text summary of the DataFrame.
    """
    missing_stats = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
    missing_summary = "\n".join([f"{col}: {val:.2f}%" for col, val in missing_stats.items()])

    column_types = "\n".join([f"{col}: {dtype}" for col, dtype in df.dtypes.items()])

    summary = f"""
        Dataset Summary:
        ----------------
        Shape: {df.shape[0]} rows x {df.shape[1]} columns

        Column Data Types:
        {column_types}

        Missing Value Percentage:
        {missing_summary}

        Sample Data (first {n_sample} rows):
        {df.head(n_sample).to_string()}"""

    return summary.strip()


def get_data_quality_report(df: pd.DataFrame) -> dict:
    """
    Measure the quality of a DataFrame, for showing to the user.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to measure.

    Returns
    -------
    dict
        Overall counts, plus a "column_details" DataFrame with one row per column.
    """
    # Same IQR rule the agent is told to use
    numeric = df.select_dtypes(include="number")
    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1
    outliers = ((numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)).sum()

    columns = pd.DataFrame({
        "Type": df.dtypes.astype(str),
        "Missing": df.isna().sum(),
        "Missing %": (df.isna().mean() * 100).round(1),
        "Unique Values": df.nunique(),
        "Outliers": outliers.reindex(df.columns).fillna(0).astype(int),
    })

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "outliers": int(outliers.sum()),
        "column_details": columns,
    }


def get_cleaning_changes(df_raw: pd.DataFrame, df_cleaned: pd.DataFrame) -> dict:
    """
    Compare the raw and cleaned data to show what the cleaning changed.

    Rows are matched on their index, which the agent is told to keep.

    Parameters
    ----------
    df_raw : pd.DataFrame
        The data before cleaning.
    df_cleaned : pd.DataFrame
        The data after cleaning.

    Returns
    -------
    dict
        The rows and columns removed, and one row per changed cell with its
        value before and after.
    """
    rows = df_raw.index.intersection(df_cleaned.index)
    columns = df_raw.columns.intersection(df_cleaned.columns)
    before = df_raw.loc[rows, columns]
    after = df_cleaned.loc[rows, columns]

    # Compared as text and as numbers, so a value that only changed type, like
    # "28" to 28.0, or a date parsed from text, does not count as a change
    before_text = before.astype(str).where(before.notna(), "missing")
    after_text = after.astype(str).where(after.notna(), "missing")
    before_numbers = before.apply(pd.to_numeric, errors="coerce")
    after_numbers = after.apply(pd.to_numeric, errors="coerce")
    changed = before_text.ne(after_text) & before_numbers.ne(after_numbers)

    cells_changed = pd.DataFrame(
        [
            {
                "Row": row,
                "Column": column,
                "Before": before_text.at[row, column],
                "After": after_text.at[row, column],
            }
            for column in columns
            for row in changed.index[changed[column]]
        ],
        columns=["Row", "Column", "Before", "After"],
    )

    return {
        "rows_removed": df_raw.loc[~df_raw.index.isin(df_cleaned.index)],
        "columns_removed": df_raw.columns.difference(df_cleaned.columns, sort=False).tolist(),
        "cells_changed": cells_changed,
    }


def execute_agent_code(state, data_key, code_snippet_key, result_key, error_key, agent_function_name):
    """
    Execute the generated agent code on the data.

    Parameters
    ----------
    state : dict
        The current state containing data and code.
    data_key : str
        Key in state where the input data is stored.
    code_snippet_key : str
        Key in state where the generated code is stored.
    result_key : str
        Key to store the result in.
    error_key : str
        Key to store any error message in.
    agent_function_name : str
        Name of the function to execute from the generated code.

    Returns
    -------
    dict
        Dictionary with result and error keys.
    """
    logger.info("Executing agent code")

    data = state.get(data_key)
    agent_code = state.get(code_snippet_key)
    df = pd.DataFrame.from_dict(data)

    # Execute the LLM-generated code in isolated namespace
    # Note: exec() can be risky - only use with trusted LLM-generated code
    local_vars = {}
    global_vars = {}
    exec(agent_code, global_vars, local_vars)

    # Get the function from executed code
    agent_function = local_vars.get(agent_function_name)
    if not agent_function or not callable(agent_function):
        raise ValueError(f"Function '{agent_function_name}' not found in generated code.")

    # Run the function and handle errors
    agent_error = None
    result = None
    try:
        result = agent_function(df)
        if isinstance(result, pd.DataFrame):
            result = result.to_dict()
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        agent_error = f"An error occurred during data cleaning: {str(e)}"

    return {result_key: result, error_key: agent_error}


def fix_agent_code(
    state,
    code_snippet_key,
    error_key,
    llm,
    prompt_template,
    function_name,
    user_instructions,
    retry_count_key="retry_count"
):
    """
    Fix errors in the generated agent code using the LLM.

    Parameters
    ----------
    state : dict
        The current state containing code and error information.
    code_snippet_key : str
        Key in state where the broken code is stored.
    error_key : str
        Key in state where the error message is stored.
    llm : LLM
        The language model to use for fixing the code.
    prompt_template : str
        Template for the fix prompt (should have {code_snippet}, {error}, {function_name},
        {user_instructions} placeholders).
    function_name : str
        Name of the function being fixed.
    user_instructions : str
        The instructions the code was generated from, so a fix cannot drift from them.
    retry_count_key : str, optional
        Key in state for tracking retry count. Defaults to "retry_count".

    Returns
    -------
    dict
        Dictionary with updated code, cleared error, and incremented retry count.
    """
    logger.info("Fixing agent code")
    logger.debug(f"Retry count: {state.get(retry_count_key)}")

    code_snippet = state.get(code_snippet_key)
    error_message = state.get(error_key)

    # Create the fix prompt
    prompt = prompt_template.format(
        code_snippet=code_snippet,
        error=error_message,
        function_name=function_name,
        user_instructions=user_instructions,
    )

    # Get fixed code from LLM
    response = (llm | PythonOutputParser()).invoke(prompt)

    return {
        code_snippet_key: response,
        error_key: None,
        retry_count_key: state.get(retry_count_key) + 1
    }
