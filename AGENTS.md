# Data Cleaning Agent

A LangGraph agent that writes and runs pandas code to clean a CSV, with a Streamlit app on top. Read `README.md` for setup.

## Commands

- `poetry install` to install
- `poetry run streamlit run app.py` to run the app
- `docker build -t data-cleaning-agent .` to build the image

## Where things live

- `data_cleaning_agent/data_cleaning_agent.py`: the agent, its graph and `CLEANING_STEPS`
- `data_cleaning_agent/utils.py`: output parsing, code execution and the data quality helpers
- `app.py`: the Streamlit interface

## Rules

- `CLEANING_STEPS` is the only list of cleaning steps. To add a step, add it there. Never list steps in a prompt: the toggles in the app would stop working.
- `fix_data_cleaner_code` gets the same instructions as `create_data_cleaner_code`, so a retry cannot drift from what the user asked for.
- Generated code must keep the original row index. "What Changed" in the app depends on it.
- Every `ChatOpenAI` client sets `max_tokens`, `max_retries=0` and a `timeout`.
- Never commit `.env` or build it into the image. Pass the key at run time.
