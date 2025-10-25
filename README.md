# Data Cleaning Agent

An AI-powered data cleaning agent that automatically cleans messy datasets using LangChain and LangGraph. The agent uses an LLM to generate and execute Python code for common data cleaning tasks like handling missing values, removing duplicates, and dropping low-quality columns.

## How It Works

The agent follows a simple workflow:
1. **Analyze**: Examines your dataset structure and identifies data quality issues
2. **Generate**: Uses an LLM to create custom Python cleaning code based on the data
3. **Execute**: Runs the generated code to clean your data
4. **Retry**: Automatically fixes errors if the generated code fails (up to 3 attempts)

This approach combines the flexibility of LLMs with the reliability of pandas operations.

## Setup

### Windows (PowerShell)

1. **Verify Python is installed** (Python 3.9 or higher required):
   ```powershell
   python --version
   ```
   If not installed, download from [python.org](https://www.python.org/downloads/)

2. **Install Poetry**:
   ```powershell
   (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
   ```
   After installation, restart your terminal or IDE. If `poetry` command is not found, add `%APPDATA%\Python\Scripts` to your system PATH.

3. **Install dependencies**:
   ```powershell
   poetry install
   ```

4. **Set up your OpenAI API key**:
   ```powershell
   copy .env.example .env
   ```
   Then edit `.env` and add your OpenAI API key: `OPENAI_API_KEY=sk-your-key-here`

### macOS/Linux

1. **Install Poetry** (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Install dependencies**:
   ```bash
   poetry install
   ```

3. **Set up your OpenAI API key**:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and add your OpenAI API key: `OPENAI_API_KEY=sk-your-key-here`

## Usage

### Streamlit Web Interface

The easiest way to use the agent is through the web interface:

```bash
poetry run streamlit run app.py
```

Then:
1. Upload your CSV file
2. Click "Clean Data"
3. Download the cleaned dataset

### Python API

For programmatic use or integration into data pipelines:

```python
import pandas as pd
from langchain_openai import ChatOpenAI
from data_cleaning_agent import LightweightDataCleaningAgent

# Initialize the agent with an LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = LightweightDataCleaningAgent(model=llm)

# Load your messy data
df = pd.read_csv("your_data.csv")

# Run the cleaning agent
agent.invoke_agent(data_raw=df)

# Get the cleaned dataset
cleaned_df = agent.get_data_cleaned()

# Save or use the cleaned data
cleaned_df.to_csv("cleaned_data.csv", index=False)
```

**Optional: Provide custom instructions**

```python
# Give specific cleaning instructions to the agent
agent.invoke_agent(
    data_raw=df,
    user_instructions="Remove columns with more than 30% missing values and standardize date formats"
)
```

## Project Structure

```
data-cleaning-agent/
├── data_cleaning_agent/
│   ├── __init__.py
│   ├── data_cleaning_agent.py  # Main agent class
│   └── utils.py                # Utility functions
├── app.py                      # Streamlit interface
├── pyproject.toml              # Dependencies
└── README.md
```
