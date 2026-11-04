# Data Cleaning Agent (Finished Version)

> This is the finished version of the Week 3 data cleaning agent, here as a lifeline if you get stuck. It shows one good way to reach the goal, not the only one, so try it yourself first. The student-facing starter is [data-cleaning-agent](https://github.com/Future-Proof-DS/data-cleaning-agent).

An AI-powered data cleaning agent that automatically cleans messy datasets using LangChain and LangGraph. The agent uses an LLM to generate and execute Python code for common data cleaning tasks like standardizing data types, removing outliers, handling missing values, removing duplicates, and dropping low-quality columns.

## How It Works

The agent follows a simple workflow:
1. **Analyze**: Examines your dataset structure and identifies data quality issues
2. **Generate**: Uses an LLM to create custom Python cleaning code based on the data
3. **Execute**: Runs the generated code to clean your data
4. **Retry**: Automatically fixes errors if the generated code fails (up to 3 attempts)

This approach combines the flexibility of LLMs with the reliability of pandas operations.

## Features

**New cleaning steps** on top of the baseline (missing values, duplicates, column removal):
- **Data type standardization**: converts numbers stored as text and dates to their proper types
- **Outlier removal**: removes rows with outliers in numeric columns, using the IQR method

Every step lives in `CLEANING_STEPS` in `data_cleaning_agent.py`. That list is the only source of instructions for the agent, including when it retries.

**Improved Streamlit UI**:
- Data quality metrics before and after cleaning: missing cells, duplicate rows and outliers
- Toggles for each cleaning step, plus custom instructions
- What changed: the rows and columns removed, and each cell before and after
- The generated code, and clear errors when something fails

## Setup

### Prerequisites

- **Python 3.9 or higher** (3.9, 3.10, 3.11, 3.12, or 3.13) - **Note**: Python 3.9.7 is not supported due to a Streamlit compatibility issue
- **Poetry** (dependency manager)
- **OpenAI API Key**

### Installation Steps

1. **Install Poetry** (if not already installed):
   
   **Windows (PowerShell)**:
   ```powershell
   (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
   ```
   
   **macOS/Linux**:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
   
   After installation, restart your terminal. If `poetry` command is not found:
   - **Windows**: Add `%APPDATA%\Python\Scripts` to your system PATH
   - **macOS/Linux**: Add `export PATH="$HOME/.local/bin:$PATH"` to your `~/.bashrc` or `~/.zshrc`

2. **Install dependencies**:
   ```bash
   poetry install
   ```
   
   This will install all dependencies with the exact versions specified in `poetry.lock`, ensuring consistency across all environments.

3. **Set up your OpenAI API key**:
   
   **Windows**:
   ```powershell
   copy .env.example .env
   ```
   
   **macOS/Linux**:
   ```bash
   cp .env.example .env
   ```
   
   Then edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-your-key-here
   ```

### Multiple Python Versions?

If you have multiple Python versions installed and want to use a specific one:

```bash
# Tell Poetry which Python to use
poetry env use python3.11  # or python3.9, python3.10, python3.12, etc.

# Then install dependencies
poetry install
```

Poetry will create a virtual environment with your chosen Python version.

## Usage

### Streamlit Web Interface

The easiest way to use the agent is through the web interface:

```bash
poetry run streamlit run app.py
```

Then:
1. Choose your cleaning steps in the sidebar, and add custom instructions if you need them
2. Upload your CSV file and review its data quality
3. Click "Clean Data"
4. Review what changed
5. Download the cleaned dataset

### Python API

For programmatic use or integration into data pipelines:

```python
import pandas as pd
from langchain_openai import ChatOpenAI
from data_cleaning_agent import LightweightDataCleaningAgent

# Initialize the agent with an LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=2000, max_retries=0, timeout=90)
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

By default the agent applies every step in `CLEANING_STEPS`. When you pass `user_instructions`, it follows those instead:

```python
# Give specific cleaning instructions to the agent
agent.invoke_agent(
    data_raw=df,
    user_instructions="Remove columns with more than 30% missing values and standardize date formats"
)

# Or pick steps from CLEANING_STEPS and add your own
from data_cleaning_agent import build_cleaning_instructions

agent.invoke_agent(
    data_raw=df,
    user_instructions=build_cleaning_instructions(
        steps=["remove_outliers", "remove_duplicates"],
        custom_instructions="Lowercase the city column"
    )
)
```

## Deployment

### Docker

Build the image from the project root:

```bash
docker build -t data-cleaning-agent .
```

Run it, passing your API key from `.env`:

```bash
docker run --rm -p 8501:8501 --env-file .env data-cleaning-agent
```

Then open [http://localhost:8501](http://localhost:8501).

**Important**: `.env` is listed in `.dockerignore`, so your key is never built into the image.

### Render (Optional)

1. Push your repository to GitHub
2. On [Render](https://render.com/docs/docker), create a new **Web Service** from the repository and choose **Docker** as the language
3. Add `OPENAI_API_KEY` as an environment variable
4. Deploy

Render sets the `PORT` variable for you, and the Dockerfile picks it up.

## Project Structure

```
data-cleaning-agent/
├── data_cleaning_agent/
│   ├── __init__.py
│   ├── data_cleaning_agent.py  # Main agent class
│   └── utils.py                # Utility functions
├── app.py                      # Streamlit interface
├── Dockerfile                  # Container image
├── pyproject.toml              # Dependencies configuration
├── poetry.lock                 # Locked dependency versions
└── README.md
```

**Important**: The `poetry.lock` file is committed to ensure all users get identical, tested dependency versions.
