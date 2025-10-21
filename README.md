# Data Cleaning Agent

An AI-powered data cleaning agent that automatically cleans messy datasets using LangChain and LangGraph. The agent uses an LLM to generate and execute Python code for common data cleaning tasks like handling missing values, removing duplicates, and dropping low-quality columns.

## How It Works

The agent follows a simple workflow:
1. **Analyze**: Examines your dataset structure and identifies data quality issues
2. **Generate**: Uses an LLM to create custom Python cleaning code based on the data
3. **Execute**: Runs the generated code to clean your data
4. **Retry**: Automatically fixes errors if the generated code fails (up to 3 attempts)

This approach combines the flexibility of LLMs with the reliability of pandas operations.

## Installation

```bash
# Install dependencies with Poetry
poetry install

# Set up environment variables
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

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

## Requirements

- Python ^3.9
- OpenAI API key

## License

MIT
