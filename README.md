# Data Cleaning Agent

A lightweight AI-powered data cleaning agent using LangChain and LangGraph.

## Features

- AI-powered data cleaning using LLMs
- Automatic removal of columns with excessive missing values
- Smart imputation (mean for numeric, mode for categorical)
- Duplicate row detection and removal
- Export cleaned data as CSV

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

```bash
poetry run streamlit run app.py
```

Then upload your CSV file and click "Clean Data"!

### Python API

```python
import pandas as pd
from langchain_openai import ChatOpenAI
from data_cleaning_agent import LightweightDataCleaningAgent

# Initialize
llm = ChatOpenAI(model="gpt-4o-mini")
agent = LightweightDataCleaningAgent(model=llm)

# Load your data
df = pd.read_csv("your_data.csv")

# Clean it
agent.invoke_agent(data_raw=df)
cleaned_df = agent.get_data_cleaned()
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
