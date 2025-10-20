# Libraries
from typing import TypedDict
import os

import pandas as pd

from langchain.prompts import PromptTemplate
from langgraph.types import Checkpointer
from langgraph.graph import StateGraph, END

from .utils import (
    PythonOutputParser,
    get_dataframe_summary,
    execute_agent_code,
    fix_agent_code,
)

# Setup
AGENT_NAME = "lightweight_data_cleaning_agent"
LOG_PATH = os.path.join(os.getcwd(), "logs/")


# ============================================================================
# AGENT CLASS
# ============================================================================


class LightweightDataCleaningAgent:
    """
    A lightweight data cleaning agent that performs basic data cleaning operations.
    
    This simplified version focuses on core cleaning tasks:
    - Removing columns with excessive missing values
    - Basic imputation for missing values
    - Removing duplicate rows
    
    Can be extended to add more sophisticated cleaning steps.

    Parameters
    ----------
    model : langchain.llms.base.LLM
        The language model used to generate the data cleaning function.
    n_samples : int, optional
        Number of samples used when summarizing the dataset. Defaults to 30.
    log : bool, optional
        Whether to log the generated code and errors. Defaults to False.
    log_path : str, optional
        Directory path for storing log files. Defaults to None.
    file_name : str, optional
        Name of the file for saving the generated response. Defaults to "data_cleaner.py".
    function_name : str, optional
        Name of the generated data cleaning function. Defaults to "data_cleaner".
    checkpointer : langgraph.types.Checkpointer, optional
        Checkpointer to save and load the agent's state. Defaults to None.

    Methods
    -------
    update_params(**kwargs)
        Updates the agent's parameters and rebuilds the compiled state graph.
    invoke_agent(user_instructions: str, data_raw: pd.DataFrame, max_retries=3, retry_count=0)
        Cleans the provided dataset based on user instructions.
    get_data_cleaned()
        Retrieves the cleaned dataset as a pandas DataFrame.
    get_data_raw()
        Retrieves the raw dataset as a pandas DataFrame.
    get_data_cleaner_function()
        Retrieves the generated Python function used for cleaning the data.
    get_response()
        Returns the response from the agent as a dictionary.
    show()
        Displays the agent's mermaid diagram.

    Examples
    --------
    ```python
    import pandas as pd
    from langchain_openai import ChatOpenAI
    from lightweight_data_cleaning_agent import LightweightDataCleaningAgent

    llm = ChatOpenAI(model="gpt-4o-mini")

    agent = LightweightDataCleaningAgent(model=llm, log=True)

    df = pd.read_csv("data/churn_data.csv")

    agent.invoke_agent(
        user_instructions="Remove columns with more than 50% missing values.",
        data_raw=df
    )

    cleaned_data = agent.get_data_cleaned()
    ```
    """
    
    def __init__(
        self, 
        model, 
        n_samples=30, 
        log=False, 
        log_path=None, 
        file_name="data_cleaner.py", 
        function_name="data_cleaner",
        checkpointer: Checkpointer = None
    ):
        self.model = model
        self.n_samples = n_samples
        self.log = log
        self.log_path = log_path
        self.file_name = file_name
        self.function_name = function_name
        self.checkpointer = checkpointer
        self.response = None
        self._compiled_graph = make_lightweight_data_cleaning_agent(
            model=model,
            n_samples=n_samples,
            log=log,
            log_path=log_path,
            file_name=file_name,
            function_name=function_name,
            checkpointer=checkpointer
        )
    
    def invoke_agent(self, data_raw: pd.DataFrame, user_instructions: str=None, max_retries:int=3, retry_count:int=0, **kwargs):
        """
        Invokes the agent. The response is stored in the response attribute.

        Parameters:
        ----------
            data_raw (pd.DataFrame): 
                The raw dataset to be cleaned.
            user_instructions (str): 
                Instructions for data cleaning agent.
            max_retries (int): 
                Maximum retry attempts for cleaning.
            retry_count (int): 
                Current retry attempt.
            **kwargs
                Additional keyword arguments to pass to invoke().

        Returns:
        --------
            None. The response is stored in the response attribute.
        """
        response = self._compiled_graph.invoke({
            "user_instructions": user_instructions,
            "data_raw": data_raw.to_dict(),
            "max_retries": max_retries,
            "retry_count": retry_count,
        }, **kwargs)
        self.response = response
        return None
    
    def get_data_cleaned(self):
        """
        Retrieves the cleaned data stored after running invoke_agent.
        """
        if self.response:
            return pd.DataFrame(self.response.get("data_cleaned"))
        
    def get_data_raw(self):
        """
        Retrieves the raw data.
        """
        if self.response:
            return pd.DataFrame(self.response.get("data_raw"))
    
    def get_data_cleaner_function(self):
        """
        Retrieves the agent's cleaning function code.
        """
        if self.response:
            return self.response.get("data_cleaner_function")


# Agent Factory Function

def make_lightweight_data_cleaning_agent(
    model, 
    n_samples=30, 
    log=False, 
    log_path=None, 
    file_name="data_cleaner.py",
    function_name="data_cleaner",
    checkpointer: Checkpointer = None
):
    """
    Creates a lightweight data cleaning agent.
    
    This agent performs basic cleaning steps:
    - Removing columns with more than 40% missing values
    - Imputing missing values (mean for numeric, mode for categorical)
    - Removing duplicate rows
    
    Can be extended to add:
    - Outlier detection and removal
    - Data type conversions
    - Custom validation rules
    - More sophisticated imputation strategies

    Parameters
    ----------
    model : langchain.llms.base.LLM
        The language model to use to generate code.
    n_samples : int, optional
        The number of samples to use when summarizing the dataset. Defaults to 30.
    log : bool, optional
        Whether or not to log the code generated. Defaults to False.
    log_path : str, optional
        The path to the directory where the log files should be stored.
    file_name : str, optional
        The name of the file to save the response to. Defaults to "data_cleaner.py".
    function_name : str, optional
        The name of the function that will be generated. Defaults to "data_cleaner".
    checkpointer : langgraph.types.Checkpointer, optional
        Checkpointer to save and load the agent's state. Defaults to None.

    Returns
    -------
    app : langchain.graphs.CompiledStateGraph
        The data cleaning agent as a state graph.
    """
    llm = model
    
    # Setup Log Directory
    if log:
        if log_path is None:
            log_path = LOG_PATH
        if not os.path.exists(log_path):
            os.makedirs(log_path)    

    # Define GraphState
    class GraphState(TypedDict):
        user_instructions: str
        data_raw: dict
        data_cleaned: dict
        data_cleaner_function: str
        data_cleaner_function_path: str
        data_cleaner_function_name: str
        data_cleaner_error: str
        max_retries: int
        retry_count: int

    
    def create_data_cleaner_code(state: GraphState):
        """
        Generate the data cleaning code based on user instructions.
        """
        print(f"--- {AGENT_NAME.upper().replace('_', ' ')} ---")
        print("    * CREATE DATA CLEANER CODE")
        
        data_raw = state.get("data_raw")
        df = pd.DataFrame.from_dict(data_raw)

        dataset_summary = get_dataframe_summary(df, n_sample=n_samples)
        
        # TODO: Expand this prompt with more detailed cleaning instructions
        data_cleaning_prompt = PromptTemplate(
            template="""
            You are a Data Cleaning Agent. Create a {function_name}() function to clean the data.

            Basic Cleaning Steps to implement:
            1. Remove columns with more than 40% missing values
            2. Impute missing values (mean for numeric, mode for categorical)
            3. Remove duplicate rows

            User Instructions:
            {user_instructions}

            Dataset Summary:
            {all_datasets_summary}

            Return Python code in ```python``` format with a single function:

            def {function_name}(data_raw):
                import pandas as pd
                import numpy as np
                # Your cleaning code here
                return data_cleaned

            Important: Ensure fit_transform() outputs are flattened with .ravel() when assigning to DataFrame columns.
            """,
            input_variables=["user_instructions", "all_datasets_summary", "function_name"]
        )

        data_cleaning_agent = data_cleaning_prompt | llm | PythonOutputParser()
        
        response = data_cleaning_agent.invoke({
            "user_instructions": state.get("user_instructions") or "Follow the basic cleaning steps.",
            "all_datasets_summary": dataset_summary,
            "function_name": function_name
        })
        
        # Simple logging if enabled
        file_path = None
        if log:
            file_path = os.path.join(log_path, file_name)
            with open(file_path, 'w') as f:
                f.write(response)
            print(f"      Code saved to: {file_path}")
   
        return {
            "data_cleaner_function": response,
            "data_cleaner_function_path": file_path,
            "data_cleaner_function_name": function_name,
        }
        
    def execute_data_cleaner_code(state):
        """
        Execute the generated cleaning code on the data.
        """
        return execute_agent_code(
            state=state,
            data_key="data_raw",
            result_key="data_cleaned",
            error_key="data_cleaner_error",
            code_snippet_key="data_cleaner_function",
            agent_function_name=state.get("data_cleaner_function_name")
        )
        
    def fix_data_cleaner_code(state: GraphState):
        """
        Fix errors in the generated data cleaning code.
        """
        data_cleaner_prompt = """
        You are a Data Cleaning Agent. Fix the broken {function_name}() function.
        
        Return Python code in ```python``` format with the corrected function definition.
        
        Broken code: 
        {code_snippet}

        Error:
        {error}
        """

        return fix_agent_code(
            state=state,
            code_snippet_key="data_cleaner_function",
            error_key="data_cleaner_error",
            llm=llm,  
            prompt_template=data_cleaner_prompt,
            function_name=state.get("data_cleaner_function_name"),
        )

    # Build the workflow graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("create_data_cleaner_code", create_data_cleaner_code)
    workflow.add_node("execute_data_cleaner_code", execute_data_cleaner_code)
    workflow.add_node("fix_data_cleaner_code", fix_data_cleaner_code)
    
    # Set entry point
    workflow.set_entry_point("create_data_cleaner_code")
    
    # Add edges
    workflow.add_edge("create_data_cleaner_code", "execute_data_cleaner_code")
    workflow.add_edge("fix_data_cleaner_code", "execute_data_cleaner_code")
    
    # Add conditional edge for error handling
    def should_retry(state):
        has_error = state.get("data_cleaner_error") is not None
        can_retry = (
            state.get("retry_count") is not None
            and state.get("max_retries") is not None
            and state["retry_count"] < state["max_retries"]
        )
        return "fix_code" if (has_error and can_retry) else "end"
    
    workflow.add_conditional_edges(
        "execute_data_cleaner_code",
        should_retry,
        {
            "fix_code": "fix_data_cleaner_code",
            "end": END,
        },
    )
    
    # Compile the workflow
    app = workflow.compile(checkpointer=checkpointer, name=AGENT_NAME)
    
    return app
