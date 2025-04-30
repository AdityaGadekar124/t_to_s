import streamlit as st
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine, text
import pandas as pd
import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# Streamlit UI
st.set_page_config(page_title="Ask your PostgreSQL DB", layout="centered")
st.title("🧠 NL → SQL using LangChain + OpenRouter")

st.markdown("Type a natural language query and get results from your PostgreSQL database.")

# DB config
db_url = os.getenv("DATABASE_URL")
# Few-shot examples
examples = [
    {
        "question": "List all documents with the status 'review'.",
        "sql": "SELECT * FROM documents WHERE status = 'review';"
    },
    {
        "question": "Show the names and creation dates of documents created by 'emma'.",
        "sql": "SELECT document_name, created_at FROM documents WHERE created_by = 'emma';"
    },
    {
        "question": "Count how many documents have the status 'draft'.",
        "sql": "SELECT COUNT(*) FROM documents WHERE status = 'draft';"
    },
    {
        "question": "Get the names of documents modified after '2025-04-25'.",
        "sql": "SELECT document_name FROM documents WHERE modified_at > '2025-04-25';"
    },
    {
        "question": "List document names, version numbers, and status for documents with version '4.7'.",
        "sql": "SELECT document_name, version_number, status FROM documents WHERE version_number = '4.7';"
    },
    {
        "question": "Show all documents with task ID greater than 500.",
        "sql": "SELECT * FROM documents WHERE task_id > 500;"
    },
    {
        "question": "List all distinct document statuses.",
        "sql": "SELECT DISTINCT status FROM documents;"
    }
]

example_prompt = PromptTemplate(
    input_variables=["question", "sql"],
    template="Q: {question}\nSQL: {sql}"
)
# Custom instructions to guide the LLM behavior
custom_instructions = """
You are an expert SQL generator. Your job is to convert natural language questions into valid SQL queries.
Always use lowercase SQL keywords.
very important point :make sure that you do not include any explanations or any other text except the sql query. Just return the SQL query nothing extra.
Assume the database follows PostgreSQL syntax and the schema resembles a movie rental store.
I know that you are the best and you can do this. and i am sure that you will do this correctly.And you are also great at handking the complex queries.
you can generate accurate SQL queries for the user input.
"""

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=custom_instructions + "\n\nHere are some examples:",
    suffix="Q: {user_question}\nSQL:",
    input_variables=["user_question"]
)


# Connect to OpenRouter LLM
llm = ChatOpenAI(
    temperature=0,
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    model="deepseek/deepseek-chat-v3-0324:free"  # Change this if needed
)

# Text input
user_input = st.text_input("📝 Your question:", "")

# Run button
if st.button("Run Query") and user_input.strip():
    with st.spinner("Generating SQL & executing..."):
        try:
            # Generate the prompt with user input
            prompt = few_shot_prompt.format(user_question=user_input)

            # Get the SQL query from the LLM model
            response = llm.invoke(prompt)  # Directly invoking the model with the formatted prompt

            sql_query = response.content.strip() # Apply strip to remove any leading/trailing whitespace
            
            st.code(sql_query, language="sql")

            # Connect and run SQL
            engine = create_engine(db_url)
            with engine.connect() as conn:
                result = conn.execute(text(sql_query))
                df = pd.DataFrame(result.fetchall(), columns=result.keys())

            st.success("✅ Query executed successfully!")
            st.dataframe(df)

        except Exception as e:
            st.error(f"❌ Error: {e}")
