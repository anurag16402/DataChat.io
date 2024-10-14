import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI

# Load environment variables
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize PandasAI with OpenAI
llm = OpenAI(api_token=API_KEY)
pandas_ai = PandasAI(llm)

# Streamlit app title
st.title("DataChat.io")
#st.header("Powered by OpenAI & PandasAI")

# File uploader for CSV
uploaded_file = st.file_uploader("Upload a CSV file for analysis", type=['csv'])

if uploaded_file is not None:
    # Read CSV file into dataframe
    df = pd.read_csv(uploaded_file)
    st.write("First few rows of your data:")
    st.write(df.head())

    # Data Preprocessing options
    st.subheader("Data Preprocessing Options")

    # Handling missing values
    if st.checkbox("Drop missing values"):
        df = df.dropna()
        st.write("Missing values have been dropped.")

    # Column selection
    selected_columns = st.multiselect("Select columns for analysis", df.columns.tolist(), default=df.columns.tolist())
    if selected_columns:
        df = df[selected_columns]
        st.write("Data with selected columns:")
        st.write(df.head())

    # Data visualization options
    st.subheader("Data Visualization Options")
    chart_type = st.selectbox("Choose chart type", ["None", "Bar Plot", "Line Plot", "Heatmap"])

    if chart_type != "None":
        if chart_type == "Bar Plot":
            column = st.selectbox("Select column for bar plot", df.columns)
            if column:
                plt.figure(figsize=(10, 5))
                df[column].value_counts().plot(kind='bar')
                st.pyplot(plt)

        elif chart_type == "Line Plot":
            x_col = st.selectbox("Select X-axis column", df.columns)
            y_col = st.selectbox("Select Y-axis column", df.columns)
            if x_col and y_col:
                plt.figure(figsize=(10, 5))
                sns.lineplot(data=df, x=x_col, y=y_col)
                st.pyplot(plt)

        elif chart_type == "Heatmap":
            if df.select_dtypes(include='number').shape[1] >= 2:
                plt.figure(figsize=(10, 5))
                sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
                st.pyplot(plt)
            else:
                st.warning("Heatmap requires at least two numeric columns.")

    # Prompt area
    st.subheader("Prompt-Driven Analysis")
    prompt = st.text_area("Enter your prompt for analysis or use one of the pre-defined prompts below:")
    
    # Predefined prompts
    predefined_prompts = ["Summarize the dataset.", 
                          "What are the top trends in the data?", 
                          "Generate a statistical analysis report."]
    
    if st.checkbox("Show predefined prompts"):
        st.write(predefined_prompts)

    selected_prompt = st.selectbox("Select a predefined prompt (optional)", ["None"] + predefined_prompts)

    if selected_prompt != "None":
        prompt = selected_prompt

    # Prompt execution
    if st.button("Generate Analysis"):
        if prompt:
            with st.spinner("Generating response..."):
                try:
                    st.write(pandas_ai.run(df, prompt=prompt))
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a prompt.")

else:
    st.info("Please upload a CSV file to start.")
