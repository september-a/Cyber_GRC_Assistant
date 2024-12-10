import os
import re
import numpy as np
import pandas as pd
from openai import OpenAI
from numpy.linalg import norm
from ast import literal_eval
import streamlit as st

# Create Client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))

# Path to your CSV file
file_path = 'data/NIST_Controls.csv'

def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))

def create_query_embedding(query):
    response = client.embeddings.create(
        input=query,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def prep_dataframe(df):
    # Convert embeddings from string to list if needed
    df['embedding'] = df['embedding'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)

    return df

def prepare_messages(query, controls):
    # Instructions for the output
    introduction = "First list each control as COMPLIANT, NONCOMPLIANT, or NOT ENOUGH INFO. Then, briefly say why the observation makes the information system compliant or non-compliant. Use the controls provided. If there is NOT ENOUGH INFO, describe the information needed to make a decision. Use passive voice."

    # User's input
    query = f"\n\nObservation: {query}"

    # Add matching controls
    message = ""
    for _, control in controls.iterrows():
        text = (
            f"Family: {control['FAMILY']}\n"
            f"Name: {control['NAME']}\n"
            f"Title: {control['TITLE']}\n"
            f"Priority: {control['PRIORITY']}\n"
            f"Baseline Impact: {control['BASELINE_IMPACT']}\n"
            f"Description: {control['DESCRIPTION']}\n"
            f"Supplemental Guidance: {control['SUPPLEMENTAL_GUIDANCE']}\n"
            f"CCIs: {control['ccis']}\n"
            f"Related Controls: {control['RELATED']}\n"
        )
        message = message + text

    message = introduction + query + message

    # Prepare messages for the chat completion
    messages = [
        {"role": "system", "content": "You make assessments about informaion system cybersecurity compliance."},
        {"role": "user", "content": message},
    ]

    return messages

def get_top_matches(query_embedding, df):
    # Compute similarity for each control
    df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity(query_embedding, np.array(x)))

    # Sort by similarity to get matches over .79
    top_matches = df[df['similarity'] > 0.78]

    # If there are no matches over .795, get top 5.
    if len(top_matches) < 2:
        top_matches = df.nlargest(2, 'similarity')

    # If there are too many matches (can lead to too many tokens in a request), only give 12
    if len(top_matches) > 5:
        top_matches = df.nlargest(5, 'similarity')

    return top_matches

def clean_up_matches(df):
    columns_to_hide = ["FAMILY", "TITLE", "PRIORITY", "BASELINE_IMPACT", "embedding"]
    filtered_df = df.drop(columns=columns_to_hide)
    return filtered_df

def create_structured_response(response):
        # Define the regular expression pattern
    pattern = r"Control:\s*(.+?)\s*Assessment:\s*(.+?)\s*Reason:\s*(.+?)(?=Control:|$)"
    
    # Find all matches in the input text
    matches = re.findall(pattern, response, re.DOTALL)
    
    # Create a regular dictionary to hold the data
    structured_data = {}
    
    for control, assessment, reason in matches:
        control = control.strip()
        assessment = assessment.strip()
        reason = reason.strip()
        
        structured_data[control] = {
            "Assessment": assessment,
            "Reason": reason
        }
    
    return structured_data

# Main function
def main():
    st.title("Cyber GRC Assistant")

    # Initialize session state
    if "query" not in st.session_state:
        st.session_state.query = ""
    if "top_matches" not in st.session_state:
        st.session_state.top_matches = None
    if "response_message" not in st.session_state:
        st.session_state.response_message = None
    if "findings" not in st.session_state:
        st.session_state.findings = {}

    # Input for user query
    st.sidebar.title("Findings")
    query = st.sidebar.text_area("Input finding:")

    # Load and prepare the DataFrame
    file_path = "data/NIST_Controls.csv"  # Update with your actual file path
    try:
        df = pd.read_csv(file_path)
        df = prep_dataframe(df)
    except Exception as e:
        st.error(f"Error loading or processing the file: {e}")
        return

    if query.strip() == "":
        st.sidebar.warning("Please enter a valid query.")
    else:
        # Update session state with the current query
        st.session_state.query = query

        # Generate query embedding and find top matches
        query_embedding = create_query_embedding(query)
        top_matches = get_top_matches(query_embedding, df)

        # Clean up matches and save to session state
        top_matches_pretty = clean_up_matches(top_matches)
        st.session_state.top_matches = top_matches_pretty

        # Prepare messages and get the response
        messages = prepare_messages(query, top_matches)
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0,
            )
            st.session_state.response_message = response.choices[0].message.content
        except Exception as e:
            st.error(f"Error generating response: {e}")
            return

    # Display results if they exist in session state
    if st.session_state.top_matches is not None:
        st.subheader("Top Matches")
        st.dataframe(data=st.session_state.top_matches, use_container_width=True)

    if st.session_state.response_message is not None:
        st.subheader("Generated Response")
        controls = create_structured_response(st.session_state.response_message)
        for control, details in controls.items():
            with st.expander(control):
                st.text(f"Assessment: {details['Assessment']}")
                st.text(f"Reason: {details['Reason']}")
        if st.sidebar.button("Save Finding Results"):
            st.session_state.findings[query] = controls
            st.success(f"Finding '{query}' saved!")

    if st.session_state.findings:
        st.sidebar.subheader("Saved Findings")
        for finding in st.session_state.findings:
            st.sidebar.button(finding)

if __name__ == "__main__":
    main()
                


    