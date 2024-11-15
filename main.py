import os
import numpy as np
import pandas as pd
from openai import OpenAI
from numpy.linalg import norm
from ast import literal_eval

# Create Client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))

# Path to your CSV file
file_path = 'data/NIST_Controls.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

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
    introduction = "First list each control with COMPLIANT, NONCOMPLIANT, or NOT ENOUGH INFO. Then, briefly say why the observation makes the information system compliant or non-compliant. Use the controls provided. Use passive voice."

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

    # Sort by similarity to get the top 10 matches
    top_matches = df[df['similarity'] > 0.79]

    # If there are no matches over .795, get top 5.
    if top_matches.empty:
        top_matches = df.nlargest(5, 'similarity')

    # If there are too many matches (can lead to too many tokens in a request), only give 12
    if len(top_matches) > 12:
        top_matches = df.nlargest(12, 'similarity')

    print(top_matches)
    return top_matches


df = prep_dataframe(df)

# User query
query = input("Enter the finding:")
query_embedding = create_query_embedding(query)

# prep messages
top_matches = get_top_matches(query_embedding, df)
messages = prepare_messages(query, top_matches)

# Generate a response using the OpenAI API
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    temperature=0
)

response_message = response.choices[0].message.content
print(response_message)