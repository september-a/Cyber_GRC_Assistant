import csv
import os
import pandas as pd
from openai import OpenAI
import tiktoken

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))

# Define the file with controls
file_path = "NIST_800-53_Rev_4_Controls.csv"

# Initialize tokenizer for token counting
tokenizer = tiktoken.get_encoding("cl100k_base")

# Define the maximum number of tokens for the embedding model
MAX_TOKENS = 8191

SAVE_PATH = "data/NIST_Controls.csv"

def read_csv(file_path):
    controls = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        current_control = {}
        for row in reader:
            # If a new control is encountered, add the last one and start a new entry
            if row['NAME'] and current_control:
                controls.append(current_control)
                current_control = {}

            # Populate control details from CSV fields
            current_control.setdefault("FAMILY", row["FAMILY"])
            current_control.setdefault("NAME", row["NAME"])
            current_control.setdefault("TITLE", row["TITLE"])
            current_control.setdefault("PRIORITY", row["PRIORITY"])
            current_control.setdefault("BASELINE_IMPACT", row["BASELINE-IMPACT"])
            current_control.setdefault("DESCRIPTION", row["DESCRIPTION"])
            current_control.setdefault("SUPPLEMENTAL_GUIDANCE", row["SUPPLEMENTAL GUIDANCE"])
            current_control.setdefault("RELATED", row["RELATED"])

        # Add the last control to the list
        if current_control:
            controls.append(current_control)
    return controls

def count_tokens(text):
    """Count tokens in a given text."""
    tokens = tokenizer.encode(text)
    return len(tokens)

def truncate_text(text, max_tokens):
    """Truncate text to fit within the max_tokens limit."""
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return tokenizer.decode(truncated_tokens)

def create_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def process_controls(controls):
    for control in controls:
        # Concatenate fields into a single string for embedding generation
        text = (
            f"Family: {control['FAMILY']}\n"
            f"Name: {control['NAME']}\n"
            f"Title: {control['TITLE']}\n"
            f"Priority: {control['PRIORITY']}\n"
            f"Baseline Impact: {control['BASELINE_IMPACT']}\n"
            f"Description: {control['DESCRIPTION']}\n"
            f"Supplemental Guidance: {control['SUPPLEMENTAL_GUIDANCE']}\n"
            f"Related Controls: {control['RELATED']}"
        )
        
        # Check token count and truncate if necessary
        token_count = count_tokens(text)
        if token_count > MAX_TOKENS:
            print(f"Warning: Truncating text for {control['NAME']} from {token_count} tokens to {MAX_TOKENS} tokens.")
            text = truncate_text(text, MAX_TOKENS)

        # Generate and store the embedding
        embedding = create_embedding(text)
        control["embedding"] = embedding  # Add embedding to control dictionary

        print(f"{control['NAME']} being processed")
    return controls

def main():
    
    # Read and process controls
    controls = read_csv(file_path)
    processed_controls = process_controls(controls)
    
    # Optionally, save embeddings to a file or database
    df = pd.DataFrame(processed_controls)
    df.to_csv(SAVE_PATH, index=False)

main()
