# Cyber GRC Assistant

**Cyber GRC Assistant** is a Streamlit-based application designed to assist with Governance, Risk, and Compliance (GRC) tasks in cybersecurity. The app leverages local LLMs via [Ollama](https://ollama.com/) to map cybersecurity findings to relevant NIST 800-53 controls and provide compliance analysis for each control.

## Features
- Quickly map findings to the most relevant NIST 800-53 controls using semantic search (embeddings)
- For each matched control, receive a detailed compliance assessment, including CCI-level analysis and a summary
- Identify gaps in information and related controls
- Guide entry-level assessors in determining compliance status
- All processing is local—no OpenAI or external API required

## How It Works
1. **User inputs a cybersecurity finding** in the sidebar
2. The app generates an embedding for the finding using an Ollama embedding model
3. The finding is semantically matched to the most relevant NIST 800-53 controls (top 2–5)
4. For each matched control, the app:
    - Presents the control and its CCIs
    - Asks the LLM (via Ollama) to assess compliance for each CCI and the overall control
    - Provides a summary and lists related controls
5. Results are displayed in an interactive, expandable format for easy review

## Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/september-a/Cyber_GRC_Assistant
   cd Cyber_GRC_Assistant
   ```
2. **Install dependencies**
   ```bash
   pip install numpy pandas streamlit ast ollama
   ```
3. **Install and run Ollama**
   - Download and install Ollama from [https://ollama.com/](https://ollama.com/)
   - Start the Ollama server:
     ```bash
     ollama serve
     ```
   - Pull the required models (e.g., nomic-embed-text for embeddings, llama3.2 for chat):
     ```bash
     ollama pull nomic-embed-text
     ollama pull llama3.2
     ```
   - Optionally, set environment variables to customize model selection:
     ```bash
     export OLLAMA_HOST="http://localhost:11434"
     export EMBEDDING_MODEL="nomic-embed-text"
     export CHAT_MODEL="llama3.2"
     ```

4. **Prepare the NIST controls data**
   - Ensure `data/NIST_Controls.csv` exists and contains the required columns and precomputed embeddings. (See the `data/` folder for scripts to build or parse controls.)

## Run
Start the application with:
```bash
streamlit run main.py
```

## Notes
- All LLM and embedding inference is performed locally via Ollama—no data leaves your machine.
- You can adjust the number of top matches and similarity threshold in `main.py`.
- For best results, ensure your Ollama models are up to date and your CSV contains valid embeddings.

## License
See [LICENSE](LICENSE).

