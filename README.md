# Cyber GRC Assistant

**Cyber GRC Assistant** is a Streamlit-based application designed to assist with Governance, Risk, and Compliance (GRC) tasks in cybersecurity. The application will take cyber security findings and map them to a select number of cybersecurity controls in the NIST 800-53 along with a brief compliancy statement.

**Use Cases**
- Quickly mapping findings to appropriate controls, perhaps while being discussed in meetings or onsite.
- Guiding entry-level assessors with determining compliance status.
- Identifying gaps in knowledge of the security posture of the system. 

# Set-up
1. Clone the repository `git clone https://github.com/september-a/Cyber_GRC_Assistant`
2. Solve dependences; `pip install openai numpy pandas streamlit ast tiktoken`
4. You will need your own OpenAI API key to run the application. Go to [the OpenAI API Reference](https://platform.openai.com/docs/api-reference/introduction) to learn how to get a key. You may need to buy some usage credits to use your key.
5. Add your key to your credentials `export OPENAI_API_KEY="YOUR SECRET KEY HERE"`

# Run
Run the application with:
`streamlit run main.py`

