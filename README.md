# AIgentChat-Dspy-

Simple chat simulation using LangChain and Dspy.

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Set your OpenAI API key:

   ```bash
   export OPENAI_API_KEY=YOUR_KEY
   ```

3. Run the simulation:

   ```bash
   python run_simulation.py
   ```

Logs are saved under the `logs/` directory.
The default LLM model is set to `gpt-4.1-nano`. Set `SHOW_LIVE_CONVERSATIONS = True` in
`config.py` if you want each conversation turn printed to the terminal while the
simulation runs.

