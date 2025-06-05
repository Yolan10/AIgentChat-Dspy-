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

When a population agent is spawned its specification is immediately written to a
log file (e.g. `Pop_001_spec_*.json`) so you can inspect it while the
simulation continues. Prompt improvements made by the wizard are also logged in
real time with filenames beginning with `improve_`.

## Summary Output

After running the simulation a `summary.json` file is written under `logs/`.
Each entry contains the results for a population agent with the following
fields:

```json
{
  "pop_agent_id": "Pop_001",
  "name": "Alice",
  "personality_description": "eager shopper",
  "system_instruction": "You are Alice. eager shopper. Respond accordingly.",
  "temperature": 0.7,
  "max_tokens": 512,
  "success": true,
  "score": 0.95
}
```

`temperature` and `max_tokens` come from the agent's LLM settings and show which
parameters were used during the conversation.

