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

   DSPy's optimizers require a configured language model. If you do not
   configure `dspy.settings` yourself, the code will initialize a default
   `dspy.LM` using the values in `config.py` and your OpenAI key.

3. Run the simulation:

   ```bash
   python main.py
   ```

Logs are saved under the `logs/` directory.
The default LLM model is set to `gpt-4o`. Set `SHOW_LIVE_CONVERSATIONS = True` in
`config.py` if you want each conversation turn printed to the terminal while the
simulation runs.
`DSPY_MINIBATCH_SIZE` controls the number of examples used per batch when the
wizard optimizer trains on conversation history. If fewer examples are
available, the system automatically falls back to `dspy.COPRO` for training.

When a population agent is spawned its specification is immediately written to a
log file (e.g. `1.1_<timestamp>_spec_*.json`) so you can inspect it while the
simulation continues. Prompt improvements made by the wizard are also logged in
real time with filenames beginning with `improve_`.

Each invocation of `IntegratedSystem.run` increments `logs/run_counter.txt` and
agents are labelled using `<run>.<index>_<timestamp>` (e.g. `2.1_20240101T120000Z`).

## Summary Output

After running the simulation a `summary_<run>.json` file is written under `logs/`.
Each entry contains the results for a population agent with the following
fields:

```json
{
  "pop_agent_id": "1.1_20240101T120000Z",
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

