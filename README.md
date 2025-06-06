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
`DSPY_COPRO_MINIBATCH_SIZE`, `DSPY_BOOTSRAP_MINIBATCH_SIZE`, and
`DSPY_MIPRO_MINIBATCH_SIZE` control how many conversation logs trigger each DSPy
optimizer.
When the history contains at most `DSPY_COPRO_MINIBATCH_SIZE` examples the
wizard runs `dspy.COPRO` with an empty dataset. With only a few examples
(fewer than `DSPY_BOOTSRAP_MINIBATCH_SIZE`) it uses `BootstrapFewShot` to
augment the data. Once at least `DSPY_BOOTSRAP_MINIBATCH_SIZE` examples are
available it trains using MIPROv2 (`OptimizePrompts`) with a minibatch size of
`DSPY_MIPRO_MINIBATCH_SIZE`.

Each conversation log now records the wizard's system instruction. The
optimization dataset therefore pairs that instruction with the conversation
transcript and the judge's score so the teleprompters can learn which prompts
lead to better outcomes.

When a population agent is spawned its specification is immediately written to a
log file (e.g. `1.1_<timestamp>_spec_*.json`) so you can inspect it while the
simulation continues. Prompt improvements made by the wizard are also logged in
real time with filenames beginning with `improve_`.

Each improved prompt is additionally appended to `logs/improved_prompts.txt`
with the run number and timestamp. Entries are prefixed with
`instructions=` and long prompts are wrapped every 150 characters for
readability. The prompt improver's own instructions are logged separately
to `logs/improver_instructions.txt` using the same format.



Each invocation of `IntegratedSystem.run` increments `logs/run_counter.txt` and
agents are labelled using `<run>.<index>_<timestamp>` (e.g. `2.1_20240101T120000Z`).
The index increases sequentially for each population agent created during a run
(`1.1`, `1.2`, ...).


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

