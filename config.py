# Configuration file containing all tunable parameters and defaults.

# Population Settings
# Default population size. The value must be at least as large as the
# highest value in ``SELF_IMPROVE_AFTER`` when that setting is a sequence.
POPULATION_SIZE = 36
POPULATION_INSTRUCTION_TEMPLATE_PATH = "templates/population_instruction.txt"

# Wizard Settings
WIZARD_DEFAULT_GOAL = "Convince population to buy"
WIZARD_PROMPT_TEMPLATE_PATH = "templates/wizard_prompt.txt"
MAX_TURNS = 20
# Trigger the wizard's self-improvement step. Provide an ``int`` to run the
# improver every ``n`` conversations or a sequence of ints to trigger on
# specific conversation counts.  For example::
#
#     SELF_IMPROVE_AFTER = [1, 10, 15]  # improve after conversations 1, 10 and 15
#     SELF_IMPROVE_AFTER = 10           # improve after 10, 20, 30, ...
# Trigger improvements after conversations 1, 5 and 36 by default.
SELF_IMPROVE_AFTER = [1, 5, 36]
SELF_IMPROVE_PROMPT_TEMPLATE_PATH = "templates/self_improve_prompt.txt"

# Judge Settings
JUDGE_PROMPT_TEMPLATE_PATH = "templates/judge_prompt.txt"

# LLM Hyperparameters
# Default model to use for all LLM calls
# Updated model name for newer OpenAI releases
LLM_MODEL = "gpt-4.1-nano"

LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 512
LLM_TOP_P = 0.9

# File/Logging Settings
LOGS_DIRECTORY = "logs"
JSON_INDENT = 2

# Runtime Options
# Set to True to print conversation turns to the terminal while running
SHOW_LIVE_CONVERSATIONS = True

# Dspy Settings
DSPY_TRAINING_ITER = 1
DSPY_LEARNING_RATE = 0.01
# Optimizer thresholds
DSPY_BOOTSRAP_MINIBATCH_SIZE = 3
DSPY_MIPRO_MINIBATCH_SIZE = 30
# Maximum number of conversation logs kept in memory for self improvement
HISTORY_BUFFER_LIMIT = 50
# Maximum conversation history stored by each population agent
POP_HISTORY_LIMIT = 50

# Miscellaneous
DEFAULT_TIMEZONE = "UTC"


def _get_last_schedule_point(schedule):
    """Return the last integer from the self-improvement schedule."""
    if isinstance(schedule, int):
        return schedule
    if isinstance(schedule, str):
        try:
            points = [int(x) for x in schedule.split(";") if x.strip()]
        except ValueError:
            return None
        return points[-1] if points else None
    try:
        points = [int(x) for x in schedule]
    except (TypeError, ValueError):
        return None
    return points[-1] if points else None


_last_point = _get_last_schedule_point(SELF_IMPROVE_AFTER)
if _last_point is not None and POPULATION_SIZE < _last_point:
    raise ValueError(
        f"POPULATION_SIZE ({POPULATION_SIZE}) must be at least "
        f"as large as the last entry in SELF_IMPROVE_AFTER ({_last_point})"
    )
