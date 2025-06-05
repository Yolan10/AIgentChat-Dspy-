# Configuration file containing all tunable parameters and defaults.

# Population Settings
POPULATION_SIZE = 10
POPULATION_INSTRUCTION_TEMPLATE_PATH = "templates/population_instruction.txt"

# Wizard Settings
WIZARD_DEFAULT_GOAL = "Convince population to buy"
WIZARD_PROMPT_TEMPLATE_PATH = "templates/wizard_prompt.txt"
MAX_TURNS = 20
SELF_IMPROVE_AFTER = 10
SELF_IMPROVE_PROMPT_TEMPLATE_PATH = "templates/self_improve_prompt.txt"

# Judge Settings
JUDGE_PROMPT_TEMPLATE_PATH = "templates/judge_prompt.txt"

# LLM Hyperparameters
# Default model to use for all LLM calls
LLM_MODEL = "gpt-4.1-nano"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 512
LLM_TOP_P = 0.9

# File/Logging Settings
LOGS_DIRECTORY = "logs"
JSON_INDENT = 2

# Runtime Options
# Set to True to print conversation turns to the terminal while running
SHOW_LIVE_CONVERSATIONS = False

# Dspy Settings
DSPY_TRAINING_ITER = 1
DSPY_LEARNING_RATE = 0.01

# Miscellaneous
DEFAULT_TIMEZONE = "UTC"
