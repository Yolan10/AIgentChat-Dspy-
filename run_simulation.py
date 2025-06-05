"""Entry point to run the chat simulation."""
import json

import config
from god_agent import GodAgent
from wizard_agent import WizardAgent
import utils


def main():
    god = GodAgent()
    population = god.spawn_population("Generate population", config.POPULATION_SIZE)
    wizard = WizardAgent(wizard_id="Wizard_001")
    summary = []
    for pop_agent in population:
        log = wizard.converse_with(pop_agent, show_live=config.SHOW_LIVE_CONVERSATIONS)

        filename = f"{wizard.wizard_id}_{pop_agent.agent_id}_{utils.get_timestamp().replace(':', '').replace('-', '')}.json"
        utils.save_conversation_log(log, filename)
        summary.append({
            "pop_agent_id": pop_agent.agent_id,
            "success": log["judge_result"].get("success"),
            "score": log["judge_result"].get("score"),
        })
    utils.save_conversation_log(summary, "summary.json")
    print(f"Completed {len(population)} conversations.")


if __name__ == "__main__":
    main()
