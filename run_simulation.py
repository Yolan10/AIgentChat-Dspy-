"""Entry point to run the chat simulation."""
import json

import config
from god_agent import GodAgent
from wizard_agent import WizardAgent
import utils


def main():
    god = GodAgent()
    population = god.spawn_population("Generate population", config.POPULATION_SIZE)
    # log initial population specs before the wizard interacts with them
    for agent in population:
        utils.save_conversation_log(agent.get_spec(), f"{agent.agent_id}_spec.json")

    wizard = WizardAgent(wizard_id="Wizard_001")
    summary = []
    for pop_agent in population:
        log = wizard.converse_with(pop_agent, show_live=config.SHOW_LIVE_CONVERSATIONS)

        filename = f"{wizard.wizard_id}_{pop_agent.agent_id}_{utils.get_timestamp().replace(':', '').replace('-', '')}.json"
        utils.save_conversation_log(log, filename)
        spec = pop_agent.get_spec()
        entry = {
            "pop_agent_id": pop_agent.agent_id,
            "name": spec.get("name"),
            "personality_description": spec.get("personality_description"),
            "system_instruction": spec.get("system_instruction"),
            "temperature": spec.get("llm_settings", {}).get("temperature"),
            "max_tokens": spec.get("llm_settings", {}).get("max_tokens"),
            "success": log["judge_result"].get("success"),
            "score": log["judge_result"].get("score"),
        }
        summary.append(entry)
    utils.save_conversation_log(summary, "summary.json")
    print(f"Completed {len(population)} conversations.")


if __name__ == "__main__":
    main()
