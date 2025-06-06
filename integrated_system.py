"""High level integration layer tying all components together."""
from __future__ import annotations

from typing import List

import config
import utils

from god_agent import GodAgent
from wizard_agent import WizardAgent
from advanced_features import PopulationGenerator
from logging_system import StructuredLogger


class IntegratedSystem:
    """Coordinates population generation, conversations and logging."""

    def __init__(self) -> None:
        self.logger = StructuredLogger()
        self.generator = PopulationGenerator()
        self.god = GodAgent()
        self.wizard = WizardAgent(wizard_id="Wizard_001")

    def run(self, instruction: str, n: int) -> None:
        self.logger.log_event("system_start", instruction=instruction, n=n)
        specs = self.generator.generate(instruction, n)
        population: List = []
        for idx, spec in enumerate(specs):
            agent = self.god.spawn_population(spec.get("personality"), 1)[0]
            population.append(agent)

        summary: List[dict] = []
        for pop in population:
            log = self.wizard.converse_with(pop, show_live=config.SHOW_LIVE_CONVERSATIONS)
            filename = f"{self.wizard.wizard_id}_{pop.agent_id}_{utils.get_timestamp().replace(':', '').replace('-', '')}.json"
            utils.save_conversation_log(log, filename)
            spec = pop.get_spec()
            entry = {
                "pop_agent_id": pop.agent_id,
                "name": spec.get("name"),
                "personality_description": spec.get("personality_description"),
                "system_instruction": spec.get("system_instruction"),
                "temperature": spec.get("llm_settings", {}).get("temperature"),
                "max_tokens": spec.get("llm_settings", {}).get("max_tokens"),
                "success": log["judge_result"].get("success"),
                "score": log["judge_result"].get("score"),
            }
            summary.append(entry)
            self.logger.log_event(
                "conversation_end",
                pop_agent=pop.agent_id,
                success=entry["success"],
            )

        utils.save_conversation_log(summary, "summary.json")
        self.logger.log_event("system_end")
        print(f"Completed {len(population)} conversations.")
