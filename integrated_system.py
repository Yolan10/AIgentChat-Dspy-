"""High level integration layer tying all components together."""
from __future__ import annotations

from typing import List

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
        for pop in population:
            log = self.wizard.converse_with(pop)
            self.logger.log_event("conversation_end", pop_agent=pop.agent_id, success=log.get("judge_result", {}).get("success"))
        self.logger.log_event("system_end")
