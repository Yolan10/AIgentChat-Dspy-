"""GodAgent spawns population agents."""
from __future__ import annotations

import json
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


import config
import utils
from population_agent import PopulationAgent


class GodAgent:
    def __init__(self, llm_settings: dict | None = None):
        self.llm_settings = llm_settings or {
            "model": config.LLM_MODEL,
            "temperature": config.LLM_TEMPERATURE,
            "max_tokens": config.LLM_MAX_TOKENS,
        }
        self.llm = ChatOpenAI(
            model=self.llm_settings["model"],
            temperature=self.llm_settings["temperature"],
            max_tokens=self.llm_settings["max_tokens"],
        )
        self.template = utils.load_template(config.POPULATION_INSTRUCTION_TEMPLATE_PATH)

    def spawn_population(self, instruction_text: str, n: int | None = None, run_no: int = 0) -> List[PopulationAgent]:
        n = n or config.POPULATION_SIZE
        prompt = utils.render_template(self.template, {"instruction": instruction_text, "n": n})
        messages = [SystemMessage(content=prompt), HumanMessage(content="Provide the JSON array only.")]
        response = self.llm.invoke(messages).content
        try:
            personas = json.loads(response)
        except json.JSONDecodeError:
            personas = utils.extract_json_array(response)
            if personas is None:
                raise
        population = []
        for idx, spec in enumerate(personas):
            agent = PopulationAgent(
                agent_id=utils.format_agent_id(run_no, idx + 1),
                name=spec.get("name"),
                personality_description=spec.get("personality"),
                llm_settings=self.llm_settings,
            )
            population.append(agent)

            # Save the agent specification immediately so users can inspect it
            log_filename = f"{agent.agent_id}_spec_{utils.get_timestamp().replace(':', '').replace('-', '')}.json"
            utils.save_conversation_log(agent.get_spec(), log_filename)
            print(f"Created {agent.agent_id} -> {log_filename}")

        return population
