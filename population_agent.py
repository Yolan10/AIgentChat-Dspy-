"""Defines the PopulationAgent persona."""
from typing import List, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


import config
import utils


class PopulationAgent:
    """Simple persona-based agent using LangChain for replies."""

    def __init__(self, agent_id: str, name: str, personality_description: str, llm_settings: dict):
        self.agent_id = agent_id
        self.name = name
        self.personality_description = personality_description
        self.llm_settings = llm_settings
        self.state = "undecided"
        self.history: List[Tuple[str, str]] = []  # (speaker, text)
        self.system_instruction = (
            f"You are {self.name}. {self.personality_description}. Respond accordingly."
        )
        self.llm = ChatOpenAI(
            model=llm_settings.get("model", config.LLM_MODEL),
            temperature=llm_settings.get("temperature", config.LLM_TEMPERATURE),
            max_tokens=llm_settings.get("max_tokens", config.LLM_MAX_TOKENS),
        )

    def respond_to(self, user_message: str) -> str:
        messages = [SystemMessage(content=self.system_instruction)]
        for speaker, text in self.history:
            if speaker == "wizard":
                messages.append(HumanMessage(content=text))
            else:
                messages.append(AIMessage(content=text))
        messages.append(HumanMessage(content=user_message))
        response = self.llm.invoke(messages).content

        self.history.append(("wizard", user_message))
        self.history.append(("pop", response))
        return response

    def get_persona(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "personality_description": self.personality_description,
        }

    def get_spec(self) -> dict:
        """Return a spec dictionary describing this population agent."""
        return {
            "name": self.name,
            "personality_description": self.personality_description,
            "system_instruction": self.system_instruction,
            "llm_settings": self.llm_settings,
        }

    def reset_history(self) -> None:
        self.history = []
