"""Defines the PopulationAgent persona."""
from typing import List, Tuple

from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

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
        self.llm = ChatOpenAI(
            model=llm_settings.get("model", config.LLM_MODEL),
            temperature=llm_settings.get("temperature", config.LLM_TEMPERATURE),
            max_tokens=llm_settings.get("max_tokens", config.LLM_MAX_TOKENS),
        )

    def respond_to(self, user_message: str) -> str:
        system_prompt = (
            f"You are {self.name}. {self.personality_description}. Respond accordingly."
        )
        messages = [SystemMessage(content=system_prompt)]
        for speaker, text in self.history:
            if speaker == "wizard":
                messages.append(HumanMessage(content=text))
            else:
                messages.append(AIMessage(content=text))
        messages.append(HumanMessage(content=user_message))
        response = self.llm(messages).content
        self.history.append(("wizard", user_message))
        self.history.append(("pop", response))
        return response

    def get_persona(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "personality_description": self.personality_description,
        }

    def reset_history(self) -> None:
        self.history = []
