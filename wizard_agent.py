"""WizardAgent interacts with population agents and self-improves."""
from __future__ import annotations

from typing import Dict, List
import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


import config
import utils
from judge_agent import JudgeAgent

# Dspy is imported as placeholder - this code assumes Dspy provides a simple API
# to fine tune prompts. Replace with actual implementation when available.
try:
    import dspy
except ImportError:  # pragma: no cover - dspy not installed
    dspy = None


class ConversationLog(dict):
    pass


class WizardAgent:
    def __init__(self, wizard_id: str, goal: str | None = None, llm_settings: dict | None = None):
        self.wizard_id = wizard_id
        self.goal = goal or config.WIZARD_DEFAULT_GOAL
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
        self.system_prompt_template = utils.load_template(config.WIZARD_PROMPT_TEMPLATE_PATH)
        self.current_prompt = utils.render_template(self.system_prompt_template, {"goal": self.goal})
        self.conversation_count = 0
        self.history_buffer: List[ConversationLog] = []
        self.improvement_history: List[dict] = []

    def converse_with(self, pop_agent, show_live: bool = False) -> ConversationLog:

        log = {
            "wizard_id": self.wizard_id,
            "pop_agent_id": pop_agent.agent_id,
            "pop_agent_spec": pop_agent.get_spec(),
            "goal": self.goal,
            "turns": [],
            "timestamp": utils.get_timestamp(),
        }
        for _ in range(config.MAX_TURNS):
            messages = [SystemMessage(content=self.current_prompt)]
            for t in log["turns"]:
                if t["speaker"] == "wizard":
                    messages.append(HumanMessage(content=t["text"]))
                else:
                    messages.append(AIMessage(content=t["text"]))

            wizard_msg = self.llm.invoke(messages).content
            log["turns"].append({"speaker": "wizard", "text": wizard_msg, "time": utils.get_timestamp()})
            if show_live:
                print(f"Wizard: {wizard_msg}")
            pop_reply = pop_agent.respond_to(wizard_msg)
            log["turns"].append({"speaker": "pop", "text": pop_reply, "time": utils.get_timestamp()})
            if show_live:
                print(f"{pop_agent.name}: {pop_reply}")

            if self._check_goal(pop_reply):
                break
        judge = JudgeAgent()
        result = judge.assess(log)
        log["judge_result"] = result
        self.history_buffer.append(log)
        self.conversation_count += 1
        if self.conversation_count % config.SELF_IMPROVE_AFTER == 0:
            self.self_improve()
        return log

    def _check_goal(self, text: str) -> bool:
        return "buy" in text.lower()

    def self_improve(self) -> None:
        if dspy is None:
            return
        logs_json = json.dumps(self.history_buffer)
        template = utils.load_template(config.SELF_IMPROVE_PROMPT_TEMPLATE_PATH)
        prompt = utils.render_template(template, {"logs": logs_json})
        try:
            new_prompt = dspy.improve_prompt(self.current_prompt, prompt,
                                             iterations=config.DSPY_TRAINING_ITER,
                                             lr=config.DSPY_LEARNING_RATE)
        except Exception:
            new_prompt = self.current_prompt + "\n# improved"

        entry = {
            "timestamp": utils.get_timestamp(),
            "old_prompt": self.current_prompt,
            "new_prompt": new_prompt,
        }
        filename = f"improve_{entry['timestamp'].replace(':', '').replace('-', '')}.json"
        utils.save_conversation_log(entry, filename)
        self.improvement_history.append(entry)
        self.current_prompt = new_prompt
        self.history_buffer.clear()
