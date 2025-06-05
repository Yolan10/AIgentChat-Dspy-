"""JudgeAgent evaluates conversation logs."""
from __future__ import annotations

from typing import Dict

import json
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

import config
import utils


class JudgeAgent:
    def __init__(self, llm_settings: dict | None = None, judge_prompt_template: str | None = None):
        self.llm_settings = llm_settings or {
            "model": config.LLM_MODEL,
            "temperature": 0.3,
            "max_tokens": config.LLM_MAX_TOKENS,
        }
        self.llm = ChatOpenAI(
            model=self.llm_settings["model"],
            temperature=self.llm_settings["temperature"],
            max_tokens=self.llm_settings["max_tokens"],
        )
        self.template = judge_prompt_template or utils.load_template(config.JUDGE_PROMPT_TEMPLATE_PATH)

    def assess(self, log: Dict) -> Dict:
        transcript = "\n".join([f"{t['speaker']}: {t['text']}" for t in log["turns"]])
        prompt = utils.render_template(self.template, {"goal": log.get("goal"), "transcript": transcript})
        messages = [SystemMessage(content=prompt), HumanMessage(content="Return JSON with success, score, rationale.")]
        result = self.llm(messages).content
        return json.loads(result)
