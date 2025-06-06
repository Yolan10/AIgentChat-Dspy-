"""Module for improving wizard prompts using DSPy optimizers."""
from __future__ import annotations

from typing import List
import re

import config
import utils

try:
    import dspy
    from dspy.teleprompt.mipro_optimizer_v2 import MIPROv2 as OptimizePrompts
except Exception:  # pragma: no cover - DSPy optional
    dspy = None


if dspy is not None:

    class ImproveSignature(dspy.Signature):
        """Signature for generating a better system prompt."""

        instruction: str = dspy.InputField()
        logs: str = dspy.InputField()
        goal: str = dspy.InputField()
        improved_prompt: str = dspy.OutputField()


    class WizardImprover(dspy.Module):
        """Wraps a ReAct agent that proposes improved prompts."""

        def __init__(self) -> None:
            super().__init__()
            self.agent = dspy.ReAct(ImproveSignature, tools=[])

        def forward(self, instruction: str, logs: str, goal: str) -> dspy.Prediction:
            return self.agent(instruction=instruction, logs=logs, goal=goal)


    def _extract_instructions(program: object) -> str:
        """Return the instructions string from a candidate program.

        ``program`` may be a raw source string or a compiled ``dspy`` module.
        When a module is provided we attempt to read the ``signature.instructions``
        attribute, falling back to regex extraction from the string
        representation of ``program``.
        """

        # If ``program`` is a compiled module, try grabbing the instructions
        sig = getattr(program, "signature", None)
        if sig is not None and hasattr(sig, "instructions"):
            text = sig.instructions

            if isinstance(text, str) and text.strip():
                cleaned = text.strip()
                if cleaned == "Signature for generating a better system prompt.":
                    return utils.load_template(config.WIZARD_PROMPT_TEMPLATE_PATH).strip()
                return cleaned


        # Otherwise handle the value as a plain string
        if not isinstance(program, str):
            program = str(program)

        match = re.search(
            r"instructions=(\"\"\".*?\"\"\"|\".*?\"|'.*?')",
            program,
            re.DOTALL,
        )
        if not match:
            return ""
        text = match.group(1)
        if text.startswith('"""') and text.endswith('"""'):
            return text[3:-3].strip()
        if text.startswith('"') and text.endswith('"'):
            return text[1:-1].strip()
        if text.startswith("'") and text.endswith("'"):
            return text[1:-1].strip()
        return text.strip()


    def build_dataset(history: List[dict]) -> List[dspy.Example]:
        """Convert conversation history into a DSPy dataset."""
        dataset = []
        for log in history:
            transcript = "\n".join(f"{t['speaker']}: {t['text']}" for t in log.get('turns', []))
            judge = log.get("judge_result", {})
            score = judge.get("score", 0)
            ex = (
                dspy.Example(
                    instruction=log.get("prompt", ""),
                    logs=f"{transcript}\nRESULT: {judge}",
                    goal=log.get("goal"),
                    score=score,
                )
                .with_inputs("instruction", "logs", "goal")
            )
            dataset.append(ex)
        return dataset


    def train_improver(dataset: List[dspy.Example]) -> tuple[WizardImprover, dict]:
        """Train a WizardImprover on the dataset."""

        if dspy.settings.lm is None:
            dspy.settings.configure(
                lm=dspy.LM(
                    model=config.LLM_MODEL,
                    temperature=config.LLM_TEMPERATURE,
                    max_tokens=config.LLM_MAX_TOKENS,
                )
            )

        def metric(example: dspy.Example, pred: dspy.Prediction) -> float:
            base = example.score or 0
            bonus = 1.0 if "buy" in pred.improved_prompt.lower() else 0.0
            return base + bonus

        improver = WizardImprover()
        if len(dataset) == 0:
            optimizer = dspy.COPRO(metric=metric)
            trained = optimizer.compile(
                improver,
                trainset=dataset,
                eval_kwargs={"provide_traceback": True},
            )
        elif len(dataset) < config.DSPY_MINIBATCH_SIZE:
            optimizer = dspy.teleprompt.BootstrapFewShot(metric=metric)
            trained = optimizer.compile(
                improver,
                trainset=dataset,
            )
        else:
            optimizer = OptimizePrompts(
                metric=metric,
                num_candidates=4,
                auto=None,
                verbose=False,
            )
            trained = optimizer.compile(
                improver,
                trainset=dataset,
                num_trials=config.DSPY_TRAINING_ITER,
                provide_traceback=True,
                minibatch_size=config.DSPY_MINIBATCH_SIZE,
            )

        candidates = getattr(trained, "candidate_programs", [])
        if candidates:
            best = max(candidates, key=lambda c: c.get("score", 0))
            best_prompt = _extract_instructions(best.get("program", ""))
            best_score = best.get("score", 0)
        else:
            best_prompt = _extract_instructions(trained)
            best_score = 0
        metrics = {
            "best_score": best_score,
            "iterations": candidates,
            "best_prompt": best_prompt,
        }
        return trained, metrics

else:  # DSPy not available
    WizardImprover = None
    build_dataset = None
    train_improver = None
