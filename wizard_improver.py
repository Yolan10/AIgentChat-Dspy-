"""Module for improving wizard prompts using DSPy optimizers."""
from __future__ import annotations

from typing import List

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

        logs: str = dspy.InputField()
        goal: str = dspy.InputField()
        improved_prompt: str = dspy.OutputField()


    class WizardImprover(dspy.Module):
        """Wraps a ReAct agent that proposes improved prompts."""

        def __init__(self) -> None:
            super().__init__()
            self.agent = dspy.ReAct(ImproveSignature, tools=[])

        def forward(self, logs: str, goal: str) -> dspy.Prediction:
            return self.agent(logs=logs, goal=goal)


    def build_dataset(history: List[dict]) -> List[dspy.Example]:
        """Convert conversation history into a DSPy dataset."""
        dataset = []
        for log in history:
            transcript = "\n".join(f"{t['speaker']}: {t['text']}" for t in log.get('turns', []))
            judge = log.get("judge_result", {})
            score = judge.get("score", 0)
            ex = (
                dspy.Example(
                    logs=f"{transcript}\nRESULT: {judge}",
                    goal=log.get("goal"),
                    score=score,
                )
                .with_inputs("logs", "goal")
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
        optimizer = OptimizePrompts(metric=metric, num_candidates=4, auto=None, verbose=False)
        trained = optimizer.compile(improver, trainset=dataset, num_trials=config.DSPY_TRAINING_ITER, provide_traceback=True)

        best_score = max((c.get("score", 0) for c in getattr(trained, "candidate_programs", [])), default=0)
        metrics = {
            "best_score": best_score,
            "iterations": getattr(trained, "candidate_programs", []),
        }
        return trained, metrics

else:  # DSPy not available
    WizardImprover = None
    build_dataset = None
    train_improver = None
