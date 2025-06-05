"""Module for improving wizard prompts using DSPy optimizers."""
from __future__ import annotations

from typing import List

import config
import utils

try:
    import dspy
    from dspy.teleprompt.copro_optimizer import COPRO as OptimizePrompts
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
            ex = dspy.Example(
                logs=f"{transcript}\nRESULT: {judge}",
                goal=log.get("goal"),
                score=score,
            ).with_inputs("logs", "goal").with_outputs("score")
            dataset.append(ex)
        return dataset


    def train_improver(dataset: List[dspy.Example]) -> tuple[WizardImprover, dict]:
        """Train a WizardImprover on the dataset."""

        def metric(example: dspy.Example, pred: dspy.Prediction) -> float:
            base = example.score or 0
            bonus = 1.0 if "buy" in pred.improved_prompt.lower() else 0.0
            return base + bonus

        improver = WizardImprover()
        optimizer = OptimizePrompts(metric=metric, depth=config.DSPY_TRAINING_ITER)
        trained = optimizer.compile(improver, trainset=dataset, eval_kwargs={})

        best_score = max((c["score"] for c in getattr(trained, "candidate_programs", [])), default=0)
        metrics = {"best_score": best_score}
        return trained, metrics

else:  # DSPy not available
    WizardImprover = None
    build_dataset = None
    train_improver = None
