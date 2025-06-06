"""Module for improving wizard prompts using DSPy optimizers."""
from __future__ import annotations

from typing import Any, List

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


    def _extract_instructions(program) -> str:
        """Return the instructions string from a candidate program."""
        if program is None:
            return ""

        # Handle DSPy modules returned by the optimizer
        try:
            if hasattr(program, "signature"):
                text = getattr(program.signature, "instructions", None)
                if isinstance(text, str) and text:
                    return text.strip()
            if hasattr(program, "agent") and hasattr(program.agent, "signature"):
                text = getattr(program.agent.signature, "instructions", None)
                if isinstance(text, str) and text:
                    return text.strip()
        except Exception:
            pass

        text = str(program)
        match = re.search(
            r"instructions=(\"\"\".*?\"\"\"|\".*?\"|'.*?')",
            text,

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
        if len(dataset) < config.DSPY_MINIBATCH_SIZE:
            optimizer = dspy.COPRO(metric=metric)
            trained = optimizer.compile(
                improver,
                trainset=dataset,
                eval_kwargs={"provide_traceback": True},
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
        best_score = max((c.get("score", 0) for c in candidates), default=0)
        best_prompt = ""
        if candidates:
            best = max(candidates, key=lambda c: c.get("score", 0))
            best_prompt = _extract_instructions(best.get("program", ""))
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
