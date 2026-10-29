"""One policy step with optional verification; no environment executes here."""
from __future__ import annotations

from swe_verifier.inference.messages import command, policy_message
from swe_verifier.inference.scoring import Scorer, SlateSelector
from swe_verifier.inference.session import ModelSession
from swe_verifier.inference.types import Candidate, Decision, InvalidOutput, Message, QueryModel, RuntimeConfig, Score


def choose_action(
    task: str, history: list[Message], policy: QueryModel, policy_name: str,
    gate: Scorer | None, selector: Scorer | SlateSelector | None,
    config: RuntimeConfig, session: ModelSession,
) -> Decision:
    """Generate alternatives only after a low score; append nothing to history.

    The initial action's score is reused only when gate and selector are the
    same Scorer instance. Separate selectors never inherit that score.
    """
    if config.strategy == "selective" and (gate is None or selector is None):
        raise ValueError("Selective verification requires explicit gate and selector")
    if config.strategy == "all_step" and selector is None:
        raise ValueError("All-step verification requires an explicit selector")
    start = len(session.usage)
    # Mini's policy adapter understands provider-native reasoning fields. The
    # verifier's narrower message projection must not change policy history.
    policy_history = [policy_message(message) for message in history]
    candidates = [Candidate("0", session.query(policy, policy_name, policy_history, "policy"))]
    valid: list[Candidate] = []
    invalid: list[str] = []
    try:
        command(candidates[0])
        valid.append(candidates[0])
    except InvalidOutput as exc:
        invalid.append(str(exc))
    if not valid:
        raise InvalidOutput(f"Initial policy action is invalid: {invalid[0]}")
    if config.strategy == "none":
        return Decision(valid[0], tuple(candidates), None, (), False, None, tuple(session.usage[start:]), ())
    gate_score: Score | None = None
    scores: list[tuple[str, Score]] = []
    triggered = config.strategy == "all_step"
    try:
        if config.strategy == "selective":
            assert gate is not None
            gate_score = gate.score(task, history, valid[0], "gate", session)
            if gate_score.value is None:
                reason = "No available gate score; research behavior retains the initial valid action without resampling"
                if config.failure_policy == "raise":
                    raise InvalidOutput(reason)
                session.diagnostics.append({"event": "missing_gate_score", "selected_id": valid[0].id,
                                            "triggered": False, "reason": reason})
                return Decision(valid[0], tuple(candidates), gate_score, (), False, reason,
                                tuple(session.usage[start:]), ())
            if gate_score.value >= config.threshold:
                return Decision(valid[0], tuple(candidates), gate_score, (), False, None, tuple(session.usage[start:]), ())
        triggered = True
        for index in range(1, config.candidate_count):
            candidate = Candidate(str(index), session.query(policy, policy_name, policy_history, "policy"))
            candidates.append(candidate)
            try:
                command(candidate)
            except InvalidOutput as exc:
                invalid.append(str(exc))
            else:
                valid.append(candidate)
        if isinstance(selector, SlateSelector):
            selected, outputs = selector.select(task, history, tuple(valid), session)
        else:
            assert selector is not None
            for candidate in valid:
                score = (gate_score if candidate.id == "0" and selector is gate and gate_score is not None
                         else selector.score(task, history, candidate, "selector", session))
                scores.append((candidate.id, score))
            available = [(candidate_id, score.value) for candidate_id, score in scores if score.value is not None]
            if not available:
                raise InvalidOutput("No available candidate scores; research first-valid fallback required")
            chosen_id = max(available, key=lambda item: item[1])[0]
            selected = next(candidate for candidate in valid if candidate.id == chosen_id)
            outputs = ()
        return Decision(selected, tuple(candidates), gate_score, tuple(scores), True,
                        "; ".join(invalid) if invalid else None, tuple(session.usage[start:]), outputs)
    except InvalidOutput as exc:
        if config.failure_policy == "raise":
            raise
        session.diagnostics.append({"event": "first_valid_fallback", "selected_id": valid[0].id,
                                    "triggered": triggered, "reason": str(exc)})
        return Decision(valid[0], tuple(candidates), gate_score, tuple(scores), triggered,
                        str(exc), tuple(session.usage[start:]), ())
