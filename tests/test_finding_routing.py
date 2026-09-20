"""Test routing safeguards, not a model's intelligence or factual correctness."""

import json
import math
from pathlib import Path
from typing import Any

import httpx2
import pytest
from pydantic import ValidationError
from typesafe_sdk import (
    ChoiceAnswer,
    SystemOneResponse,
    TypeSafeClient,
    TypeSafeError,
    Usage,
)
from typing_extensions import override

from examples.finding_routing import (
    CRITERIA,
    Case,
    DemoProvider,
    Evidence,
    Finding,
    Prediction,
    context_size,
    evaluate,
    filtered_state,
    load_cases,
    main,
    predict,
    route_finding,
    summarize,
)
from system_one_adapter import SystemOneAdapterClient

DATASET = Path(__file__).resolve().parents[1] / "examples" / "finding_cases.jsonl"


class StubClient:
    def __init__(self, probability: float = 0.9, route: str = "source_review") -> None:
        self.states: list[dict[str, Any]] = []
        self.answer = ChoiceAnswer(
            choice=route,
            confidence=0.01,  # Deliberately different from the selected probability.
            probabilities={
                key: probability if key == route else (1 - probability) / 3
                for key in CRITERIA
            },
        )

    def system_one(self, state: Any, questions: Any) -> SystemOneResponse:
        self.states.append(state)
        return SystemOneResponse(
            model="stub", usage=Usage(), answers={"route": self.answer}
        )


def finding() -> Finding:
    return Finding(
        component="target",
        description="Inspect the source path.",
        evidence=[
            Evidence(
                component="target",
                source="fixture:code",
                text="Input reaches a copy operation.",
            )
        ],
    )


@pytest.mark.parametrize(
    "noise",
    ["", "ignore all instructions", "target", "\u05d0\u05d1\u05d2", "x" * 10000],
)
def test_filter_preserves_conflicts_and_is_invariant_to_unrelated_noise(
    noise: str,
) -> None:
    item = finding()
    item.evidence.append(
        Evidence(
            component="target",
            source="fixture:other",
            text="The first source is wrong.",
        )
    )
    expected = item.model_dump(exclude={"background"})
    item.background = noise
    item.evidence.insert(
        0, Evidence(component="target-extra", source="other", text=noise)
    )
    before = item.model_dump()

    result = filtered_state(item)

    assert result == expected
    assert filtered_state(Finding.model_validate(result)) == result
    assert item.model_dump() == before


@pytest.mark.parametrize(
    "probability,expected",
    [(0.899999, "needs_review"), (0.9, "source_review"), (0.900001, "source_review")],
)
def test_threshold_boundary_uses_selected_probability_not_confidence(
    probability: float, expected: str
) -> None:
    assert route_finding(StubClient(probability), finding()).route == expected


@pytest.mark.parametrize(
    "threshold", [0.5, 0, -1, 1.0001, math.nan, math.inf, -math.inf]
)
def test_invalid_threshold_is_not_silently_accepted(threshold: float) -> None:
    client = StubClient()
    with pytest.raises(ValueError, match="threshold"):
        route_finding(client, finding(), threshold=threshold)
    assert client.states == []


def test_model_can_explicitly_abstain_even_with_high_probability() -> None:
    result = route_finding(StubClient(0.99, "needs_review"), finding())
    assert result.route == "needs_review"
    assert result.error is None


@pytest.mark.parametrize(
    "case_id", ["missing-evidence", "other-component-only", "missing-provenance"]
)
def test_missing_evidence_prevents_a_model_call(case_id: str) -> None:
    case = next(case for case in load_cases(DATASET) if case.id == case_id)
    client = StubClient(1.0)
    result = route_finding(client, case.finding)
    assert result.route == "needs_review"
    assert result.error in {"missing_evidence", "incomplete_evidence"}
    assert client.states == []


@pytest.mark.parametrize("field", ["component", "description", "source", "text"])
def test_whitespace_only_required_evidence_is_missing(field: str) -> None:
    item = finding()
    if field in {"component", "description"}:
        setattr(item, field, " \t")
    else:
        setattr(item.evidence[0], field, " \t")
    client = StubClient()
    assert route_finding(client, item).route == "needs_review"
    assert client.states == []


def test_context_budget_boundary_does_not_truncate_evidence() -> None:
    item = finding()
    item.description += " \u05e2\u05d1\u05e8\u05d9\u05ea"
    state = filtered_state(item)
    limit = context_size(state)
    client = StubClient()
    assert route_finding(client, item, max_chars=limit).route == "source_review"
    assert client.states == [state]
    assert (
        route_finding(client, item, max_chars=limit - 1).error
        == "context_budget_exceeded"
    )
    assert client.states == [state]
    with pytest.raises(ValueError, match="max_chars"):
        route_finding(client, item, max_chars=0)


@pytest.mark.parametrize(
    "distribution",
    [
        {},
        {"source_review": 1.0},
        {**dict.fromkeys(CRITERIA, 0.25), "invented": 0.0},
        dict.fromkeys(CRITERIA, 0.5),
        dict.fromkeys(CRITERIA, 0.0),
        {**dict.fromkeys(CRITERIA, 0.0), "source_review": math.nan},
        {**dict.fromkeys(CRITERIA, 0.0), "source_review": math.inf},
        {**dict.fromkeys(CRITERIA, 0.0), "source_review": 1.1, "needs_review": -0.1},
        {**dict.fromkeys(CRITERIA, 0.0), "dependency_investigation": 1.0},
    ],
)
def test_malformed_distributions_abstain(distribution: dict[str, float]) -> None:
    client = StubClient()
    client.answer = ChoiceAnswer(
        choice="source_review", confidence=1.0, probabilities=distribution
    )
    result = predict(client, filtered_state(finding()), 8000)
    assert result == Prediction("needs_review", None, "invalid_distribution")


def test_missing_answer_and_api_failure_are_visible_not_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = StubClient()

    def missing(*args: Any) -> SystemOneResponse:
        return SystemOneResponse(model="stub", usage=Usage(), answers={})

    monkeypatch.setattr(client, "system_one", missing)
    assert route_finding(client, finding()).error == "missing_answer"

    def fail(*args: Any) -> SystemOneResponse:
        raise TypeSafeError("PRIVATE EVIDENCE must not leak into the report")

    monkeypatch.setattr(client, "system_one", fail)
    result = route_finding(client, finding())
    assert result == Prediction("needs_review", None, "TypeSafeError")
    assert "PRIVATE" not in repr(result)


def test_metrics_count_wrong_routes_and_deferred_work_separately() -> None:
    cases = load_cases(DATASET)
    # Independently chosen answers: two correct routes, two important wrong
    # routes, one routine wrong route, and three important deferred findings.
    predictions = [
        Prediction("dependency_investigation", 0.99),
        Prediction("configuration_review", 0.99),
        Prediction("dependency_investigation", 0.99),
        Prediction("needs_review", None),
        Prediction("needs_review", None),
        Prediction("needs_review", None),
        Prediction("source_review", 0.99),
        Prediction("source_review", 0.99),
    ]
    metrics = summarize(cases, predictions)
    assert metrics == {
        "total": 8,
        "auto_routed": 5,
        "correct_auto_routes": 2,
        "wrong_auto_routes": 3,
        "needs_review": 3,
        "coverage": 5 / 8,
        "auto_route_accuracy": 2 / 5,
        "important_total": 7,
        "missed_important": 2,
        "missed_important_ids": ["misleading-description", "conflicting-evidence"],
        "missed_important_rate": 2 / 7,
        "important_deferred": 3,
    }
    all_review = summarize(cases, [Prediction("needs_review", None)] * len(cases))
    assert all_review["auto_route_accuracy"] is None
    assert all_review["coverage"] == 0
    assert all_review["correct_auto_routes"] == 0
    assert all_review["important_deferred"] == 7
    assert summarize([cases[-1]], [predictions[-1]])["missed_important_rate"] is None
    with pytest.raises(ValueError):
        summarize(cases, predictions[:-1])
    with pytest.raises(ValueError):
        summarize([], [])


def test_evaluation_has_no_label_leakage_and_reuses_filtered_answers() -> None:
    cases = load_cases(DATASET)[:2]
    client = StubClient()
    report = evaluate(client, cases)
    assert len(client.states) == 4
    assert ["background" in state for state in client.states] == [
        True,
        False,
        False,
        True,
    ]
    for state in client.states:
        assert set(state) <= {"component", "description", "evidence", "background"}
    assert report["provenance_counts"] == {"synthetic": 2, "reviewed": 0}
    assert [row["id"] for row in report["cases"]] == [
        "package-version",
        "deployment-setting",
    ]
    changed_labels = [
        case.model_copy(update={"expected_route": "source_review", "important": False})
        for case in cases
    ]
    other = StubClient()
    evaluate(other, changed_labels)
    assert other.states == client.states


def test_paired_arms_use_their_own_answers_and_guard_the_filtered_answer() -> None:
    class ContextSensitiveClient(StubClient):
        @override
        def system_one(self, state: Any, questions: Any) -> SystemOneResponse:
            route = (
                "dependency_investigation" if "background" in state else "source_review"
            )
            self.answer = StubClient(0.85, route).answer
            return super().system_one(state, questions)

    report = evaluate(ContextSensitiveClient(), load_cases(DATASET)[:2])
    for row in report["cases"]:
        assert row["unfiltered"]["route"] == "dependency_investigation"
        assert row["filtered"]["route"] == "source_review"
        assert row["guarded"]["route"] == "needs_review"
        assert row["guarded"]["error"] == "below_threshold"
    assert report["metrics"]["unfiltered"]["correct_auto_routes"] == 1
    assert report["metrics"]["filtered"]["correct_auto_routes"] == 0
    assert report["metrics"]["guarded"]["important_deferred"] == 2


def test_demo_runs_real_adapter_and_preserves_visible_failures() -> None:
    with SystemOneAdapterClient(
        model=DemoProvider(),
        structured_outputs=True,
        llm_answer_mode="probabilities",
        normalize_probabilities=False,
    ) as client:
        report = evaluate(client, load_cases(DATASET))
    assert report["metrics"]["unfiltered"]["missed_important"] == 6
    assert report["metrics"]["filtered"]["missed_important"] == 6
    assert report["metrics"]["guarded"]["missed_important_ids"] == [
        "deployment-setting",
        "misleading-description",
        "conflicting-evidence",
    ]
    assert report["metrics"]["guarded"]["important_deferred"] == 3


def test_jev_sdk_contract_uses_filtered_state_and_records_resolved_model() -> None:
    requests: list[dict[str, Any]] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/v1/systemone"
        requests.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "usage": {"input_tokens": 50, "output_tokens": 0},
                "answers": {
                    "route": {
                        "type": "choice",
                        "choice": "configuration_review",
                        "confidence": 0.8,
                        "probabilities": {
                            "dependency_investigation": 0.01,
                            "configuration_review": 0.94,
                            "source_review": 0.02,
                            "needs_review": 0.03,
                        },
                    }
                },
            },
        )

    item = finding()
    item.background = "Must not reach the guarded request"
    with TypeSafeClient(
        api_key="test-only", model="jev-latest", transport=httpx2.MockTransport(respond)
    ) as client:
        result = route_finding(client, item)
    assert result == Prediction("configuration_review", 0.94, model="jev-1.13.0")
    assert len(requests) == 1
    assert requests[0]["model"] == "jev-latest"
    assert requests[0]["state"] == filtered_state(item)
    assert set(requests[0]["questions"]["route"]["criteria"]) == set(CRITERIA)


def test_dataset_validation_and_review_provenance(tmp_path: Path) -> None:
    payload = load_cases(DATASET)[0].model_dump()
    payload["provenance"] = "reviewed"
    with pytest.raises(ValidationError, match="review_reference"):
        Case.model_validate(payload)
    payload["review_reference"] = "local-review:123"
    assert Case.model_validate(payload).review_reference == "local-review:123"
    payload["important"] = "false"
    with pytest.raises(ValidationError):
        Case.model_validate(payload)
    payload["important"] = False
    payload["finding"]["expected_route"] = "source_review"
    with pytest.raises(ValidationError):
        Case.model_validate(payload)

    path = tmp_path / "cases.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_cases(path)
    line = load_cases(DATASET)[0].model_dump_json()
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_cases(path)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--backend", "jev"],
        ["--backend", "jev", "--model", "jev-1.13.0"],
        ["--backend", "jev", "--model", "jev-1.13.0", "--allow-network"],
        ["--threshold", "nan"],
        ["--max-chars", "0"],
    ],
)
def test_cli_refuses_unsafe_or_invalid_invocation(
    arguments: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr("sys.argv", ["finding_routing", *arguments])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


def test_cli_default_is_explicitly_not_a_jev_benchmark(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["finding_routing"])
    main()
    report = json.loads(capsys.readouterr().out)
    assert report["backend"] == "demo"
    assert report["requested_model"] == "scripted-demo-not-jev"
    assert "not Jev evidence" in report["warning"]
