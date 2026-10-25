import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import FastAPI

from app.core.concurrency import SessionLocks
from app.core.config import settings
from app.retrieval.faithfulness import evaluate_faithfulness
from app.schemas.models import ChatRequest, QueryRequest, QueryRequestStream
from app.service.rag_engine import RAGEngine


def context(chunk_id="a", content="Sales were 12.", **extra):
    return {"id": chunk_id, "content": content, "file_name": "report.pdf", "page_number": 2, **extra}


def reply(payload):
    return {
        "response": json.dumps(payload, ensure_ascii=False),
        "input_tokens": 20, "output_tokens": 10, "reasoning_tokens": 2,
    }


def verdict(claim_id=1, outcome="supported", quote="Sales were 12.", rank=1):
    return {
        "claim_id": claim_id, "verdict": outcome, "reason": "Based on the source.",
        "evidence": [] if outcome == "not_enough_evidence" else [{"context_rank": rank, "quote": quote}],
    }


def judge_model(statements=None, judgments=None):
    return SimpleNamespace(complete=AsyncMock(side_effect=[
        reply({"claims": statements if statements is not None else ["Sales were 12."]}),
        reply({"claims": judgments if judgments is not None else [verdict()]}),
    ]))


def test_score_uses_all_claims_and_returns_checked_source_evidence():
    model = judge_model(
        ["Sales were 12.", "Sales were 50.", "Profit was 5."],
        [verdict(3, "not_enough_evidence"), verdict(1), verdict(2, "contradicted")],
    )
    report = asyncio.run(evaluate_faithfulness(
        "How were sales?", "Sales were 12 and 50. Profit was 5.", [context()],
        enabled=True, language_model=model,
    ))
    assert report.status == "completed"
    assert report.score == pytest.approx(1 / 3)
    assert report.claim_count == 3
    assert report.supported_claim_count == 1
    assert report.unsupported_claim_count == 2
    assert [claim.claim_id for claim in report.claims] == [1, 2, 3]
    assert report.claims[1].statement == "Sales were 50."
    evidence = report.claims[0].evidence[0]
    assert (evidence.chunk_id, evidence.file_name, evidence.page_number) == ("a", "report.pdf", 2)
    assert evidence.quote == "Sales were 12."
    assert report.judge_usage == {"input_tokens": 40, "output_tokens": 20, "reasoning_tokens": 4}
    extraction_payload = json.loads(model.complete.await_args_list[0].args[0][1]["content"])
    assert set(extraction_payload) == {"question", "answer"}
    verification_payload = json.loads(model.complete.await_args_list[1].args[0][1]["content"])
    assert set(verification_payload) == {"claims", "contexts"}
    assert len(verification_payload["claims"]) == 3


@pytest.mark.parametrize("outcome,expected", [("supported", 1), ("contradicted", 0), ("not_enough_evidence", 0)])
def test_score_extremes_are_computed_from_verdicts(outcome, expected):
    report = asyncio.run(evaluate_faithfulness(
        "sales", "Sales were 12.", [context()], enabled=True,
        language_model=judge_model(judgments=[verdict(outcome=outcome)]),
    ))
    assert report.status == "completed"
    assert report.score == expected


@pytest.mark.parametrize("judgments", [
    [],
    [verdict(), verdict()],
    [verdict(claim_id=2)],
    [verdict(claim_id=True)],
    [verdict(rank=True)],
    [verdict(rank=0)],
    [verdict(rank=2)],
    [verdict(quote="Sales were 999.")],
    [{**verdict(), "evidence": []}],
    [{**verdict(outcome="contradicted"), "evidence": []}],
    [{**verdict(), "verdict": "not_enough_evidence"}],
    [{**verdict(), "verdict": "probably supported"}],
])
def test_invalid_verdicts_never_produce_a_score(judgments):
    report = asyncio.run(evaluate_faithfulness(
        "sales", "Sales were 12.", [context()], enabled=True,
        language_model=judge_model(judgments=judgments),
    ))
    assert report.status == "unavailable"
    assert report.score is None
    assert report.claim_count is None
    assert report.claims == []
    assert report.judge_usage["input_tokens"] == 40


@pytest.mark.parametrize("extraction", [
    "not JSON", '{"claims": [""]}', '{"claims": [42]}',
    '{"claims": ["Sales were 12.", "  Sales were 12. "]}',
])
def test_invalid_extraction_stops_before_verification(extraction):
    model = SimpleNamespace(complete=AsyncMock(return_value={"response": extraction, "input_tokens": 20}))
    report = asyncio.run(evaluate_faithfulness(
        "sales", "Sales were 12.", [context()], enabled=True, language_model=model,
    ))
    assert report.status == "unavailable"
    assert report.score is None
    model.complete.assert_awaited_once()


@pytest.mark.parametrize("answer,contexts,status", [
    ("", [context()], "no_answer"),
    (" \n", [context()], "no_answer"),
    ("Sales were 12.", [], "no_contexts"),
    ("Sales were 12.", [context(content=" \n")], "no_contexts"),
])
def test_missing_answer_or_sources_does_not_get_a_perfect_score(answer, contexts, status):
    model = SimpleNamespace(complete=AsyncMock())
    report = asyncio.run(evaluate_faithfulness(
        "sales", answer, contexts, enabled=True, language_model=model,
    ))
    assert report.status == status
    assert report.score is None
    model.complete.assert_not_awaited()


def test_no_factual_claims_returns_no_score_and_skips_verification():
    model = judge_model(statements=[])
    report = asyncio.run(evaluate_faithfulness(
        "sales", "I cannot answer from these sources.", [context()], enabled=True, language_model=model,
    ))
    assert report.status == "no_claims"
    assert report.claim_count == 0
    assert report.score is None
    model.complete.assert_awaited_once()


def test_disabled_evaluation_does_not_call_model():
    model = SimpleNamespace(complete=AsyncMock())
    assert asyncio.run(evaluate_faithfulness("sales", "Sales were 12.", [context()], language_model=model)) is None
    model.complete.assert_not_awaited()


def test_persian_evidence_allows_pdf_line_wrapping_and_keeps_ocr_flag():
    statement = "فروش ۱۲ واحد بود."
    model = judge_model([statement], [verdict(quote=statement)])
    model.complete.side_effect = [
        {"response": '```json\n{"claims": ["فروش ۱۲ واحد بود."]}\n```'},
        reply({"claims": [verdict(quote=statement)]}),
    ]
    report = asyncio.run(evaluate_faithfulness(
        "فروش چقدر بود؟", statement, [context(content="فروش ۱۲\nواحد بود.", requires_ocr=True)],
        enabled=True, language_model=model,
    ))
    assert report.score == 1
    payload = json.loads(model.complete.await_args_list[1].args[0][1]["content"])
    assert payload["contexts"][0]["requires_ocr"] is True


def test_judge_cancellation_propagates():
    model = SimpleNamespace(complete=AsyncMock(side_effect=asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(evaluate_faithfulness("sales", "Sales were 12.", [context()], enabled=True, language_model=model))


def make_engine():
    engine = RAGEngine.__new__(RAGEngine)
    engine._initialized = True
    engine.session_locks = SessionLocks()
    engine.history_manager = SimpleNamespace(
        embed_query=AsyncMock(return_value=[1.0, 0.0]),
        get_formatted_history_for_context=AsyncMock(return_value=[]),
        has_uploaded_documents=AsyncMock(return_value=True),
        search_documents=AsyncMock(return_value=[context(), context("b", "Profit was 5.")]),
        add_conversation=AsyncMock(),
    )
    engine.language_model = judge_model()
    return engine


def test_engine_checks_only_selected_sources_after_saving_and_keeps_answer_usage():
    engine = make_engine()
    answer = {"response": "Sales were 12.", "input_tokens": 10, "output_tokens": 5, "reasoning_tokens": 0}
    responses = [answer, reply({"claims": ["Sales were 12."]}), reply({"claims": [verdict()]})]

    async def complete(*args, **kwargs):
        if len(responses) < 3:
            engine.history_manager.add_conversation.assert_awaited_once()
        return responses.pop(0)

    engine.language_model.complete.side_effect = complete
    with patch.object(settings, "max_context_chars", len("Sales were 12.")):
        result = asyncio.run(engine.answer_question("sales", "s1", evaluation_mode="off", evaluate_faithfulness=True))
    assert result["metadata"]["context_evaluation"] is None
    assert result["metadata"]["faithfulness_evaluation"]["score"] == 1
    assert result["response"] == answer["response"]
    assert result["input_tokens"] == 10
    payload = json.loads(engine.language_model.complete.await_args_list[2].args[0][1]["content"])
    assert len(payload["contexts"]) == 1
    assert payload["contexts"][0]["content"] == "Sales were 12."
    engine.history_manager.add_conversation.assert_awaited_once_with("sales", "Sales were 12.", "s1")


def test_faithfulness_failure_still_returns_and_saves_the_answer():
    engine = make_engine()
    engine.language_model.complete.side_effect = [
        {"response": "Sales were 12.", "input_tokens": 10, "output_tokens": 5, "reasoning_tokens": 0},
        RuntimeError("judge offline"),
    ]
    result = asyncio.run(engine.answer_question("sales", "s1", evaluate_faithfulness=True))
    assert result["response"] == "Sales were 12."
    assert result["metadata"]["faithfulness_evaluation"]["status"] == "unavailable"
    engine.history_manager.add_conversation.assert_awaited_once()


def test_default_engine_request_has_no_additional_judge_calls():
    engine = make_engine()
    engine.language_model.complete.side_effect = None
    engine.language_model.complete.return_value = {
        "response": "Sales were 12.", "input_tokens": 10, "output_tokens": 5, "reasoning_tokens": 0,
    }
    result = asyncio.run(engine.answer_question("sales", "s1"))
    assert result["metadata"]["faithfulness_evaluation"] is None
    engine.language_model.complete.assert_awaited_once()


async def answer_stream(*args, **kwargs):
    yield "Sales were "
    yield "12."
    yield {"input_tokens": 10, "output_tokens": 5, "reasoning_tokens": 0}


def test_stream_evaluates_full_answer_and_returns_score_only_in_terminal_event():
    engine = make_engine()
    engine.language_model.stream = answer_stream

    async def run():
        return [event async for event in engine.answer_question_stream(
            "sales", "s1", evaluation_mode="off", evaluate_faithfulness=True,
        )]

    events = asyncio.run(run())
    assert events[:2] == ["Sales were ", "12."]
    done = events[-1]
    assert done["type"] == "done"
    assert done["faithfulness_evaluation"]["score"] == 1
    assert done["usage"]["prompt_tokens"] == 10
    payload = json.loads(engine.language_model.complete.await_args_list[0].args[0][1]["content"])
    assert payload["answer"] == "Sales were 12."
    engine.history_manager.add_conversation.assert_awaited_once_with("sales", "Sales were 12.", "s1")


def test_incomplete_stream_is_saved_as_partial_without_judging():
    engine = make_engine()

    async def broken_stream(*args, **kwargs):
        yield "Sales were "
        raise RuntimeError("connection lost")

    engine.language_model.stream = broken_stream

    async def run():
        return [event async for event in engine.answer_question_stream("sales", "s1", evaluate_faithfulness=True)]

    with pytest.raises(RuntimeError, match="connection lost"):
        asyncio.run(run())
    engine.language_model.complete.assert_not_awaited()
    engine.history_manager.add_conversation.assert_awaited_once_with("sales", "Sales were ", "s1", partial=True)


@pytest.mark.parametrize("request_type", [QueryRequest, QueryRequestStream, ChatRequest])
def test_faithfulness_is_opt_in_on_all_request_schemas(request_type):
    query_field = "message" if request_type is ChatRequest else "question"
    base = {query_field: "sales", "session_id": "s1"}
    assert request_type(**base).evaluate_faithfulness is False
    assert request_type(**base, evaluate_faithfulness=True, evaluation_mode="off").evaluate_faithfulness is True


@pytest.mark.parametrize("path,body", [
    ("/chat", {"question": "sales", "session_id": "s1"}),
    ("/query", {"message": "sales", "session_id": "s1"}),
    ("/chat/stream", {"question": "sales", "session_id": "s1"}),
])
def test_api_supports_faithfulness_and_separate_judge_usage(path, body):
    from app.api.routes import get_chatbot, router

    engine = make_engine()
    if path.endswith("/stream"):
        engine.language_model.stream = answer_stream
    else:
        engine.language_model.complete.side_effect = [
            {"response": "Sales were 12.", "input_tokens": 10, "output_tokens": 5, "reasoning_tokens": 0},
            reply({"claims": ["Sales were 12."]}), reply({"claims": [verdict()]}),
        ]
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_chatbot] = lambda: engine

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.post(path, json={**body, "evaluation_mode": "off", "evaluate_faithfulness": True})

    response = asyncio.run(run())
    assert response.status_code == 200
    if path.endswith("/stream"):
        events = [line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ")]
        data = next(json.loads(event) for event in events if event.startswith('{"type": "done"'))
        assert events[-1] == "[DONE]"
        assert data["usage"]["prompt_tokens"] == 10
    else:
        data = response.json()
        assert data["input_tokens"] == 10
    assert data["context_evaluation"] is None
    assert data["faithfulness_evaluation"]["score"] == 1
    assert data["faithfulness_evaluation"]["judge_usage"]["input_tokens"] == 40
