import asyncio
import json
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import FastAPI
from pydantic import ValidationError

from app.core.concurrency import SessionLocks
from app.core.config import settings
from app.retrieval.evaluation import context_diagnostics, evaluate_contexts, ranking_metrics
from app.schemas.models import ChatRequest, QueryRequest, QueryRequestStream
from app.service.rag_engine import RAGEngine


def context(chunk_id, content, **extra):
    return {"id": chunk_id, "content": content, "file_name": "report.pdf", "page_number": 2, **extra}


def test_diagnostics_normalize_persian_and_digits():
    contexts = [context("one", "مقدار کالا 12")]
    report = context_diagnostics("مقدار كالا ۱۲ چیست؟", contexts, contexts, 5)
    assert report.query_term_coverage == 1
    assert report.contexts[0].query_term_coverage == 1
    assert report.ranking_metrics is None
    assert report.contexts[0].relevance_score is None


def test_reports_only_prompt_contexts_but_counts_dropped_and_duplicates():
    retrieved = [context("a", "sales ۱۲"), context("b", "sales 12"), context("c", "tax")]
    report = context_diagnostics("sales tax", retrieved, retrieved[:1], 3)
    assert (report.retrieved_count, report.evaluated_count, report.dropped_count) == (3, 1, 2)
    assert report.duplicate_count == 1
    assert report.query_term_coverage == 0.5
    assert [item.chunk_id for item in report.contexts] == ["a"]


def test_search_scores_are_not_used_as_relevance_probabilities():
    contexts = [context("one", "unrelated content", score=0.04, rerank_score=-6.5)]
    report = context_diagnostics("sales", contexts, contexts, 5)
    assert report.contexts[0].retrieval_score == 0.04
    assert report.contexts[0].rerank_score == -6.5
    assert report.contexts[0].query_term_coverage == 0
    assert report.judged_context_precision is None
    assert report.mean_relevance_score is None


def test_unavailable_scores_and_ocr_are_explicit():
    contexts = [context("one", "Table I", score=float("nan"), rerank_score="bad", requires_ocr=True)]
    report = context_diagnostics("table", contexts, contexts, 1)
    assert report.contexts[0].retrieval_score is None
    assert report.contexts[0].rerank_score is None
    assert report.contexts[0].requires_ocr
    assert report.ocr_required_count == 1


def test_empty_query_terms_are_undefined_instead_of_perfect_coverage():
    contexts = [context("one", "some text")]
    report = context_diagnostics("what is the", contexts, contexts, 1)
    assert report.query_term_coverage is None
    assert report.contexts[0].query_term_coverage is None


def test_reference_metrics_account_for_missing_results_and_rank():
    contexts = [context("wrong", "x"), context("a", "x"), context("b", "x")]
    metrics = ranking_metrics(contexts, ["a", "b", "c"], 4)
    assert metrics.precision_at_k == 0.5
    assert metrics.recall_at_k == pytest.approx(2 / 3)
    assert metrics.reciprocal_rank == 0.5
    assert metrics.ndcg_at_k == pytest.approx((1 / math.log2(3) + 0.5) / (1 + 1 / math.log2(3) + 0.5))


def test_duplicates_and_unknown_ids_do_not_receive_extra_relevance_credit():
    contexts = [context("a", "x"), context("a", "x"), context(None, "x"), context("b", "x")]
    metrics = ranking_metrics(contexts, ["a", "a", "b"], 3)
    assert metrics.relevant_count == 2
    assert metrics.relevant_retrieved == 1
    assert metrics.precision_at_k == pytest.approx(1 / 3)
    assert metrics.recall_at_k == 0.5
    assert metrics.reciprocal_rank == 1


def test_empty_retrieval_has_zero_reference_metrics_and_skips_judge():
    model = SimpleNamespace(complete=AsyncMock())
    report = asyncio.run(evaluate_contexts("sales", [], [], 5, mode="llm", relevant_chunk_ids=["a"], language_model=model))
    assert report.status == "no_contexts"
    assert report.ranking_metrics.precision_at_k == 0
    assert report.ranking_metrics.recall_at_k == 0
    assert report.ranking_metrics.ndcg_at_k == 0
    assert report.judged_context_precision is None
    model.complete.assert_not_awaited()


def test_judge_scores_are_attached_by_rank_and_usage_is_separate():
    contexts = [context("a", "Sales were 12."), context("b", "Weather was cold.")]
    response = {"contexts": [{"rank": 2, "score": 0, "reason": "Unrelated."}, {"rank": 1, "score": 3, "reason": "Contains sales."}]}
    model = SimpleNamespace(complete=AsyncMock(return_value={
        "response": "```json\n" + json.dumps(response) + "\n```",
        "input_tokens": 100, "output_tokens": 50, "reasoning_tokens": 4,
    }))
    report = asyncio.run(evaluate_contexts("sales", contexts, contexts, 2, mode="llm", language_model=model))
    assert report.status == "completed"
    assert [item.relevance_score for item in report.contexts] == [3, 0]
    assert report.mean_relevance_score == 1.5
    assert report.judged_context_precision == 0.5
    assert report.ranking_metrics is None
    assert report.judge_usage == {"input_tokens": 100, "output_tokens": 50, "reasoning_tokens": 4}
    assert "relevance_score" not in contexts[0]


@pytest.mark.parametrize("response", [
    "not JSON",
    '{"contexts": []}',
    '{"contexts": [{"rank": 2, "score": 1, "reason": "wrong rank"}]}',
    '{"contexts": [{"rank": 1, "score": 1.5, "reason": "fractional"}]}',
    '{"contexts": [{"rank": 1, "score": true, "reason": "boolean"}]}',
    '{"contexts": [{"rank": 1, "score": 4, "reason": "out of range"}]}',
    '{"contexts": [{"rank": 1, "score": 3, "reason": "a"}, {"rank": 1, "score": 0, "reason": "b"}]}',
])
def test_bad_judge_responses_never_become_successful_zero_scores(response):
    contexts = [context("a", "sales")]
    model = SimpleNamespace(complete=AsyncMock(return_value={"response": response, "input_tokens": 10}))
    report = asyncio.run(evaluate_contexts("sales", contexts, contexts, 1, mode="llm", language_model=model))
    assert report.status == "unavailable"
    assert report.contexts[0].relevance_score is None
    assert report.query_term_coverage == 1
    assert report.judge_usage["input_tokens"] == 10


def test_judge_failure_retains_reference_metrics_and_diagnostics():
    contexts = [context("a", "sales")]
    model = SimpleNamespace(complete=AsyncMock(side_effect=RuntimeError("offline")))
    report = asyncio.run(evaluate_contexts("sales", contexts, contexts, 1, mode="llm", relevant_chunk_ids=["a"], language_model=model))
    assert report.status == "unavailable"
    assert report.ranking_metrics.recall_at_k == 1
    assert report.query_term_coverage == 1


def test_off_and_diagnostics_do_not_call_the_model():
    contexts = [context("a", "sales")]
    model = SimpleNamespace(complete=AsyncMock())
    assert asyncio.run(evaluate_contexts("sales", contexts, contexts, 1, mode="off", language_model=model)) is None
    assert asyncio.run(evaluate_contexts("sales", contexts, contexts, 1, language_model=model)).status == "completed"
    model.complete.assert_not_awaited()


@pytest.mark.parametrize("request_type", [QueryRequest, QueryRequestStream, ChatRequest])
def test_evaluation_request_validation(request_type):
    query_field = "message" if request_type is ChatRequest else "question"
    base = {query_field: "sales", "session_id": "s1"}
    assert request_type(**base).evaluation_mode == "diagnostics"
    for extra in ({"evaluation_mode": "invented"}, {"relevant_chunk_ids": []}, {"relevant_chunk_ids": [""]}):
        with pytest.raises(ValidationError):
            request_type(**base, **extra)


def make_engine(retrieved, has_documents=True):
    engine = RAGEngine.__new__(RAGEngine)
    engine._initialized = True
    engine.session_locks = SessionLocks()
    engine.history_manager = SimpleNamespace(
        embed_query=AsyncMock(return_value=[1.0, 0.0]),
        get_formatted_history_for_context=AsyncMock(return_value=[]),
        has_uploaded_documents=AsyncMock(return_value=has_documents),
        search_documents=AsyncMock(return_value=retrieved),
        add_conversation=AsyncMock(),
    )
    engine.language_model = SimpleNamespace(complete=AsyncMock(return_value={
        "response": "Sales were 12.", "input_tokens": 10, "output_tokens": 5, "reasoning_tokens": 0,
    }))
    return engine


def test_engine_evaluates_after_budget_selection_without_extra_generation():
    retrieved = [context("a", "sales " * 10), context("b", "sales " * 10), context("c", "tax " * 10)]
    engine = make_engine(retrieved)
    with patch.object(settings, "max_context_chars", 60):
        result = asyncio.run(engine.answer_question("sales tax", "s1", top_k=3, relevant_chunk_ids=["a", "c"]))
    evaluation = result["metadata"]["context_evaluation"]
    assert evaluation["evaluated_count"] == 1
    assert evaluation["duplicate_count"] == 1
    assert evaluation["ranking_metrics"]["recall_at_k"] == 0.5
    assert result["metadata"]["context_used"] == retrieved[:1]
    assert engine.language_model.complete.await_count == 1
    engine.history_manager.embed_query.assert_awaited_once()
    assert engine.history_manager.search_documents.await_args.kwargs["query_vector"] == [1.0, 0.0]


def test_no_documents_returns_explicit_no_contexts_evaluation():
    engine = make_engine([], has_documents=False)
    result = asyncio.run(engine.answer_question("sales", "s1", evaluation_mode="llm"))
    assert result["metadata"]["context_evaluation"]["status"] == "no_contexts"
    engine.history_manager.search_documents.assert_not_awaited()
    assert engine.language_model.complete.await_count == 1


def test_judge_failure_does_not_prevent_answer_generation_or_persistence():
    engine = make_engine([context("a", "sales")])
    engine.language_model.complete.side_effect = [RuntimeError("judge unavailable"), {
        "response": "Sales were 12.", "input_tokens": 10, "output_tokens": 5, "reasoning_tokens": 0,
    }]
    result = asyncio.run(engine.answer_question("sales", "s1", evaluation_mode="llm"))
    assert result["response"] == "Sales were 12."
    assert result["metadata"]["context_evaluation"]["status"] == "unavailable"
    assert result["input_tokens"] == 10
    engine.history_manager.add_conversation.assert_awaited_once_with("sales", "Sales were 12.", "s1")


def test_stream_terminal_event_contains_evaluation():
    engine = make_engine([context("a", "sales")])

    async def stream(*args, **kwargs):
        yield "Sales were 12."
        yield {"input_tokens": 10, "output_tokens": 5, "reasoning_tokens": 0}

    engine.language_model.stream = stream

    async def run():
        return [event async for event in engine.answer_question_stream("sales", "s1", relevant_chunk_ids=["a"])]

    events = asyncio.run(run())
    assert events[-1]["type"] == "done"
    assert events[-1]["context_evaluation"]["ranking_metrics"]["recall_at_k"] == 1
    assert events[-1]["context_evaluation"]["contexts"][0]["chunk_id"] == "a"


@pytest.mark.parametrize("path,body", [
    ("/chat", {"question": "sales", "session_id": "s1"}),
    ("/query", {"message": "sales", "session_id": "s1"}),
])
def test_api_exposes_evaluation(path, body):
    from app.api.router import get_chatbot, router

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_chatbot] = lambda: make_engine([context("a", "sales")])

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.post(path, json={**body, "relevant_chunk_ids": ["a"]})

    response = asyncio.run(run())
    assert response.status_code == 200
    evaluation = response.json()["context_evaluation"]
    assert evaluation["evaluated_count"] == 1
    assert evaluation["ranking_metrics"]["recall_at_k"] == 1


def test_stream_api_forwards_evaluation_options_and_serializes_terminal_result():
    from app.api.router import get_chatbot, router

    engine = make_engine([context("a", "sales")])
    engine.language_model.complete.return_value = {"response": json.dumps({
        "contexts": [{"rank": 1, "score": 3, "reason": "Contains sales."}],
    }), "input_tokens": 12, "output_tokens": 8}

    async def stream(*args, **kwargs):
        yield "Sales were 12."
        yield {"input_tokens": 10, "output_tokens": 5, "reasoning_tokens": 0}

    engine.language_model.stream = stream
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_chatbot] = lambda: engine

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/chat/stream", json={
                "question": "sales", "session_id": "s1", "evaluation_mode": "llm",
                "relevant_chunk_ids": ["a"],
            })

    response = asyncio.run(run())
    assert response.status_code == 200
    events = [line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ")]
    done = next(json.loads(event) for event in events if event.startswith('{"type": "done"'))
    assert done["context_evaluation"]["contexts"][0]["relevance_score"] == 3
    assert done["context_evaluation"]["ranking_metrics"]["recall_at_k"] == 1
    assert done["context_evaluation"]["judge_usage"]["input_tokens"] == 12
    assert done["usage"]["prompt_tokens"] == 10
    assert events[-1] == "[DONE]"
