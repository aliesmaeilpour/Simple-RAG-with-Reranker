import pytest
from pydantic import ValidationError

from app.schemas.models import ChatRequest, QueryRequest, QueryRequestStream


@pytest.mark.parametrize("model", [QueryRequest, QueryRequestStream])
def test_session_id_must_match_safe_pattern(model):
    assert model(question="سلام", session_id="example-1").session_id == "example-1"
    with pytest.raises(ValidationError):
        model(question="سلام", session_id="bad session id!")
    with pytest.raises(ValidationError):
        model(question="سلام", session_id="")


@pytest.mark.parametrize("model", [QueryRequest, QueryRequestStream, ChatRequest])
def test_question_length_is_capped(model):
    field = "message" if model is ChatRequest else "question"
    with pytest.raises(ValidationError):
        model(**{field: "x" * 100_000, "session_id": "s1"})


def test_temperature_and_max_tokens_bounds():
    with pytest.raises(ValidationError):
        QueryRequest(question="q", session_id="s1", temperature=5)
    with pytest.raises(ValidationError):
        QueryRequest(question="q", session_id="s1", max_tokens=0)
    request = QueryRequest(question="q", session_id="s1", temperature=0.2)
    assert request.temperature == 0.2


def test_top_k_bounds():
    with pytest.raises(ValidationError):
        QueryRequest(question="q", session_id="s1", top_k=0)
    with pytest.raises(ValidationError):
        QueryRequest(question="q", session_id="s1", top_k=101)
