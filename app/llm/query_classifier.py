import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any
from openai import AsyncOpenAI

from app.llm.provider import LanguageModel

logger = logging.getLogger(__name__)


@dataclass
class ParsedQuery:
    """Structured query ready for RAG retrieval."""
    semantic_query: str
    filters: dict[str, Any] = field(default_factory=dict)
    original_query: str = ""
    # Hints that don't map to DB columns but affect retrieval strategy
    prefer_tables: bool = False       # user clearly wants tabular data
    prefer_text: bool = False         # user clearly wants prose
    is_location_anchored: bool = False  # has page/sheet/section filter


_PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

_PAGE_PATTERNS = [
    r"\bpage\s+(\d+)\b",
    r"\bon\s+page\s+(\d+)\b",
    r"\bصفحه[‌\s]+(\d+|[۰-۹]+)\b",
]
_PAGE_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "یک": 1, "دو": 2, "سه": 3, "چهار": 4, "پنج": 5,
    "شش": 6, "هفت": 7, "هشت": 8, "نه": 9, "ده": 10,
}
_TABLE_SIGNALS = re.compile(
    r"\b(table|جدول|spreadsheet|grid|tabular|chart)\b", re.IGNORECASE
)
_TEXT_SIGNALS = re.compile(
    r"\b(text|paragraph|prose|explain|describe|summary|متن|پاراگراف)\b",
    re.IGNORECASE,
)


def _strip(query: str, pattern: str) -> str:
    cleaned = re.sub(pattern, "", query, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


class QueryClassifier:
    """
    Decompose a natural-language query into a semantic query + metadata filters.

    Supports:
      - page_number  (PDF / DOCX)
      - sheet_name   (XLSX)
      - hierarchy_path fuzzy hint (PDF / DOCX sections)
      - content_type ("table" | "text")
      - file_name    (multi-document workspaces)

    Strategy: rule-based fast path → LLM fallback with a single unified prompt.
    """

    # One system prompt that covers all doc types and all filter types.
    _SYSTEM = """You are a query parser for a document search system that indexes PDF, DOCX, and XLSX files.

Extract structured metadata filters from the user's query and return valid JSON.

Available filters — only include a key when the user clearly refers to it:
  page_number   : integer — for queries referencing a specific page (PDF / DOCX)
  sheet_name    : string  — for queries referencing a named sheet/tab (XLSX)
  section_hint  : string  — the section, chapter, or heading the user mentions (PDF / DOCX)
  content_type  : "table" | "text"  — when the user explicitly wants only tables or only prose
  file_name     : string  — when the user refers to a specific document by name or partial name

Also return:
  semantic_query : the core information need with location references removed
  prefer_tables  : true when the user clearly wants tabular / structured data
  prefer_text    : true when the user clearly wants prose / paragraphs

Rules:
- Convert ordinal words to integers: "two" → 2, "دو" → 2, "second" → 2
- Persian digits (۱٢٣) → Arabic digits
- If no filter applies, return an empty filters object
- section_hint should be the raw section name as the user stated it, not a paraphrase

Respond ONLY with JSON. Example:
Input: "در جدول صفحه سه چه اطلاعاتی هست؟"
Output:
{
  "semantic_query": "اطلاعات",
  "filters": { "page_number": 3, "content_type": "table" },
  "prefer_tables": true,
  "prefer_text": false
}"""

    def __init__(self, model: str = "gpt-4.1-nano") -> None:
        self.client = LanguageModel()
        self.model = model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def parse(
        self,
        raw_query: str,
        available_sheets: list[str] | None = None,
        available_files: list[str] | None = None,
    ) -> ParsedQuery:
        """
        Parse a user query. Optionally pass known sheet names and file names
        so the rule-based path can match them without an LLM call.
        """
        result = self._rule_based(raw_query, available_sheets, available_files)
        if result is not None:
            print("not LLM Classifier")
            return result
        print("LLM classifier")
        return await self._llm_parse(raw_query, available_sheets, available_files)

    # ------------------------------------------------------------------
    # Rule-based fast path
    # ------------------------------------------------------------------

    def _rule_based(
        self,
        query: str,
        available_sheets: list[str] | None,
        available_files: list[str] | None,
    ) -> ParsedQuery | None:
        """
        Handle obvious single-filter patterns without an LLM call.
        Returns None when the query is too complex for rules.
        """
        filters: dict[str, Any] = {}
        semantic = query
        matched = False

        # --- page number (numeric) ---
        for pat in _PAGE_PATTERNS:
            m = re.search(pat, query, re.IGNORECASE)
            if m:
                page_str = m.group(1).translate(_PERSIAN_DIGITS)
                filters["page_number"] = int(page_str)
                semantic = _strip(semantic, pat)
                matched = True
                break

        # --- page number (word form) ---
        if not matched:
            for word, num in _PAGE_WORDS.items():
                pat = rf"\b(?:page|صفحه)[‌\s]+{re.escape(word)}\b"
                if re.search(pat, query, re.IGNORECASE):
                    filters["page_number"] = num
                    semantic = _strip(semantic, pat)
                    matched = True
                    break

        # --- sheet name exact / case-insensitive match ---
        if available_sheets and "page_number" not in filters:
            q_lower = query.lower()
            for sheet in available_sheets:
                pat = rf"\b{re.escape(sheet.lower())}\b"
                if re.search(pat, q_lower):
                    filters["sheet_name"] = sheet
                    semantic = re.sub(
                        rf"\b{re.escape(sheet)}\b", "", semantic, flags=re.IGNORECASE
                    )
                    semantic = re.sub(r"\s{2,}", " ", semantic).strip()
                    matched = True
                    break

        # --- file name ---
        if available_files:
            q_lower = query.lower()
            for fname in available_files:
                stem = re.escape(fname.lower().rsplit(".", 1)[0])
                if re.search(rf"\b{stem}\b", q_lower):
                    filters["file_name"] = fname
                    semantic = re.sub(stem, "", semantic, flags=re.IGNORECASE)
                    semantic = re.sub(r"\s{2,}", " ", semantic).strip()
                    matched = True
                    break

        if not matched:
            return None  # hand off to LLM

        # content-type hint from simple signals (even in rule path)
        prefer_tables = bool(_TABLE_SIGNALS.search(query))
        prefer_text = bool(_TEXT_SIGNALS.search(query)) and not prefer_tables
        if prefer_tables and "content_type" not in filters:
            filters["content_type"] = "table"

        return ParsedQuery(
            semantic_query=semantic or query,
            filters=filters,
            original_query=query,
            prefer_tables=prefer_tables,
            prefer_text=prefer_text,
            is_location_anchored=bool(filters),
        )

    # ------------------------------------------------------------------
    # LLM path
    # ------------------------------------------------------------------

    async def _llm_parse(
        self,
        raw_query: str,
        available_sheets: list[str] | None,
        available_files: list[str] | None,
    ) -> ParsedQuery:
        context_lines: list[str] = []
        if available_sheets:
            context_lines.append(f"Available sheets: {', '.join(available_sheets)}")
        if available_files:
            context_lines.append(f"Available files: {', '.join(available_files)}")

        user_content = raw_query
        if context_lines:
            user_content = "\n".join(context_lines) + "\n\nQuery: " + raw_query

        try:
            response = await self.client.complete(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                temperature=0
            )
            raw = json.loads(response.choices[0].message.content)
        except Exception as exc:
            logger.warning("QueryClassifier LLM call failed: %s", exc)
            return ParsedQuery(
                semantic_query=raw_query,
                filters={},
                original_query=raw_query,
            )

        filters: dict[str, Any] = {}
        raw = raw.get("response", {})
        llm_filters = raw.get("filters", {})

        if "page_number" in llm_filters:
            try:
                filters["page_number"] = int(str(llm_filters["page_number"]).translate(_PERSIAN_DIGITS))
            except (ValueError, TypeError):
                pass

        if "sheet_name" in llm_filters and llm_filters["sheet_name"]:
            sheet = llm_filters["sheet_name"]
            # Validate / normalize against known sheets
            if available_sheets:
                sheet_lower = sheet.lower()
                matched = next(
                    (s for s in available_sheets if s.lower() == sheet_lower)
                    or (s for s in available_sheets if sheet_lower in s.lower()),
                    None,
                )
                if matched:
                    filters["sheet_name"] = matched
            else:
                filters["sheet_name"] = sheet

        if "section_hint" in llm_filters and llm_filters["section_hint"]:
            # Not a hard DB filter — callers use this for hierarchy_path ILIKE
            filters["section_hint"] = llm_filters["section_hint"]

        if llm_filters.get("content_type") in ("table", "text"):
            filters["content_type"] = llm_filters["content_type"]

        if "file_name" in llm_filters and llm_filters["file_name"]:
            filters["file_name"] = llm_filters["file_name"]

        prefer_tables = bool(raw.get("prefer_tables"))
        prefer_text = bool(raw.get("prefer_text")) and not prefer_tables

        return ParsedQuery(
            semantic_query=raw.get("semantic_query", raw_query).strip() or raw_query,
            filters=filters,
            original_query=raw_query,
            prefer_tables=prefer_tables,
            prefer_text=prefer_text,
            is_location_anchored=bool(filters),
        )