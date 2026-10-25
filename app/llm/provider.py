import asyncio
from collections.abc import AsyncIterator
from typing import Any, Optional

from openai import AsyncOpenAI

from app.core.config import settings


class LanguageModel:
    def __init__(self) -> None:
        self.client: AsyncOpenAI | None = None
        self.tokenizer = None
        self.model = None
        self._initialized = False
        self._initialize_lock = asyncio.Lock()
        self._generation_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        async with self._initialize_lock:
            if self._initialized:
                return
            if settings.llm_provider == "openai":
                if not settings.openai_api_key:
                    raise ValueError("OPENAI_API_KEY is required for the OpenAI provider")
                self.client = AsyncOpenAI(
                    api_key=settings.openai_api_key,
                    timeout=settings.openai_timeout,
                    max_retries=settings.openai_max_retries,
                )
            elif settings.llm_provider == "qwen":
                await asyncio.to_thread(self._load_local_model)
            else:
                raise ValueError("LLM_PROVIDER must be 'openai' or 'qwen'")
            self._initialized = True

    async def close(self) -> None:
        try:
            if self.client is not None:
                await self.client.close()
        finally:
            self.client = None
            self.tokenizer = None
            self.model = None
            self._initialized = False

    async def __aenter__(self) -> "LanguageModel":
        await self.initialize()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    def __repr__(self) -> str:
        return (
            f"LanguageModel("
            f"provider={settings.llm_provider!r}, "
            f"initialized={self._initialized})"
        )

    # ------------------------------------------------------------------
    # Local model loader (run in a thread)
    # ------------------------------------------------------------------

    def _load_local_model(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(settings.qwen_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.qwen_model_name, device_map=settings.qwen_device
        )
        self.model.eval()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "LanguageModel is not initialized; call initialize() first"
            )

    @staticmethod
    def usage(usage: Any) -> dict[str, int]:
        details = getattr(usage, "completion_tokens_details", None)
        return {
            "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "reasoning_tokens": getattr(details, "reasoning_tokens", 0) or 0,
        }

    def _generate_local(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> dict[str, Any]:
        import torch

        self._require_initialized()
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        effective_temperature = (
            settings.qwen_temperature if temperature is None else temperature
        )
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=settings.qwen_max_tokens if max_tokens is None else max_tokens,
                temperature=effective_temperature,
                do_sample=effective_temperature > 0,
            )
        input_tokens = inputs["input_ids"].shape[-1]
        generated = outputs[0][input_tokens:]
        return {
            "response": self.tokenizer.decode(generated, skip_special_tokens=True),
            "input_tokens": input_tokens,
            "output_tokens": len(generated),
            "reasoning_tokens": 0,
        }

    # ------------------------------------------------------------------
    # Completions
    # ------------------------------------------------------------------

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: str = None
    ) -> dict[str, Any]:
        self._require_initialized()
        if settings.llm_provider == "qwen":
            async with self._generation_lock:
                return await asyncio.to_thread(
                    self._generate_local, messages, temperature, max_tokens
                )

        model = model if model else settings.openai_model
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=(
                settings.openai_temperature if temperature is None else temperature
            ),
            max_completion_tokens=(
                settings.openai_max_tokens if max_tokens is None else max_tokens
            ),
        )
        return {
            "response": response.choices[0].message.content or "",
            **self.usage(response.usage),
        }

    async def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str | dict[str, int]]:
        self._require_initialized()
        if settings.llm_provider == "qwen":
            result = await self.complete(
                messages, temperature=temperature, max_tokens=max_tokens
            )
            yield result["response"]
            yield {k: result[k] for k in ("input_tokens", "output_tokens", "reasoning_tokens")}
            return
        stream = await self.client.chat.completions.create(
            model=settings.openai_model,
            messages=messages,
            temperature=(
                settings.openai_temperature if temperature is None else temperature
            ),
            max_completion_tokens=(
                settings.openai_max_tokens if max_tokens is None else max_tokens
            ),
            stream=True,
            stream_options={"include_usage": True},
        )
        async with stream:
            async for chunk in stream:
                if chunk.usage is not None:
                    yield self.usage(chunk.usage)
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
