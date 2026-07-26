"""DeepEval model adapter for the university OpenAI-compatible gateway."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Any

from deepeval.models import DeepEvalBaseLLM
from langchain_openai import ChatOpenAI

from simple_agent.config import Settings, get_settings


class GatewayEvaluationModel(DeepEvalBaseLLM):
    """Use the configured gateway for DeepEval generation and judging."""

    def __init__(
        self,
        settings: Settings | None = None,
        request_delay_seconds: float | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.request_delay_seconds = (
            float(os.getenv("DEEPEVAL_REQUEST_DELAY_SECONDS", "1"))
            if request_delay_seconds is None
            else request_delay_seconds
        )
        self.chat_model = ChatOpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url,
            model=self.settings.openai_model,
            reasoning_effort=self.settings.openai_reasoning_effort,
            temperature=self.settings.openai_temperature,
            timeout=self.settings.openai_timeout_seconds,
            max_retries=self.settings.openai_max_retries,
        )
        super().__init__(model=self.settings.openai_model)

    def load_model(self) -> ChatOpenAI:
        return self.chat_model

    @staticmethod
    def _content_to_text(response: Any) -> str:
        content = response.content
        if isinstance(content, str):
            return content
        return str(content)

    @staticmethod
    def _parse_structured(text: str, schema: Any) -> Any:
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("The gateway returned an empty response")
        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
        if fenced:
            cleaned = fenced.group(1)
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            candidates = [
                index for index in (cleaned.find("{"), cleaned.find("[")) if index >= 0
            ]
            if not candidates:
                raise ValueError("The gateway response did not contain JSON")
            start = min(candidates)
            end = max(cleaned.rfind("}"), cleaned.rfind("]")) + 1
            payload = json.loads(cleaned[start:end])
        return schema.model_validate(payload)

    @staticmethod
    def _json_retry_prompt(prompt: str) -> str:
        return (
            f"{prompt}\n\nReturn only valid JSON matching the requested schema. "
            "Do not include markdown, explanations, or chain-of-thought."
        )

    def _invoke(self, prompt: str) -> Any:
        if self.request_delay_seconds > 0:
            time.sleep(self.request_delay_seconds)
        return self.model.invoke(prompt)

    async def _ainvoke(self, prompt: str) -> Any:
        if self.request_delay_seconds > 0:
            await asyncio.sleep(self.request_delay_seconds)
        return await self.model.ainvoke(prompt)

    def generate(self, prompt: str, schema: Any = None, **kwargs: Any) -> Any:
        if schema is not None:
            response = self._invoke(prompt)
            try:
                return self._parse_structured(self._content_to_text(response), schema)
            except (ValueError, json.JSONDecodeError):
                retry = self._invoke(self._json_retry_prompt(prompt))
                return self._parse_structured(self._content_to_text(retry), schema)
        response = self._invoke(prompt)
        return self._content_to_text(response)

    async def a_generate(self, prompt: str, schema: Any = None, **kwargs: Any) -> Any:
        if schema is not None:
            response = await self._ainvoke(prompt)
            try:
                return self._parse_structured(self._content_to_text(response), schema)
            except (ValueError, json.JSONDecodeError):
                retry = await self._ainvoke(self._json_retry_prompt(prompt))
                return self._parse_structured(self._content_to_text(retry), schema)
        response = await self._ainvoke(prompt)
        return self._content_to_text(response)

    def get_model_name(self) -> str:
        return self.settings.openai_model
