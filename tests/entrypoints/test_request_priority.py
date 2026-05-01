# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.generative_scoring.serving import (
    GenerativeScoringRequest,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.pooling.embed.protocol import (
    CohereEmbedRequest,
    EmbeddingCompletionRequest,
)
from vllm.entrypoints.serve.disagg.protocol import GenerateRequest
from vllm.sampling_params import SamplingParams

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    ("request_cls", "kwargs"),
    [
        (
            CompletionRequest,
            {
                "model": "test-model",
                "prompt": "hello",
            },
        ),
        (
            ChatCompletionRequest,
            {
                "model": "test-model",
                "messages": [],
            },
        ),
        (
            ResponsesRequest,
            {
                "model": "test-model",
                "input": "hello",
            },
        ),
        (
            GenerativeScoringRequest,
            {
                "model": "test-model",
                "query": "Is this the capital?",
                "items": ["Paris", "London"],
                "label_token_ids": [1, 2],
            },
        ),
        (
            EmbeddingCompletionRequest,
            {
                "model": "test-model",
                "input": "hello",
            },
        ),
        (
            CohereEmbedRequest,
            {
                "model": "test-model",
                "texts": ["hello"],
            },
        ),
        (
            GenerateRequest,
            {
                "token_ids": [1, 2, 3],
                "sampling_params": SamplingParams(max_tokens=1),
            },
        ),
    ],
)
def test_request_priority_accepts_floats(request_cls, kwargs):
    request = request_cls(priority=0.5, **kwargs)

    assert request.priority == pytest.approx(0.5)
