# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.request_queue import PriorityRequestQueue
from vllm.v1.request import Request

pytestmark = pytest.mark.skip_global_cleanup


def _make_request(request_id: str, priority: float, arrival_time: float) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        priority=priority,
        arrival_time=arrival_time,
    )


def test_priority_request_queue_orders_float_priorities():
    queue = PriorityRequestQueue()

    queue.add_request(_make_request("low", 2.5, 1.0))
    queue.add_request(_make_request("high", 0.25, 2.0))
    queue.add_request(_make_request("mid", 1.5, 3.0))

    assert queue.pop_request().request_id == "high"
    assert queue.pop_request().request_id == "mid"
    assert queue.pop_request().request_id == "low"
