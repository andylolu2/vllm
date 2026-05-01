# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import LLM

pytestmark = pytest.mark.skip_global_cleanup


def test_priority_to_seq_accepts_floats():
    llm = object.__new__(LLM)

    priorities = llm._priority_to_seq([0.5, 1.25], 2)

    assert list(priorities) == [0.5, 1.25]


def test_priority_to_seq_validates_length_for_floats():
    llm = object.__new__(LLM)

    with pytest.raises(ValueError, match="priority"):
        llm._priority_to_seq([0.5], 2)
