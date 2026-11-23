# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for MRv2 GPUModelRunner.add_requests streaming input support."""

from unittest.mock import MagicMock

import pytest
import torch

from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    PromptExtension,
    SchedulerOutput,
)
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache


def _make_scheduler_output(new_reqs, cached_reqs=None):
    return SchedulerOutput(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=cached_reqs or CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _audio_feature(identifier, offset, length):
    return MultiModalFeatureSpec(
        data=None,
        modality="audio",
        identifier=identifier,
        mm_position=PlaceholderRange(offset=offset, length=length),
    )


def test_e2e_streaming_request_update_basic_flow(
    gpu_model_runner_for_requests,
):
    """Test that streaming sessions are updated correctly.

    This test validates that when a streaming session is updated with new
    prompt tokens:
    1. The old request state is removed (no free_indices leak)
    2. The new state is written with updated prefill_token_ids
    3. block tables contain the new blocks
    """
    runner = gpu_model_runner_for_requests
    req_states = runner.req_states
    req_id = "streaming_req_0"
    initial_free = len(req_states.free_indices)

    # Step 1: Add initial request with 3 prompt tokens, all computed
    initial_req_data = NewRequestData(
        req_id=req_id,
        prompt_token_ids=[1, 2, 3],
        prefill_token_ids=[1, 2, 3],
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=([0],),
        num_computed_tokens=3,
        lora_request=None,
    )
    runner.add_requests(_make_scheduler_output([initial_req_data]))
    assert req_id in req_states.req_id_to_index
    assert len(req_states.free_indices) == initial_free - 1

    # Step 2: Create streaming update with extended prompt
    # The scheduler has already set prefill_token_ids to the full sequence
    # (original prompt + intermediate output + new prompt tokens)
    updated_req_data = NewRequestData(
        req_id=req_id,
        prompt_token_ids=[1, 2, 3],
        prefill_token_ids=[1, 2, 3, 10, 4, 5],
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=([0, 1],),
        num_computed_tokens=4,  # 3 original prompt + 1 intermediate output
        lora_request=None,
    )
    runner.add_requests(_make_scheduler_output([updated_req_data]))

    # Step 3: Verify no free_indices leak (old slot recycled)
    assert len(req_states.free_indices) == initial_free - 1

    # Verify the request is still tracked with exactly one index
    assert req_id in req_states.req_id_to_index
    assert sum(1 for v in req_states.index_to_req_id.values() if v == req_id) == 1

    # Verify state was updated with new values
    new_idx = req_states.req_id_to_index[req_id]
    assert req_states.prompt_len.np[new_idx] == 3
    assert req_states.prefill_len.np[new_idx] == 6
    assert req_states.num_computed_prefill_tokens[new_idx] == 4

    runner.block_tables.apply_staged_writes()
    torch.testing.assert_close(
        runner.block_tables.block_tables[0].gpu[new_idx, :2].cpu(),
        torch.tensor([0, 1], dtype=torch.int32),
    )


def test_e2e_streaming_with_multimodal_features(
    gpu_model_runner_for_requests,
):
    """Test that streaming sessions with multimodal features are updated.

    This test validates that when a streaming session with mm features
    is updated:
    1. The old request state is removed (no free_indices leak)
    2. encoder_cache is cleaned up and re-registered with new mm_features
    3. the encoder cache holds all features from the updated request
    """
    runner = gpu_model_runner_for_requests
    req_states = runner.req_states
    req_id = "streaming_mm_req_0"
    initial_free = len(req_states.free_indices)

    # Enable encoder_cache for multimodal
    runner.encoder_cache = EncoderCache()

    # Step 1: Add initial request with one audio feature
    mm_feature_1 = MultiModalFeatureSpec(
        data=None,
        modality="audio",
        identifier="audio-1",
        mm_position=PlaceholderRange(offset=2, length=10),
    )
    initial_req_data = NewRequestData(
        req_id=req_id,
        prompt_token_ids=[1, 2] + [0] * 10 + [3, 4],
        prefill_token_ids=[1, 2] + [0] * 10 + [3, 4],
        mm_features=[mm_feature_1],
        sampling_params=None,
        pooling_params=None,
        block_ids=([0],),
        num_computed_tokens=14,
        lora_request=None,
    )
    runner.add_requests(_make_scheduler_output([initial_req_data]))
    assert req_id in req_states.req_id_to_index

    assert runner.encoder_cache.mm_features[req_id] == [mm_feature_1]

    # Step 2: Create streaming update with additional multimodal feature
    # The scheduler has folded the intermediate output (100) into
    # prefill_token_ids and added a new audio chunk
    mm_feature_2 = MultiModalFeatureSpec(
        data=None,
        modality="audio",
        identifier="audio-2",
        mm_position=PlaceholderRange(offset=15, length=5),
    )
    updated_req_data = NewRequestData(
        req_id=req_id,
        prompt_token_ids=[1, 2] + [0] * 10 + [3, 4],
        prefill_token_ids=[1, 2] + [0] * 10 + [3, 4, 100] + [0] * 5 + [5],
        mm_features=[mm_feature_1, mm_feature_2],
        sampling_params=None,
        pooling_params=None,
        block_ids=([0, 1],),
        num_computed_tokens=14,
        lora_request=None,
    )
    runner.add_requests(_make_scheduler_output([updated_req_data]))

    # Step 3: Verify no free_indices leak
    assert len(req_states.free_indices) == initial_free - 1
    assert sum(1 for v in req_states.index_to_req_id.values() if v == req_id) == 1

    assert runner.encoder_cache.mm_features[req_id] == [mm_feature_1, mm_feature_2]
    new_idx = req_states.req_id_to_index[req_id]

    # Verify updated prefill length
    assert req_states.prefill_len.np[new_idx] == 21


@pytest.mark.parametrize(
    ("prefix_len", "new_token_ids"),
    [(5, []), (5, [20, 21]), (4, [20, 21])],
    ids=["kept_token", "kept_token_and_new_tokens", "new_tokens"],
)
def test_prompt_extension_matches_readmission(
    gpu_model_runner_for_requests, prefix_len, new_token_ids
):
    """Extending a request's prompt in place leaves its state as re-admitting
    it with the folded prompt, features and new sampling params would, and
    re-admits it to the sampler and the speculator."""
    runner = gpu_model_runner_for_requests
    runner.sampler = MagicMock()
    runner.prompt_logprobs_worker = MagicMock()
    runner.speculator = MagicMock()
    runner.adaptive_verification = MagicMock()
    runner.encoder_cache = EncoderCache()
    req_states = runner.req_states
    held_sampling_params = SamplingParams(max_tokens=4)
    sampling_params = SamplingParams(max_tokens=6)
    # The workers hold the prompt and the samples 10 and 11, of which only 10
    # has been computed.
    held = [1, 2, 3, 10, 11]
    folded = held[:prefix_len] + new_token_ids
    held_features = [_audio_feature("audio-0", offset=1, length=1)]
    new_features = (
        [_audio_feature("audio-1", offset=prefix_len, length=len(new_token_ids))]
        if new_token_ids
        else []
    )

    def new_req_data(
        req_id, prompt_token_ids, prefill_token_ids, mm_features, sampling_params
    ):
        return NewRequestData(
            req_id=req_id,
            prompt_token_ids=prompt_token_ids,
            prefill_token_ids=prefill_token_ids,
            mm_features=mm_features,
            sampling_params=sampling_params,
            pooling_params=None,
            block_ids=([0],),
            num_computed_tokens=4,
            lora_request=None,
        )

    runner.add_requests(
        _make_scheduler_output(
            [
                new_req_data(
                    "in_place", [1, 2, 3], held, held_features, held_sampling_params
                ),
                new_req_data(
                    "readmitted",
                    folded,
                    folded,
                    held_features + new_features,
                    sampling_params,
                ),
            ]
        )
    )
    for worker in (
        runner.sampler,
        runner.prompt_logprobs_worker,
        runner.speculator,
        runner.adaptive_verification,
    ):
        worker.reset_mock()
    i = req_states.req_id_to_index["in_place"]
    j = req_states.req_id_to_index["readmitted"]
    # A stop that drops accepted drafts leaves the workers ahead of the
    # scheduler.
    req_states.num_computed_tokens.gpu[i] = len(held)

    runner.update_requests(
        _make_scheduler_output(
            [],
            CachedRequestData(
                req_ids=["in_place"],
                new_token_ids=[],
                new_block_ids=[None],
                num_computed_tokens=[4],
                num_output_tokens=[0],
                prompt_extensions={
                    "in_place": PromptExtension(
                        prefix_len=prefix_len,
                        token_ids=new_token_ids,
                        mm_features=new_features,
                        sampling_params=sampling_params,
                    )
                },
            ),
        )
    )

    for state in (
        req_states.prompt_len.np,
        req_states.prefill_len.np,
        req_states.max_seq_len,
        req_states.num_computed_tokens_np,
        req_states.num_computed_prefill_tokens,
    ):
        assert state[i] == state[j]
    assert req_states.prompt_len.np[i] == len(folded)
    total_len = req_states.total_len.gpu.cpu()
    assert total_len[i] == total_len[j] == len(folded)
    num_computed_tokens = req_states.num_computed_tokens.gpu.cpu()
    assert num_computed_tokens[i] == num_computed_tokens[j] == 4
    all_token_ids = req_states.all_token_ids.gpu[:, : len(folded)].cpu()
    assert all_token_ids[i].tolist() == all_token_ids[j].tolist() == folded
    assert (
        runner.encoder_cache.mm_features["in_place"]
        == runner.encoder_cache.mm_features["readmitted"]
        == held_features + new_features
    )

    runner.sampler.add_request.assert_called_once_with(i, sampling_params)
    runner.sampler.apply_staged_writes.assert_called_once()
    runner.prompt_logprobs_worker.add_request.assert_called_once_with(
        "in_place", i, sampling_params
    )
    runner.speculator.add_request.assert_called_once_with(i)
    runner.adaptive_verification.add_request.assert_called_once_with(i)
