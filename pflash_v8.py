from types import SimpleNamespace
from typing import Any

import torch
from transformers import AutoModelForCausalLM, DynamicCache

from agreement_metrics import append_batch_agreement_metric
from dflash import cuda_time, dflash_generate, empty_stage_times
from ddtree import DDTREE_TREE_BUILD_STAGE_ORDER, follow_verified_tree, compact_dynamic_cache
from model import DFlashDraftModel, extract_context_feature, sample
from pflash import build_pflash_tree
from pflash_v2 import repeat_dynamic_cache_batch, select_dynamic_cache_batch
from pflash_v7 import select_multiverse_anchor_tokens


PFLASH_V8_BATCH_SIZE = 4
PFLASH_V8_STAGE_ORDER = ("draft", "tree_build", "tree_compile", "verify", "commit")


def compile_shared_tree_forest(
    root_token_ids: torch.Tensor,
    start: int,
    node_token_ids: torch.Tensor,
    node_depths: torch.Tensor,
    visibility_cpu: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
    verify_input_ids_buffer: torch.Tensor,
    verify_position_ids_buffer: torch.Tensor,
    attention_mask_buffer: torch.Tensor,
    tree_visibility_buffer: torch.Tensor,
    previous_tree_start: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    batch_size, max_tree_nodes = verify_input_ids_buffer.shape
    current_length = 1 + int(node_token_ids.numel())

    if previous_tree_start is not None:
        attention_mask_buffer[:, 0, :, previous_tree_start : previous_tree_start + max_tree_nodes] = 0

    verify_input_ids_buffer.copy_(root_token_ids[:, None].expand(batch_size, max_tree_nodes))
    verify_position_ids_buffer.fill_(start)
    if current_length > 1:
        verify_input_ids_buffer[:, 1:current_length].copy_(node_token_ids.unsqueeze(0).expand(batch_size, -1), non_blocking=False)
        verify_position_ids_buffer[:, 1:current_length].copy_(node_depths.unsqueeze(0).expand(batch_size, -1), non_blocking=False)
        verify_position_ids_buffer[:, 1:current_length].add_(start)

    tree_visibility_buffer.zero_()
    diag_indices = torch.arange(max_tree_nodes, device=device)
    tree_visibility_buffer[:, diag_indices, diag_indices] = True
    tree_visibility_buffer[:, :current_length, :current_length].copy_(
        visibility_cpu.to(device=device, dtype=torch.bool).unsqueeze(0).expand(batch_size, -1, -1),
        non_blocking=False,
    )

    tree_block = attention_mask_buffer[:, 0, :, start : start + max_tree_nodes]
    tree_block.fill_(torch.finfo(dtype).min)
    tree_block.masked_fill_(tree_visibility_buffer, 0)

    attention_mask = attention_mask_buffer[:, :, :, : start + max_tree_nodes]
    return verify_input_ids_buffer, verify_position_ids_buffer, attention_mask, start


def follow_verified_shared_tree_forest(
    child_maps: list[dict[int, int]],
    posterior_logits: torch.Tensor,
    temperature: float,
) -> tuple[int, list[int], int, list[int]]:
    posterior = sample(posterior_logits, temperature)

    best_branch_idx = 0
    best_accepted_indices = [0]
    best_next_token = int(posterior[0, 0])
    branch_acceptance_lengths = []

    for branch_idx in range(posterior.shape[0]):
        accepted_indices, next_token = follow_verified_tree(
            child_maps=child_maps,
            posterior=posterior[branch_idx : branch_idx + 1],
        )
        branch_acceptance_lengths.append(int(len(accepted_indices)))
        if len(accepted_indices) > len(best_accepted_indices):
            best_branch_idx = branch_idx
            best_accepted_indices = accepted_indices
            best_next_token = next_token

    return best_branch_idx, best_accepted_indices, best_next_token, branch_acceptance_lengths


@torch.inference_mode()
def pflash_v8_generate(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
    tree_budget: int | None = None,
    merge_prefix_branches: bool = False,
    prefix_support_bonus_weight: float = 0.0,
    measure_batch_agreement: bool = False,
    save_tree_traces: bool = False,
) -> SimpleNamespace:
    if block_size <= 1:
        return dflash_generate(
            model=model,
            target=target,
            input_ids=input_ids,
            mask_token_id=mask_token_id,
            max_new_tokens=max_new_tokens,
            block_size=block_size,
            stop_token_ids=stop_token_ids,
            temperature=temperature,
        )

    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    draft_horizon = block_size - 1
    tree_budget = draft_horizon if tree_budget is None else max(tree_budget, 0)
    max_tree_nodes = 1 + tree_budget

    output_ids = torch.full(
        (1, max_length + max_tree_nodes),
        mask_token_id,
        dtype=torch.long,
        device=model.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=model.device).unsqueeze(0)
    stop_token_ids_tensor = None if stop_token_ids is None else torch.tensor(stop_token_ids, device=model.device)

    multiverse_input_ids_buffer = torch.full(
        (PFLASH_V8_BATCH_SIZE, block_size),
        mask_token_id,
        dtype=torch.long,
        device=model.device,
    )
    verify_input_ids_buffer = torch.empty(
        (PFLASH_V8_BATCH_SIZE, max_tree_nodes),
        dtype=torch.long,
        device=model.device,
    )
    verify_position_ids_buffer = torch.empty(
        (PFLASH_V8_BATCH_SIZE, max_tree_nodes),
        dtype=torch.long,
        device=model.device,
    )
    attention_mask_buffer = torch.zeros(
        (PFLASH_V8_BATCH_SIZE, 1, max_tree_nodes, max_length + max_tree_nodes),
        dtype=target.dtype,
        device=model.device,
    )
    tree_visibility_buffer = torch.empty(
        (PFLASH_V8_BATCH_SIZE, max_tree_nodes, max_tree_nodes),
        dtype=torch.bool,
        device=model.device,
    )

    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()
    stage_times = empty_stage_times(PFLASH_V8_STAGE_ORDER + DDTREE_TREE_BUILD_STAGE_ORDER)

    prefill_start = cuda_time()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )

    initial_root_token = sample(output.logits, temperature)
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = initial_root_token
    target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)
    anchor_logits = output.logits[0, 0].detach()

    time_to_first_token = cuda_time() - prefill_start

    decode_start = cuda_time()
    round_clock_start = cuda_time()
    start = input_ids.shape[1]
    acceptance_lengths = []
    round_timestamps = []
    round_trees = [] if save_tree_traces else None
    batch_agreement_metrics = [] if measure_batch_agreement else None
    pflash_v8_metrics: list[dict[str, Any]] = []
    draft_prefill = True
    previous_tree_start = None

    while start < max_length:
        base_root_token = int(output_ids[0, start])
        anchor_tokens, anchor_ranks = select_multiverse_anchor_tokens(
            anchor_logits=anchor_logits,
            branch0_token_id=base_root_token,
            num_branches=PFLASH_V8_BATCH_SIZE,
        )

        draft_stage_start = cuda_time()
        multiverse_input_ids = multiverse_input_ids_buffer
        multiverse_input_ids.fill_(mask_token_id)
        multiverse_input_ids[:, 0].copy_(anchor_tokens)

        draft_past_key_values = repeat_dynamic_cache_batch(
            past_key_values=past_key_values_draft,
            batch_size=PFLASH_V8_BATCH_SIZE,
        )
        projected_target_hidden = model.project_target_hidden(target_hidden)
        batched_target_hidden = projected_target_hidden.expand(PFLASH_V8_BATCH_SIZE, -1, -1)
        draft_position_ids = position_ids[
            :,
            past_key_values_draft.get_seq_length() : start + block_size,
        ].expand(PFLASH_V8_BATCH_SIZE, -1)
        draft_logits = target.lm_head(model(
            target_hidden=batched_target_hidden,
            target_hidden_is_projected=True,
            noise_embedding=target.model.embed_tokens(multiverse_input_ids),
            position_ids=draft_position_ids,
            past_key_values=draft_past_key_values,
            use_cache=True,
            is_causal=False,
        )[:, -draft_horizon:, :])
        past_key_values_draft = select_dynamic_cache_batch(draft_past_key_values, 0)
        past_key_values_draft.crop(start)
        draft_stage_elapsed = cuda_time() - draft_stage_start
        if draft_prefill:
            draft_prefill = False
            decode_start = cuda_time()
        else:
            stage_times["draft"] += draft_stage_elapsed

        tree_build_start = cuda_time()
        anchor_log_priors = torch.log_softmax(anchor_logits.float(), dim=-1).index_select(0, anchor_tokens)
        (
            node_token_ids,
            node_depths,
            parents,
            child_maps,
            visibility_cpu,
            tree_build_subtimes,
        ) = build_pflash_tree(
            draft_logits=draft_logits,
            budget=tree_budget,
            branch_log_priors=anchor_log_priors,
            merge_prefix_branches=merge_prefix_branches,
            prefix_support_bonus_weight=prefix_support_bonus_weight,
        )
        stage_times["tree_build"] += cuda_time() - tree_build_start
        for stage_name, stage_elapsed in tree_build_subtimes.items():
            stage_times[stage_name] += stage_elapsed

        tree_compile_start = cuda_time()
        verify_input_ids, verify_position_ids, verify_attention_mask, previous_tree_start = compile_shared_tree_forest(
            root_token_ids=anchor_tokens,
            start=start,
            node_token_ids=node_token_ids,
            node_depths=node_depths,
            visibility_cpu=visibility_cpu,
            dtype=target.dtype,
            device=model.device,
            verify_input_ids_buffer=verify_input_ids_buffer,
            verify_position_ids_buffer=verify_position_ids_buffer,
            attention_mask_buffer=attention_mask_buffer,
            tree_visibility_buffer=tree_visibility_buffer,
            previous_tree_start=previous_tree_start,
        )
        stage_times["tree_compile"] += cuda_time() - tree_compile_start

        verify_stage_start = cuda_time()
        verify_past_key_values = repeat_dynamic_cache_batch(
            past_key_values=past_key_values_target,
            batch_size=PFLASH_V8_BATCH_SIZE,
        )
        output = target(
            verify_input_ids,
            position_ids=verify_position_ids,
            attention_mask=verify_attention_mask,
            past_key_values=verify_past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )
        stage_times["verify"] += cuda_time() - verify_stage_start

        commit_stage_start = cuda_time()
        selected_branch_idx, accepted_indices, next_token, branch_acceptance_lengths = follow_verified_shared_tree_forest(
            child_maps=child_maps,
            posterior_logits=output.logits,
            temperature=temperature,
        )
        append_batch_agreement_metric(batch_agreement_metrics, draft_logits, accepted_indices)
        accepted_index_tensor = torch.tensor(accepted_indices, dtype=torch.long, device=verify_input_ids.device)
        selected_verify_input_ids = verify_input_ids[selected_branch_idx : selected_branch_idx + 1]
        accepted_tokens = selected_verify_input_ids.index_select(1, accepted_index_tensor)

        output_ids[:, start : start + len(accepted_indices)] = accepted_tokens
        output_ids[:, start + len(accepted_indices)] = next_token

        past_key_values_target = select_dynamic_cache_batch(verify_past_key_values, selected_branch_idx)
        compact_dynamic_cache(past_key_values_target, start, accepted_indices)

        target_hidden_batch = extract_context_feature(output.hidden_states, model.target_layer_ids)
        target_hidden = target_hidden_batch[selected_branch_idx : selected_branch_idx + 1].index_select(1, accepted_index_tensor)
        anchor_logits = output.logits[selected_branch_idx, len(accepted_indices) - 1].detach()

        pflash_v8_metrics.append({
            "anchor_tokens": [int(token_id) for token_id in anchor_tokens.tolist()],
            "anchor_ranks": [None if rank is None else int(rank) for rank in anchor_ranks],
            "branch_acceptance_lengths": [int(length) for length in branch_acceptance_lengths],
            "selected_branch": int(selected_branch_idx),
            "selected_anchor_rank": (
                None if anchor_ranks[selected_branch_idx] is None else int(anchor_ranks[selected_branch_idx])
            ),
            "base_acceptance_length": int(branch_acceptance_lengths[0]),
            "selected_acceptance_length": int(branch_acceptance_lengths[selected_branch_idx]),
            "alternative_branch_selected": bool(selected_branch_idx != 0),
            "tree_node_count": int(node_token_ids.numel()),
        })

        acceptance_lengths.append(len(accepted_indices))
        start += len(accepted_indices)
        stage_times["commit"] += cuda_time() - commit_stage_start
        round_timestamps.append(cuda_time() - round_clock_start)
        if save_tree_traces:
            round_trees.append({
                "selected_branch": int(selected_branch_idx),
                "anchor_tokens": [int(token_id) for token_id in anchor_tokens.tolist()],
                "branch_acceptance_lengths": [int(length) for length in branch_acceptance_lengths],
                "tree": {
                    "node_token_ids": [int(token_id) for token_id in node_token_ids.tolist()],
                    "node_depths": [int(depth) for depth in node_depths.tolist()],
                    "parents": [int(parent) for parent in parents],
                },
            })

        if stop_token_ids_tensor is not None:
            new_tokens = output_ids[:, start - len(accepted_indices) : start + 1]
            if torch.isin(new_tokens[0], stop_token_ids_tensor).any():
                break

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]
    if stop_token_ids_tensor is not None:
        stop_token_indices = torch.isin(output_ids[0][num_input_tokens:], stop_token_ids_tensor).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = cuda_time() - decode_start
    time_per_output_token = total_decode_time / max(num_output_tokens, 1)

    return SimpleNamespace(
        output_ids=output_ids.cpu(),
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        acceptance_lengths=acceptance_lengths,
        decode_rounds=len(acceptance_lengths),
        stage_times=stage_times,
        round_timestamps=round_timestamps,
        round_trees=round_trees,
        batch_agreement_metrics=batch_agreement_metrics,
        pflash_v8_metrics=pflash_v8_metrics,
    )
