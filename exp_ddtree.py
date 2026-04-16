from collections import Counter
from types import SimpleNamespace
from typing import Any

import torch
from transformers import AutoModelForCausalLM, DynamicCache

from agreement_metrics import build_batch_agreement_snapshot
from ddtree import (
    DDTREE_TREE_BUILD_STAGE_ORDER,
    build_ddtree_tree,
    compact_dynamic_cache,
    compile_ddtree_tree,
    follow_verified_tree,
)
from dflash import cuda_time, dflash_generate, empty_stage_times
from model import DFlashDraftModel, extract_context_feature, sample
from pexpress import build_perturbed_noise_embedding_batch


EXP_DDTREE_BATCH_SIZE = 4
EXP_DDTREE_STAGE_ORDER = ("draft", "tree_build", "tree_compile", "verify", "commit")


def summarize_tree_shape(
    node_depths: torch.Tensor,
    child_maps: list[dict[int, int]],
) -> dict[str, Any]:
    depth_list = [int(depth) for depth in node_depths.tolist()]
    depth_histogram = Counter(depth_list)
    node_count = len(depth_list)
    max_depth = max(depth_list, default=0)
    max_width = max(depth_histogram.values(), default=0)
    mean_width = (node_count / max_depth) if max_depth > 0 else 0.0
    leaf_count = sum(1 for child_map in child_maps[1:] if not child_map)
    non_leaf_count = max(node_count - leaf_count, 0)
    mean_branching = (node_count - depth_histogram.get(1, 0)) / non_leaf_count if non_leaf_count > 0 else 0.0

    return {
        "tree_node_count": int(node_count),
        "tree_max_depth": int(max_depth),
        "tree_max_width": int(max_width),
        "tree_mean_width": float(mean_width),
        "tree_root_width": int(depth_histogram.get(1, 0)),
        "tree_leaf_count": int(leaf_count),
        "tree_non_leaf_count": int(non_leaf_count),
        "tree_mean_branching": float(mean_branching),
        "tree_width_by_depth": [int(depth_histogram.get(depth, 0)) for depth in range(1, max_depth + 1)],
    }


@torch.inference_mode()
def exp_ddtree_generate(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
    tree_budget: int | None = None,
    perturbation_temperature: float = 0.75,
    position_temperature_decay: float = 0.0,
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

    verify_input_ids_buffer = torch.empty((1, max_tree_nodes), dtype=torch.long, device=model.device)
    verify_position_ids_buffer = torch.empty((1, max_tree_nodes), dtype=torch.long, device=model.device)
    attention_mask_buffer = torch.zeros(
        (1, 1, max_tree_nodes, max_length + max_tree_nodes),
        dtype=target.dtype,
        device=model.device,
    )
    tree_visibility_buffer = torch.empty((max_tree_nodes, max_tree_nodes), dtype=torch.bool, device=model.device)

    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()
    past_key_values_alignment = DynamicCache()
    stage_times = empty_stage_times(EXP_DDTREE_STAGE_ORDER + DDTREE_TREE_BUILD_STAGE_ORDER)

    prefill_start = cuda_time()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(output.logits, temperature)
    target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)

    time_to_first_token = cuda_time() - prefill_start

    decode_start = cuda_time()
    round_clock_start = cuda_time()
    start = input_ids.shape[1]
    acceptance_lengths = []
    round_timestamps = []
    round_trees = [] if save_tree_traces else None
    batch_agreement_metrics = []
    exp_ddtree_metrics = []
    draft_prefill = True
    previous_tree_start = 0
    previous_tree_length = 0

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        root_token = block_output_ids[:, :1]

        draft_stage_start = cuda_time()
        base_noise_embedding = target.model.embed_tokens(block_output_ids)
        base_draft_position_ids = position_ids[
            :,
            past_key_values_draft.get_seq_length() : start + block_size,
        ]
        draft_logits = target.lm_head(model(
            target_hidden=target_hidden,
            noise_embedding=base_noise_embedding,
            position_ids=base_draft_position_ids,
            past_key_values=past_key_values_draft,
            use_cache=True,
            is_causal=False,
        )[:, -draft_horizon:, :])
        past_key_values_draft.crop(start)

        # Keep the DDTree path bitwise-aligned with vanilla DDTree and use a
        # separate batched pass only for alignment analysis.
        noise_embedding_batch = build_perturbed_noise_embedding_batch(
            base_noise_embedding=base_noise_embedding,
            num_branches=EXP_DDTREE_BATCH_SIZE,
            perturbation_temperature=perturbation_temperature,
            position_temperature_decay=position_temperature_decay,
        )
        projected_target_hidden = model.project_target_hidden(target_hidden)
        batched_target_hidden = projected_target_hidden.expand(EXP_DDTREE_BATCH_SIZE, -1, -1)
        alignment_position_ids = position_ids[
            :,
            past_key_values_alignment.get_seq_length() : start + block_size,
        ].expand(EXP_DDTREE_BATCH_SIZE, -1)
        alignment_draft_logits = target.lm_head(model(
            target_hidden=batched_target_hidden,
            target_hidden_is_projected=True,
            noise_embedding=noise_embedding_batch,
            position_ids=alignment_position_ids,
            past_key_values=past_key_values_alignment,
            use_cache=True,
            is_causal=False,
        )[:, -draft_horizon:, :])
        past_key_values_alignment.crop(start)
        draft_stage_elapsed = cuda_time() - draft_stage_start
        if draft_prefill:
            draft_prefill = False
            decode_start = cuda_time()
        else:
            stage_times["draft"] += draft_stage_elapsed

        tree_build_start = cuda_time()
        node_token_ids, node_depths, parents, child_maps, visibility_cpu, tree_build_subtimes = build_ddtree_tree(
            draft_logits[0],
            tree_budget,
        )
        stage_times["tree_build"] += cuda_time() - tree_build_start
        for stage_name, stage_elapsed in tree_build_subtimes.items():
            stage_times[stage_name] += stage_elapsed

        tree_compile_start = cuda_time()
        verify_input_ids, verify_position_ids, verify_attention_mask, previous_tree_start, previous_tree_length = compile_ddtree_tree(
            root_token_id=root_token[0, 0],
            start=start,
            node_token_ids=node_token_ids,
            node_depths=node_depths,
            visibility_cpu=visibility_cpu,
            past_length=start,
            dtype=target.dtype,
            device=model.device,
            verify_input_ids_buffer=verify_input_ids_buffer,
            verify_position_ids_buffer=verify_position_ids_buffer,
            attention_mask_buffer=attention_mask_buffer,
            tree_visibility_buffer=tree_visibility_buffer,
            previous_tree_start=previous_tree_start,
            previous_tree_length=previous_tree_length,
        )
        stage_times["tree_compile"] += cuda_time() - tree_compile_start

        verify_stage_start = cuda_time()
        output = target(
            verify_input_ids,
            position_ids=verify_position_ids,
            attention_mask=verify_attention_mask,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )
        stage_times["verify"] += cuda_time() - verify_stage_start

        commit_stage_start = cuda_time()
        posterior = sample(output.logits, temperature)
        accepted_indices, next_token = follow_verified_tree(child_maps, posterior)
        accepted_index_tensor = torch.tensor(accepted_indices, dtype=torch.long, device=verify_input_ids.device)
        accepted_tokens = verify_input_ids.index_select(1, accepted_index_tensor)

        output_ids[:, start : start + len(accepted_indices)] = accepted_tokens
        output_ids[:, start + len(accepted_indices)] = next_token

        compact_dynamic_cache(past_key_values_target, start, accepted_indices)
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids).index_select(1, accepted_index_tensor)

        agreement_snapshot = build_batch_agreement_snapshot(alignment_draft_logits)
        if agreement_snapshot is not None:
            accepted_draft_tokens = max(0, min(len(accepted_indices) - 1, agreement_snapshot["depth_count"]))
            agreement_snapshot["acceptance_length"] = int(len(accepted_indices))
            agreement_snapshot["accepted_draft_tokens"] = int(accepted_draft_tokens)
            batch_agreement_metrics.append(agreement_snapshot)

            exp_metric = {
                "acceptance_length": int(len(accepted_indices)),
                "accepted_draft_tokens": int(accepted_draft_tokens),
                "mean_alignment": float(sum(agreement_snapshot["majority_agreement"]) / len(agreement_snapshot["majority_agreement"])),
                "first_alignment": float(agreement_snapshot["majority_agreement"][0]),
                "mean_base_alignment": float(sum(agreement_snapshot["base_agreement"]) / len(agreement_snapshot["base_agreement"])),
            }
            exp_metric.update(summarize_tree_shape(node_depths, child_maps))
            exp_ddtree_metrics.append(exp_metric)

        acceptance_lengths.append(len(accepted_indices))
        start += len(accepted_indices)
        stage_times["commit"] += cuda_time() - commit_stage_start
        round_timestamps.append(cuda_time() - round_clock_start)
        if save_tree_traces:
            round_summary = exp_ddtree_metrics[-1] if exp_ddtree_metrics else None
            round_trees.append({
                "accepted_indices": [int(index) for index in accepted_indices],
                "analysis": round_summary,
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
        exp_ddtree_metrics=exp_ddtree_metrics,
    )
