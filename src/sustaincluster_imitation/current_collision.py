from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from sustaincluster_imitation.structured_current import (
    ACTION_DIM,
    TrainOnlyStandardizer,
    canonical_digest,
    label_entropy,
)


def mask_signatures(masks: np.ndarray) -> np.ndarray:
    powers = (1 << np.arange(masks.shape[1], dtype=np.int64))[None, :]
    return (masks.astype(np.int64) * powers).sum(axis=1)


def row_digests(features: np.ndarray, masks: np.ndarray) -> np.ndarray:
    values = []
    for feature, mask in zip(features, masks):
        digest = hashlib.blake2b(digest_size=16)
        digest.update(np.ascontiguousarray(feature).tobytes())
        digest.update(np.packbits(mask.astype(np.uint8)).tobytes())
        values.append(digest.hexdigest())
    return np.asarray(values, dtype=str)


def collision_groups(
    *,
    keys: np.ndarray,
    frame: pd.DataFrame,
    method: str,
    representation: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = pd.DataFrame(
        {
            "key": keys,
            "split": frame["split"].astype(str).to_numpy(),
            "label": frame["h1_action_index"].to_numpy(dtype=np.int64),
            "sample_id": frame["sample_id"].astype(str).to_numpy(),
        }
    )
    detail_rows = []
    summary_rows = []
    scopes = [("all", working)] + [
        (split, working.loc[working["split"] == split])
        for split in ("train", "validation", "test")
    ]
    for scope, scoped in scopes:
        counts = scoped.groupby(["key", "label"], sort=False).size().unstack(fill_value=0)
        counts = counts.reindex(columns=range(ACTION_DIM), fill_value=0)
        totals = counts.sum(axis=1)
        distinct = (counts > 0).sum(axis=1)
        duplicate = totals >= 2
        multi = distinct >= 2
        summary_rows.append(
            {
                "representation": representation,
                "method": method,
                "split": scope,
                "samples": len(scoped),
                "duplicate_groups": int(duplicate.sum()),
                "multi_label_collision_groups": int(multi.sum()),
                "rows_in_duplicate_groups": int(totals[duplicate].sum()),
                "rows_in_multi_label_groups": int(totals[multi].sum()),
                "empirical_duplicate_group_ceiling": float(
                    counts.max(axis=1).sum() / max(1, len(scoped))
                ),
            }
        )
        samples_by_key = scoped.groupby("key", sort=False)["sample_id"].agg(list)
        for key in counts.index[duplicate]:
            row_counts = counts.loc[key].to_numpy(dtype=np.int64)
            total = int(row_counts.sum())
            labels = np.repeat(np.arange(ACTION_DIM), row_counts)
            detail_rows.append(
                {
                    "representation": representation,
                    "method": method,
                    "split": scope,
                    "group_id": key,
                    "sample_count": total,
                    "distinct_h1_labels": int((row_counts > 0).sum()),
                    "h1_label_histogram": json.dumps(
                        {str(i): int(value) for i, value in enumerate(row_counts)}
                    ),
                    "majority_label_rate": float(row_counts.max() / total),
                    "label_entropy_bits": label_entropy(labels),
                    "label_collision": bool((row_counts > 0).sum() > 1),
                    "sample_ids": " | ".join(samples_by_key.loc[key][:8]),
                }
            )
    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


def current34_collision_audit(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, TrainOnlyStandardizer, pd.DataFrame, pd.DataFrame]:
    observations = np.stack(
        frame["student_observation"].map(lambda value: np.asarray(value, dtype=np.float32))
    )
    masks = np.stack(
        frame["feasible_action_mask"].map(lambda value: np.asarray(value, dtype=bool))
    )
    train = frame["split"].eq("train").to_numpy()
    scaler = TrainOnlyStandardizer.fit(observations[train], split="train")
    normalized = scaler.transform(observations)
    decimals = int(config["exact_collision"]["standardized_round_decimals"])
    raw_detail, raw_summary = collision_groups(
        keys=row_digests(observations, masks),
        frame=frame,
        method="raw_float32_exact",
        representation="Current34",
    )
    rounded_detail, rounded_summary = collision_groups(
        keys=row_digests(np.round(normalized, decimals=decimals), masks),
        frame=frame,
        method=f"train_zscore_round_{decimals}_decimals",
        representation="Current34",
    )
    return (
        normalized,
        scaler,
        pd.concat((raw_detail, rounded_detail), ignore_index=True),
        pd.concat((raw_summary, rounded_summary), ignore_index=True),
    )


def near_neighbor_audit(
    normalized: np.ndarray,
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    settings = config["near_neighbor"]
    query_limit = int(settings["query_limit_per_split"])
    reference_limit = int(settings["reference_limit_per_split"])
    ks = tuple(int(value) for value in settings["ks"])
    k_max = max(ks)
    rng = np.random.default_rng(int(settings["seed"]))
    labels = frame["h1_action_index"].to_numpy(dtype=np.int64)
    masks = np.stack(
        frame["feasible_action_mask"].map(lambda value: np.asarray(value, dtype=bool))
    )
    signatures = mask_signatures(masks)
    risk = frame["deployable_risk_score"].to_numpy(dtype=float)
    consensus = (
        frame["teacher_action_index"].to_numpy(dtype=np.int64)
        == frame["h1_action_index"].to_numpy(dtype=np.int64)
    )
    threshold = float(config["risk_threshold_p95"])
    rows = []
    pairs = []
    for split in ("train", "validation", "test"):
        split_indices = np.flatnonzero(frame["split"].eq(split).to_numpy())
        reference = rng.permutation(split_indices)[: min(reference_limit, len(split_indices))]
        query = reference[: min(query_limit, len(reference))]
        neighbors = np.full((len(query), k_max), -1, dtype=np.int64)
        opposite_indices = np.full(len(query), -1, dtype=np.int64)
        opposite_distances = np.full(len(query), np.inf, dtype=np.float64)
        query_masks = signatures[query]
        reference_masks = signatures[reference]
        for signature in np.unique(query_masks):
            q_positions = np.flatnonzero(query_masks == signature)
            r_positions = np.flatnonzero(reference_masks == signature)
            if not len(r_positions):
                continue
            r_global = reference[r_positions]
            tree = cKDTree(normalized[r_global])
            requested = min(k_max + 1, len(r_global))
            _, local = tree.query(normalized[query[q_positions]], k=requested)
            if requested == 1:
                local = local[:, None]
            for local_q, q_position in enumerate(q_positions):
                candidates = r_global[np.asarray(local[local_q], dtype=np.int64)]
                candidates = candidates[candidates != query[q_position]][:k_max]
                neighbors[q_position, : len(candidates)] = candidates
            for target_label in range(ACTION_DIM):
                label_reference = r_global[labels[r_global] == target_label]
                if not len(label_reference):
                    continue
                target_tree = cKDTree(normalized[label_reference])
                for query_label in range(ACTION_DIM):
                    if query_label == target_label:
                        continue
                    selected = q_positions[labels[query[q_positions]] == query_label]
                    if not len(selected):
                        continue
                    distance, local_index = target_tree.query(normalized[query[selected]], k=1)
                    distance = np.atleast_1d(distance)
                    local_index = np.atleast_1d(local_index).astype(np.int64)
                    improve = distance < opposite_distances[selected]
                    improved = selected[improve]
                    opposite_distances[improved] = distance[improve]
                    opposite_indices[improved] = label_reference[local_index[improve]]
        strata = {
            "ALL": np.ones(len(query), dtype=bool),
            "LOW_RISK": risk[query] < threshold,
            "HIGH_RISK": risk[query] >= threshold,
            "H1_ORACLE_CONSENSUS": consensus[query],
            "H1_ORACLE_DISAGREEMENT": ~consensus[query],
        }
        for stratum, selector in strata.items():
            for k in ks:
                selected_neighbors = neighbors[selector, :k]
                valid = selected_neighbors >= 0
                if valid.any():
                    query_labels = labels[query[selector]][:, None]
                    neighbor_labels = np.where(
                        valid, labels[np.maximum(selected_neighbors, 0)], -1
                    )
                    agreement = float(
                        ((neighbor_labels == query_labels) & valid).sum() / valid.sum()
                    )
                    entropy = float(
                        np.mean(
                            [
                                label_entropy(values[row_valid])
                                for values, row_valid in zip(neighbor_labels, valid)
                            ]
                        )
                    )
                else:
                    agreement = float("nan")
                    entropy = float("nan")
                opposite = opposite_distances[selector]
                finite = opposite[np.isfinite(opposite)]
                rows.append(
                    {
                        "split": split,
                        "stratum": stratum,
                        "k": k,
                        "query_samples": int(selector.sum()),
                        "reference_samples": len(reference),
                        "same_feasible_mask_only": True,
                        "label_agreement": agreement,
                        "near_neighbor_disagreement": 1.0 - agreement,
                        "mean_local_label_entropy_bits": entropy,
                        "median_nearest_opposite_label_distance": float(np.median(finite))
                        if len(finite)
                        else float("nan"),
                        "opposite_label_reachable_rate": float(np.isfinite(opposite).mean())
                        if len(opposite)
                        else float("nan"),
                    }
                )
        for position, (other, distance) in enumerate(zip(opposite_indices, opposite_distances)):
            if other >= 0 and np.isfinite(distance):
                pairs.append(
                    {
                        "split": split,
                        "left_index": int(query[position]),
                        "right_index": int(other),
                        "current34_distance": float(distance),
                    }
                )
    pairs.sort(key=lambda row: row["current34_distance"])
    unique = []
    seen = set()
    for row in pairs:
        pair = tuple(sorted((row["left_index"], row["right_index"])))
        if pair in seen:
            continue
        seen.add(pair)
        unique.append(row)
        if len(unique) >= 100:
            break
    return pd.DataFrame(rows), unique


def structured_keys(
    pending: np.ndarray,
    running: np.ndarray,
    dcs: np.ndarray,
    state_offsets: np.ndarray,
    running_offsets: np.ndarray,
    masks: np.ndarray,
    decimals: int | None,
) -> np.ndarray:
    keys = np.empty(len(pending), dtype="U32")
    for state_index in range(len(state_offsets) - 1):
        task_start, task_stop = state_offsets[state_index : state_index + 2]
        run_start, run_stop = running_offsets[state_index : state_index + 2]
        state_pending = pending[task_start:task_stop]
        state_running = running[run_start:run_stop]
        state_dc = dcs[state_index]
        if decimals is not None:
            state_pending = np.round(state_pending, decimals)
            state_running = np.round(state_running, decimals)
            state_dc = np.round(state_dc, decimals)
        pending_order = np.argsort(state_pending[:, 10], kind="stable")
        if len(state_running):
            state_running = state_running[np.lexsort(state_running.T[::-1])]
        state_digest = canonical_digest(state_pending[pending_order], state_running, state_dc)
        for task_index in range(task_start, task_stop):
            digest = hashlib.blake2b(digest_size=16)
            digest.update(state_digest.encode("ascii"))
            digest.update(np.ascontiguousarray(pending[task_index]).tobytes())
            digest.update(np.packbits(masks[task_index].astype(np.uint8)).tobytes())
            keys[task_index] = digest.hexdigest()
    return keys


def representation_collision_summary(
    *,
    frame: pd.DataFrame,
    masks: np.ndarray,
    current34: np.ndarray,
    dense_raw: np.ndarray,
    dense_normalized: np.ndarray,
    pending_raw: np.ndarray,
    pending_normalized: np.ndarray,
    running_raw: np.ndarray,
    running_normalized: np.ndarray,
    dc_raw: np.ndarray,
    dc_normalized: np.ndarray,
    state_offsets: np.ndarray,
    running_offsets: np.ndarray,
    decimals: int,
) -> pd.DataFrame:
    representations = [
        ("Current34", "raw_exact", row_digests(current34, masks)),
        ("CurrentDense", "raw_exact", row_digests(dense_raw, masks)),
        (
            "CurrentDense",
            f"train_zscore_round_{decimals}_decimals",
            row_digests(np.round(dense_normalized, decimals), masks),
        ),
        (
            "StructuredCurrent",
            "raw_exact",
            structured_keys(
                pending_raw,
                running_raw,
                dc_raw,
                state_offsets,
                running_offsets,
                masks,
                None,
            ),
        ),
        (
            "StructuredCurrent",
            f"train_zscore_round_{decimals}_decimals",
            structured_keys(
                pending_normalized,
                running_normalized,
                dc_normalized,
                state_offsets,
                running_offsets,
                masks,
                decimals,
            ),
        ),
    ]
    rows = []
    for representation, method, keys in representations:
        _, summary = collision_groups(
            keys=keys, frame=frame, method=method, representation=representation
        )
        rows.append(summary.loc[summary["split"] == "all"])
    return pd.concat(rows, ignore_index=True)
