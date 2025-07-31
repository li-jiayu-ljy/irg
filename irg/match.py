# Contains large-scale Google OR-Tools implementation only.
import warnings
from collections import defaultdict, deque
from typing import List, Optional, Tuple

import numpy as np
import torch
from joblib import Parallel, delayed
from ortools.graph.python.min_cost_flow import SimpleMinCostFlow
from torch.nn import functional as F
from tqdm import tqdm


def compute_distances(
        norm_x: torch.Tensor, norm_y: torch.Tensor, pools: List[Optional[np.ndarray]],
        connected_components: List[np.array], connected_pool_indices: List[np.array],
        compute_all_distances: np.array,
        x_indices=None, y_indices=None, int_scale=None
) -> torch.Tensor:
    size_a, size_b = norm_x.shape[0], norm_y.shape[0]
    if x_indices is not None:
        x_indices = np.arange(norm_x.shape[0])[x_indices]
        global_to_x = {v: i for i, v in enumerate(x_indices)}
        norm_x = norm_x[x_indices]
        connected_pool_indices = [
            np.array([global_to_x[pi] for pi in pool_indices if pi in global_to_x])
            for pool_indices in connected_pool_indices
        ]
        pools = [pools[i] for i in x_indices]
    if y_indices is not None:
        y_indices = np.arange(norm_y.shape[0])[y_indices]
        global_to_y = {v: i for i, v in enumerate(y_indices)}
        norm_y = norm_y[y_indices]
        connected_components = [
            np.array([global_to_y[c] for c in component if c in global_to_y])
            for component in connected_components
        ]
        pools = [
            np.array([global_to_y[p] for p in pool if p in global_to_y]) if pool is not None else None
            for pool in pools
        ]
    if all(p is None for p in pools):
        distance_matrix = torch.cdist(norm_x, norm_y, p=2) / 2
        rows, cols = torch.meshgrid(
            (torch.from_numpy(x_indices) if
             x_indices is not None else torch.arange(size_a)).to(device=distance_matrix.device),
            (torch.from_numpy(y_indices) if
             y_indices is not None else torch.arange(size_b)).to(device=distance_matrix.device),
            indexing='ij')
        sparse_indices = torch.stack([rows.flatten(), cols.flatten()], dim=0)
        return torch.sparse_coo_tensor(
            indices=sparse_indices,
            values=distance_matrix.flatten(),
            size=(size_a, size_b)
        ).coalesce()
    keep_component = np.array([
        pool.shape[0] > 0 and component.shape[0] > 0
        for pool, component in zip(connected_pool_indices, connected_components)
    ])
    filtered_connected_pool_indices = [None] * keep_component.sum()
    filtered_connected_components = [None] * keep_component.sum()
    filtered_compute_all_distances = np.zeros(keep_component.sum(), dtype=np.bool_)
    idx = 0
    for i, keep in enumerate(keep_component):
        if keep:
            filtered_connected_pool_indices[idx] = connected_pool_indices[i]
            filtered_connected_components[idx] = connected_components[i]
            filtered_compute_all_distances[idx] = compute_all_distances[i]
            idx += 1
    row_indices, col_indices, dist_values = [], [], []
    for pool_indices, component, compute_all in zip(
            filtered_connected_pool_indices, filtered_connected_components, filtered_compute_all_distances
    ):
        this_x = norm_x[pool_indices]
        if compute_all:
            this_y = norm_y
            component = np.arange(norm_y.shape[0])
        else:
            this_y = norm_y[component]
        this_distances = torch.cdist(this_x, this_y, p=2) / 2
        if int_scale is not None:
            this_distances = (this_distances * int_scale).long()
        for i, p in enumerate(pool_indices):
            if pools[p] is not None:
                row_indices.append(torch.full((pools[p].shape[0],), p, dtype=torch.long, device=norm_x.device))
                col_indices.append(torch.from_numpy(pools[p]).to(device=norm_x.device))
                dist_values.append(this_distances[i][np.isin(component, pools[p])])
            else:
                row_indices.append(torch.full((component.shape[0],), p, dtype=torch.long, device=norm_x.device))
                col_indices.append(torch.arange(component.shape[0], device=norm_x.device))
                dist_values.append(this_distances[i])
    if len(row_indices) > 0:
        row_indices = torch.cat(row_indices)
        col_indices = torch.cat(col_indices)
        dist_values = torch.cat(dist_values)
    else:
        row_indices = torch.tensor([])
        col_indices = torch.tensor([])
        dist_values = torch.tensor([])
    distances = torch.sparse_coo_tensor(
        indices=torch.stack([row_indices, col_indices]),
        values=dist_values,
        size=(norm_x.shape[0], norm_y.shape[0]),
    ).coalesce()
    new_row_indices = (
        (torch.from_numpy(x_indices) if x_indices is not None else torch.arange(size_a)).to(device=distances.device)
    )[distances.indices()[0]]
    new_col_indices = (
        (torch.from_numpy(y_indices) if y_indices is not None else torch.arange(size_b)).to(device=distances.device)
    )[distances.indices()[1]]
    new_indices = torch.stack([new_row_indices, new_col_indices])
    distances = torch.sparse_coo_tensor(
        indices=new_indices,
        values=distances.values(),
        size=(size_a, size_b)
    ).coalesce()
    return distances


def swap_isna(isna, pools):
    pool_isna = np.array([p is None for p in pools])
    if pool_isna.shape[0] > 0 and not pool_isna.all() and pool_isna.any():
        if pool_isna.sum() <= isna.sum():
            isna_swap_in = (pool_isna & (~isna)).nonzero()[0]
            isna_swap_out = np.random.choice((isna & (~pool_isna)).nonzero()[0], size=isna_swap_in.shape[0])
            isna = isna.copy()
            isna[isna_swap_in] = True
            isna[isna_swap_out] = False
        else:
            isna = pool_isna
    return isna


def filter_na(device, isna, non_overlapping_groups, parent, pools, values):
    isna = swap_isna(isna, pools)
    notna = ~isna
    not_na_indices = torch.arange(values.shape[0])[notna]  # i -> e
    all_to_not_na_indices = torch.zeros(values.shape[0], dtype=torch.long) - 1  # e -> i
    all_to_not_na_indices[notna] = torch.arange(not_na_indices.shape[0])
    dist_x = torch.from_numpy(values[notna].astype(np.float32)).to(device)
    dist_y = torch.from_numpy(parent.astype(np.float32)).to(device)
    dist_x = F.normalize(dist_x, p=2, dim=1, eps=1e-6)
    dist_y = F.normalize(dist_y, p=2, dim=1, eps=1e-6)
    filtered_pools = [
        p for i, p in enumerate(pools) if notna[i]
    ]
    filtered_non_overlapping_groups = [
        all_to_not_na_indices[group][all_to_not_na_indices[group] >= 0].cpu().numpy()
        for group in non_overlapping_groups
    ]
    return dist_x, dist_y, filtered_non_overlapping_groups, filtered_pools, isna, notna


def find_connected_components(groups: List[np.ndarray], max_matrix_size: int) -> Tuple[
    List[np.ndarray], List[np.ndarray]
]:
    adjacency = defaultdict(set)
    all_nodes = set()

    for idx, group in enumerate(tqdm(groups, desc="Collect adjacency")):
        all_nodes.update(group)
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                adjacency[group[i]].add(group[j])
                adjacency[group[j]].add(group[i])

    visited = set()

    def bfs(start):
        queue = deque([start])
        component = []
        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                component.append(node)
                queue.extend(adjacency[node] - visited)
        return component

    component_indices = []
    for node in tqdm(all_nodes, desc="Collect results"):
        if node not in visited:
            component = bfs(node)
            component_indices.append(np.array(sorted(component)))

    width = max(g.max() for g in groups if g.shape[0] > 0) + 1
    chunk_size = max(1, max_matrix_size // width)
    def process_chunk(st, ed, gs):
        pool_matrix = np.zeros((ed - st, width), dtype=np.bool_)
        for i, group in enumerate(gs):
            pool_matrix[i, group] = True
        chunk_results = []
        for i, component in enumerate(component_indices):
            chunk_results.append(np.nonzero(pool_matrix[:, component].any(axis=1))[0] + st)
        return chunk_results
    group_indices = [[] for _ in component_indices]
    chunk_results = Parallel(n_jobs=10)(
        delayed(process_chunk)(st, min(st + chunk_size, len(groups)), groups[st:min(st + chunk_size, len(groups))])
        for st in tqdm(range(0, len(groups), chunk_size), desc="Collect indices")
    )
    for chunk in chunk_results:
        for i, indices in enumerate(chunk):
            group_indices[i].append(indices)
    group_indices = [np.concatenate(x) for x in group_indices]

    return component_indices, group_indices


def swap_degrees(component_min_size, connected_components, degrees, component_pred_degrees):
    n_need_to_swap = np.clip(component_min_size - component_pred_degrees, 0, None).sum()
    if n_need_to_swap > 0:
        warnings.warn(f"Need to swap degrees: "
                      f"{np.clip(component_min_size - component_pred_degrees, 0, None).sum()}")
    for component_index in tqdm(range(len(connected_components)), "Swap degrees", total=len(connected_components)):
        pool, min_size, pred_size = (
            connected_components[component_index], component_min_size[component_index], component_pred_degrees[component_index]
        )
        if pred_size >= min_size:
            continue
        pool_indices = np.random.choice(
            np.arange(len(connected_components)).repeat(
                np.clip(component_pred_degrees - component_min_size, 0, None)
            ), size=min_size - pred_size, replace=False
        )
        vals, cnts = np.unique(pool_indices, return_counts=True)
        for pool_index, swap_cnt in zip(vals, cnts):
            swap_out = np.random.choice(
                connected_components[pool_index].repeat(degrees[connected_components[pool_index]]),
                size=swap_cnt, replace=False
            )
            swap_in = np.random.choice(pool, size=swap_cnt, replace=True)
            for out_index, out_cnt in zip(*np.unique(swap_out, return_counts=True)):
                degrees[out_index] -= out_cnt
            for in_index, in_cnt in zip(*np.unique(swap_in, return_counts=True)):
                degrees[in_index] += in_cnt
            component_pred_degrees[pool_index] -= swap_cnt
            component_pred_degrees[component_index] += swap_cnt
    if (degrees < 0).any():
        raise RuntimeError(f"Invalid swapping degree result: {degrees}.")
    if (component_pred_degrees < 0).any():
        raise RuntimeError("Invalid swapping degree result on components.")
    return degrees, component_pred_degrees


def collect_connected_components(degrees, dist_x, filtered_pools, max_matrix_size):
    if any(pool is not None for pool in filtered_pools):
        connected_components, connected_pool_indices = find_connected_components(
            [pool if pool is not None else np.array([], dtype=np.int32) for pool in filtered_pools], max_matrix_size
        )
        mentioned_in_pools = np.zeros_like(degrees, dtype=np.bool_)
        for pool in connected_components:
            mentioned_in_pools[np.array(pool, dtype=np.int32)] = True
        if not all(pool is not None for pool in filtered_pools):
            connected_components.append(np.arange(degrees.shape[0])[~mentioned_in_pools])
            connected_pool_indices.append(np.array([i for i, pool in enumerate(filtered_pools) if pool is None]))
            compute_all_distances = np.zeros(len(connected_components), dtype=np.bool_)
            compute_all_distances = np.r_[compute_all_distances, True]
        else:
            for idle_parent in np.arange(degrees.shape[0])[~mentioned_in_pools]:
                connected_components.append(np.array([idle_parent]))
                connected_pool_indices.append(np.array([], dtype=np.int32))
            compute_all_distances = np.zeros(len(connected_components), dtype=np.bool_)
    else:
        connected_components = [np.arange(degrees.shape[0])]
        connected_pool_indices = [np.arange(dist_x.shape[0])]
        compute_all_distances = np.array([False])
    component_min_size = np.array([len(x) for x in connected_pool_indices])
    pred_degrees = np.array([degrees[c].sum() for c in connected_components])
    row_component_match = np.zeros(dist_x.shape[0], dtype=np.int32) - 1
    for i, pool in enumerate(connected_pool_indices):
        row_component_match[pool] = i
    col_component_match = np.zeros(degrees.shape[0], dtype=np.int32) - 1
    for i, component in enumerate(connected_components):
        col_component_match[component] = i
    if (row_component_match < 0).any() or (col_component_match < 0).any():
        raise RuntimeError("Match not found.")
    return (
        component_min_size, compute_all_distances, connected_components, connected_pool_indices, pred_degrees,
        row_component_match, col_component_match
    )


def sample_parent_degrees(
        unique_groups: List[np.array], remaining_degrees: np.array,
        connected_components: List[np.array], row_component_match: np.array, pools: List[Optional[np.ndarray]]
) -> Tuple[np.array, np.array, np.array]:
    selected_degrees = np.zeros_like(remaining_degrees)
    min_unique_per_component = np.zeros(len(connected_components), dtype=np.int32)
    for group in unique_groups:
        vals, cnts = np.unique(row_component_match[group], return_counts=True)
        group_min_unique_per_component = np.zeros(len(connected_components), dtype=np.int32)
        group_min_unique_per_component[vals] = cnts
        min_unique_per_component = np.maximum(min_unique_per_component, group_min_unique_per_component)
    row_indices = np.concatenate(unique_groups)
    relaxed_degrees = np.array([], dtype=np.int32)
    row_component_indices = row_component_match[row_indices]
    for ci, cc in zip(*np.unique(row_component_indices, return_counts=True)):
        pool = connected_components[ci]
        relevant_rows = row_indices[row_component_indices == ci]
        appeared_values = None
        for ri in relevant_rows:
            if pools[ri] is not None:
                if appeared_values is None:
                    appeared_values = pools[ri]
                else:
                    appeared_values = np.union1d(pools[ri], appeared_values)
            else:
                appeared_values = np.arange(remaining_degrees.shape[0])
        if appeared_values is not None:
            pool = np.intersect1d(pool, appeared_values)
        min_unique = min_unique_per_component[ci]
        allowed_pool = np.intersect1d(pool, remaining_degrees.nonzero()[0])
        if allowed_pool.shape[0] > min_unique:
            selected_pool = np.random.choice(
                allowed_pool, size=min_unique, replace=False
            )
        elif allowed_pool.shape[0] == min_unique:
            selected_pool = allowed_pool
        else:
            remaining_choices = np.setdiff1d(
                pool, remaining_degrees.nonzero()[0]
            )
            relaxed_degrees = np.random.choice(
                remaining_choices, size=min(min_unique - allowed_pool.shape[0], remaining_choices.shape[0]),
                replace=False
            )
            selected_pool = np.sort(np.concatenate([allowed_pool, relaxed_degrees]))
            warnings.warn(f"Degrees violated: {relaxed_degrees.shape[0]} relaxed.")
        unique_selected = np.zeros_like(remaining_degrees)
        unique_selected[selected_pool] = 1
        sampled = np.random.choice(
            pool.repeat(np.clip(remaining_degrees[pool] - unique_selected[pool], 0, None)),
            size=cc - min_unique, replace=False
        )
        for v, c in zip(*np.unique(sampled, return_counts=True)):
            selected_degrees[v] = c
        selected_degrees[selected_pool] += 1
    selected_parents = np.arange(selected_degrees.shape[0])[selected_degrees > 0]
    return selected_parents, selected_degrees[selected_parents], relaxed_degrees


def validate_matching(
        values: np.ndarray, parent: np.ndarray, degrees: np.ndarray, isna: np.ndarray,
        pools: List[Optional[np.ndarray]], non_overlapping_groups: List[np.ndarray],
        matched: np.array
):
    if np.nanmax(matched) >= parent.shape[0]:
        raise RuntimeError("Unexpected parent index is matched.")
    if matched.shape[0] != values.shape[0]:
        raise RuntimeError("Some matches are not found.")
    n_pool_vio = 0
    for i, pool in enumerate(pools):
        if pool is None:
            continue
        if isna[i] and not np.isnan(matched[i]):
            raise ValueError("NaN not matched.")
        elif not isna[i] and not np.isin(matched[i], pool) and not matched[i] < 0:
            n_pool_vio += 1
    if n_pool_vio:
        raise ValueError(f"Pool violated {n_pool_vio}/{len(pools)}")
    for group in non_overlapping_groups:
        successful_matches = (~np.isnan(matched[group].astype(np.float64))) & (matched[group] >= 0)
        if len(np.unique(matched[group][successful_matches])) != successful_matches.sum():
            raise ValueError("Non-overlapping not satisfied.")
    vals, cnts = np.unique(matched[(~np.isnan(matched.astype(np.float64))) & (matched >= 0)], return_counts=True)
    deg_diff = degrees[vals.astype(np.int32)] - cnts
    if (deg_diff < 0).any():
        raise ValueError("Degree violated.")
    print("Checked, correct!!")


def match_trial(
        values: np.ndarray, parent: np.ndarray, degrees: np.ndarray, isna: np.ndarray,
        pools: List[Optional[np.ndarray]], non_overlapping_groups: List[np.ndarray],
        int_scale: float = 10_000, max_matrix_size: int = 100_000_000,
) -> np.ndarray:
    degrees = degrees.astype(np.int64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dist_x, dist_y, filtered_non_overlapping_groups, filtered_pools, isna, notna = filter_na(
        device, isna, non_overlapping_groups, parent, pools, values
    )

    (component_min_size, compute_all_distances, connected_components,
     connected_pool_indices, pred_degrees, row_component_match, col_component_match) = collect_connected_components(
        degrees, dist_x, filtered_pools, max_matrix_size)

    degrees, pred_degrees = swap_degrees(component_min_size, connected_components, degrees, pred_degrees)

    succeed = True
    pool_sizes = np.array([len(pool) if pool is not None else degrees.shape[0] for pool in filtered_pools])
    if filtered_non_overlapping_groups:
        # bipartite doesn't work
        lengths = [len(group) for group in filtered_non_overlapping_groups]
        # prioritize big groups
        lengths, filtered_non_overlapping_groups = zip(
            *sorted(zip(lengths, filtered_non_overlapping_groups), key=lambda x: -x[0])
        )
        lengths = np.array(list(lengths))
        filtered_non_overlapping_groups = list(filtered_non_overlapping_groups)
        if lengths.sum() != dist_x.shape[0]:
            raise RuntimeError(f"The non-overlapping groups is not equal to the size of values: "
                               f"{lengths.sum()} != {dist_x.shape[0]}.")
        remaining_degrees = degrees.copy()
        remaining_component_degrees = pred_degrees.copy()
        matched = np.full(dist_x.shape[0], -1)
        st = 0
        pbar = tqdm(desc="Match MCMF", total=dist_x.shape[0])
        mean_pool_size = pool_sizes.mean()
        while st < len(filtered_non_overlapping_groups):
            curr_idx_range = set()
            ed = st
            while len(curr_idx_range) * mean_pool_size < max_matrix_size and ed < len(filtered_non_overlapping_groups):
                group = filtered_non_overlapping_groups[ed]
                curr_idx_range |= set(group)
                ed += 1
            if not curr_idx_range:
                if ed == len(filtered_non_overlapping_groups):
                    break
                continue
            curr_idx_range = np.array([*curr_idx_range])
            smcf = SimpleMinCostFlow()
            src_idx, tgt_idx, curr_offset = 0, 1, 2
            parent_offset = curr_offset + dist_x.shape[0]
            comb_offset = parent_offset + dist_y.shape[0]
            smcf.add_arcs_with_capacity_and_unit_cost(
                np.full(curr_idx_range.shape[0], src_idx, np.int32),
                curr_idx_range + curr_offset,
                np.ones_like(curr_idx_range, np.int32), np.zeros_like(curr_idx_range, np.int32)
            )
            selected_parents, selected_degrees, relaxed_degrees = sample_parent_degrees(
                filtered_non_overlapping_groups[st:ed], remaining_degrees, connected_components, row_component_match,
                filtered_pools
            )
            remaining_degrees[relaxed_degrees] += 1
            degrees[relaxed_degrees] += 1
            rv, rc = np.unique(col_component_match[relaxed_degrees], return_counts=True)
            remaining_component_degrees[rv] += rc
            smcf.add_arcs_with_capacity_and_unit_cost(
                selected_parents + parent_offset,
                np.full_like(selected_parents, tgt_idx),
                selected_degrees, np.zeros_like(selected_parents)
            )

            def collect_group(gi, group):
                tails = []
                heads = []
                costs = []
                last_tails = []
                last_heads = []
                local_distances = compute_distances(
                    norm_x=dist_x, norm_y=dist_y, pools=filtered_pools,
                    x_indices=group, y_indices=selected_parents,
                    connected_components=connected_components, connected_pool_indices=connected_pool_indices,
                    compute_all_distances=compute_all_distances, int_scale=int_scale
                )
                for j, i in enumerate(selected_parents):
                    middle_idx = comb_offset + gi * selected_parents.shape[0] + j
                    dist_values = local_distances.transpose(0, 1)[i].coalesce()
                    indices = dist_values.indices()[0].cpu().numpy()
                    if indices.shape[0] <= 0:
                        continue
                    dist_values = dist_values.values().cpu().numpy()
                    tails.append(curr_offset + indices)
                    heads.append(np.full_like(indices, middle_idx))
                    costs.append(dist_values)
                    last_tails.append(middle_idx)
                    last_heads.append(parent_offset + i)
                tails.append(np.array(last_tails))
                heads.append(np.array(last_heads))
                tails = np.concatenate(tails)
                heads = np.concatenate(heads)
                capacities = np.ones_like(tails)
                costs.append(np.zeros_like(np.array(last_tails)))
                costs = np.concatenate(costs)
                return tails, heads, capacities, costs

            results = Parallel(n_jobs=10)(
                delayed(collect_group)(gi, group) for gi, group in enumerate(filtered_non_overlapping_groups[st:ed])
            )
            tails, heads, capacities, costs = [np.concatenate(x) for x in zip(*results)]
            if not tails.shape == heads.shape == capacities.shape == costs.shape:
                raise RuntimeError(
                    f"Shapes not matched: {tails.shape}, {heads.shape}, {capacities.shape}, {costs.shape}."
                )
            smcf.add_arcs_with_capacity_and_unit_cost(tails, heads, capacities, costs)

            smcf.set_node_supply(src_idx, curr_idx_range.shape[0])
            smcf.set_node_supply(tgt_idx, -curr_idx_range.shape[0])
            status = smcf.solve_max_flow_with_min_cost()
            max_flow = smcf.maximum_flow()
            if max_flow < curr_idx_range.shape[0] or status.name != "OPTIMAL":
                warnings.warn(
                    f"No solution found for MCMF "
                    f"at group {st}/{len(filtered_non_overlapping_groups)}: "
                    f"{max_flow}/({curr_idx_range.shape[0]},{selected_degrees.sum()}) [{status}({st}-{ed})]."
                )
                succeed = False
                break

            curr_to = {}
            to_parent = {}
            err_cnt = 0
            for i in range(smcf.num_arcs()):
                try:
                    src, dst, flow = smcf.tail(i), smcf.head(i), smcf.flow(i)
                except Exception as e:
                    err_cnt += 1
                    continue
                if flow <= 0:
                    continue
                if curr_offset <= src < parent_offset:
                    curr_to[src - curr_offset] = dst - comb_offset
                elif parent_offset <= dst < comb_offset:
                    to_parent[src - comb_offset] = dst - parent_offset

            if err_cnt / smcf.num_arcs() > 0.1:
                raise RuntimeError("Corrupted SMCF solution.")
            for i in curr_idx_range:
                matched[i] = to_parent[curr_to[i]]
                remaining_degrees[matched[i]] -= 1
                remaining_component_degrees[col_component_match[matched[i]]] -= 1
            pbar.update(curr_idx_range.shape[0])
            st = ed
    else:
        # bipartite is sufficient
        chunk_size = max(int(round(max_matrix_size / dist_y.shape[0])), 1000)
        matched = np.zeros(dist_x.shape[0]) - 1
        pbar = tqdm(desc="Match MCMBM", total=dist_x.shape[0])
        remaining_degrees = degrees.copy()
        for st in range(0, dist_x.shape[0], chunk_size):
            smcf = SimpleMinCostFlow()
            src_idx, tgt_idx, curr_offset = 0, 1, 2
            batch_indices = np.arange(st, min(st + chunk_size, dist_x.shape[0]))
            parent_offset = dist_x.shape[0] + curr_offset
            selected_parents, selected_degrees, relaxed_degrees = sample_parent_degrees(
                [np.array([x]) for x in batch_indices], remaining_degrees, connected_components, row_component_match,
                filtered_pools
            )
            remaining_degrees[relaxed_degrees] += 1
            degrees[relaxed_degrees] += 1
            local_distances = compute_distances(
                dist_x, dist_y, filtered_pools, connected_components, connected_pool_indices, compute_all_distances,
                batch_indices, selected_parents, int_scale
            )
            smcf.add_arcs_with_capacity_and_unit_cost(
                np.full(batch_indices.shape[0], src_idx),
                batch_indices + curr_offset,
                np.ones(batch_indices.shape[0]), np.zeros(batch_indices.shape[0]),
            )
            smcf.add_arcs_with_capacity_and_unit_cost(
                selected_parents + parent_offset,
                np.full_like(remaining_degrees, tgt_idx)[selected_parents],
                selected_degrees, np.zeros_like(remaining_degrees)[selected_parents]
            )
            smcf.add_arcs_with_capacity_and_unit_cost(
                (local_distances.indices()[0] + curr_offset).cpu().numpy(),
                (local_distances.indices()[1] + parent_offset).cpu().numpy(),
                np.ones(local_distances.indices()[0].shape[0], dtype=np.int64),
                local_distances.values().cpu().numpy(),
            )
            smcf.set_node_supply(src_idx, batch_indices.shape[0])
            smcf.set_node_supply(tgt_idx, -batch_indices.shape[0])
            status = smcf.solve_max_flow_with_min_cost()
            max_flow = smcf.maximum_flow()
            if max_flow < batch_indices.shape[0] or status.name != "OPTIMAL":
                warnings.warn(
                    f"No solution found for MCMBM "
                    f"at group {st}/{dist_x.shape[0]}: "
                    f"{max_flow}/{batch_indices.shape[0]} [{status}({st}:{chunk_size})]."
                )
                succeed = False
                break

            err_cnt = 0
            for i in range(smcf.num_arcs()):
                try:
                    src, dst, flow = smcf.tail(i), smcf.head(i), smcf.flow(i)
                except Exception:
                    err_cnt += 1
                    continue
                if flow <= 0 or src == src_idx or dst == tgt_idx:
                    continue
                if (not curr_offset <= src < parent_offset or
                        not parent_offset <= dst < parent_offset + degrees.shape[0]):
                    raise ValueError("Edge out of range.")
                matched[src - curr_offset] = dst - parent_offset
                remaining_degrees[dst - parent_offset] -= 1
            if err_cnt / smcf.num_arcs() > 0.1:
                raise RuntimeError(f"Too many errors: {err_cnt}/{smcf.num_arcs()}")
            pbar.update(batch_indices.shape[0])

    if succeed and (matched < 0).any():
        raise RuntimeError("Matched index isn't updated correctly.")
    if isna.any():
        placeholder = np.empty_like(isna, dtype=np.int32)
        placeholder[notna] = matched
        placeholder = placeholder.astype("object")
        placeholder[isna] = np.nan
        matched = placeholder

    validate_matching(
        values, parent, degrees, isna, pools, non_overlapping_groups, matched
    )
    print("Matched", (matched[notna] >= 0).sum(), "of", notna.sum())
    return matched


def match(*args, max_trials: int = 10, failure_allowed: bool = False, **kwargs) -> np.array:
    ee = None
    for _ in range(max_trials):
        matched = match_trial(*args, **kwargs)
        if not (matched[~np.isnan(matched.astype(np.float32))] < 0).any():
            return matched
    if ee is not None:
        if failure_allowed:
            warnings.warn(str(ee))
        else:
            raise ee
    if not failure_allowed:
        raise ValueError(f"No solution found: {matched}")
    return matched


def _match_test(
        embed_dim: int = 100, n_current: int = 132, n_parent: int = 35, n_na: int = 30, n_groups: int = 5,
        has_inequality: bool = True, has_overlapping: bool = True, has_unique: bool = True, unique_group_size: int = 3,
        max_matrix_size: int = 10,
):
    values = np.random.rand(n_current, embed_dim)
    parents = np.random.rand(n_parent, embed_dim)
    degrees = np.random.rand(n_parent)
    degrees.sort()
    degrees *= n_current / degrees[-1]
    placeholder = np.zeros_like(degrees)
    placeholder[0] = degrees[0].round()
    placeholder[1:] = np.diff(degrees.round())
    degrees = placeholder.astype(np.int32)
    isna_indices = np.random.permutation(n_current)[:n_na]
    isna = np.zeros(n_current, dtype=np.bool_)
    isna[isna_indices] = True
    current_to_match = np.random.randint(low=0, high=n_groups, size=n_current)
    parent_to_match = np.arange(n_parent) % n_groups
    per_match_parent = []
    for m in range(n_groups):
        per_match_parent.append(np.arange(n_parent)[parent_to_match == m])
    if has_overlapping:
        pools = [
            per_match_parent[c] for c in current_to_match
        ]
        if has_inequality:
            pools[0] = pools[0][1:]
            pools[1] = pools[1][:-1]
    else:
        if has_inequality:
            pools = [
                np.concatenate([np.arange(x), np.arange(x + 1, degrees.shape[0])])
                for x in np.random.randint(low=0, high=degrees.shape[0], size=current_to_match.shape[0])
            ]
        else:
            pools = [None for _ in current_to_match]
    if has_unique:
        non_overlapping_groups = [
            np.arange(st, min(st + unique_group_size, n_current), dtype=np.int32)
            for st in range(0, n_current, unique_group_size)
        ]
    else:
        non_overlapping_groups = []
    matched = match(
        values, parents, degrees, isna, pools, non_overlapping_groups,
        max_matrix_size=max_matrix_size,
    )

    if matched.shape[0] != n_current:
        raise ValueError("Matched error wrong")
    if not np.all(np.isnan(matched.astype(np.float64)) == isna):
        raise ValueError("is-na wrong")
    if matched[~isna].max() > n_parent:
        raise ValueError("Matched range wrong")
    for i, pool in enumerate(pools):
        if not isna[i] and pool is not None and matched[i] not in pool:
            raise ValueError("pool violated")
    for group in non_overlapping_groups:
        group_matched = matched[group]
        group_matched = group_matched[~np.isnan(group_matched.astype(np.float32))]
        if np.unique(group_matched).shape[0] != group_matched.shape[0]:
            raise ValueError("non overlapping violated")


if __name__ == "__main__":
    params = []
    for max_matrix_size in [100_000, 10]:
        for n_na in [0, 30]:
            for has_inequality in [True, False]:
                for has_overlapping in [True, False]:
                    for has_unique in [True, False]:
                        params.append({
                            "max_matrix_size": max_matrix_size,
                            "n_na": n_na,
                            "has_inequality": has_inequality,
                            "has_overlapping": has_overlapping,
                            "has_unique": has_unique,
                        })
    for j, p in enumerate(params):
        for i in range(5):
            print(f"Started: Trial {j}[{i}]: {p}")
            try:
                _match_test(**p)
            except Exception as e:
                print(f"Failed: Trial {j}[{i}]: {p}")
                raise e
            print(f"Succeeded: Trial {j}[{i}]: {p}")
