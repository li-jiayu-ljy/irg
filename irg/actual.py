import gc
import json
import os
import warnings
from contextlib import contextmanager
from itertools import chain
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch.cuda
from joblib import Parallel, delayed
from tqdm import tqdm

from .utils import (
    REaLTabFormer, fit_transform_rtf, inverse_transform_rtf, sort_column_importance, transform_rtf, update_epochs
)


def train_actual_values(
        context: np.ndarray, length: np.ndarray, values: np.ndarray, groups: List[np.ndarray], model_dir: str,
        max_n_cells: int = 500, max_main_dim: int = 500, max_extend_dim: int = 500, **kwargs
):
    kwargs = update_epochs(context.shape[0], kwargs)
    os.makedirs(model_dir, exist_ok=True)
    x = np.concatenate([context, length.reshape((-1, 1))], axis=1)
    context_df = fit_transform_rtf(x, model_dir, "ctx")
    del x, context
    gc.collect()
    child = fit_transform_rtf(values, model_dir, "core")
    index = np.empty(child.shape[0])
    for i, g in enumerate(groups):
        index[g] = i
    del values, groups
    gc.collect()
    child["_index"] = index
    context_df["_index"] = context_df.index

    main_dim = -1
    base_size = -1
    stride = -1
    n_ctx = -1
    step_columns = [child.drop(columns=["_index"]).columns.tolist()]
    if (child.shape[-1] - 1) * length.max() <= max_n_cells:
        train_rtf(child, context_df, model_dir, **kwargs)
    else:
        # learn main columns
        if child.shape[-1] - 1 > max_main_dim:
            main_dim = max_main_dim
            sorted_columns = sort_column_importance(child.drop(columns=["_index"]))
            idx = main_dim
            step_columns = [sorted_columns[:idx]]
            while idx < len(sorted_columns):
                step_columns.append(sorted_columns[idx:idx + max_extend_dim])
                idx += max_extend_dim
            base_size = max_n_cells // max_main_dim
        else:
            base_size = max_n_cells // (child.shape[-1] - 1)
        if base_size < length.max():
            train_rtf(
                child[["_index", *step_columns[0]]].groupby("_index").head(base_size),
                context_df, model_dir, "base-", **kwargs
            )
            stride = max(min(10, base_size), base_size // 5)
            n_ctx = min(5, base_size)

            def process_group(g, g_data, context_row):
                if g_data.shape[0] <= n_ctx:
                    return [], []

                context_list = []
                child_list = []
                for s in range(n_ctx, g_data.shape[0]):
                    if (s - n_ctx) % stride != 0:
                        continue

                    rolling_ctx = g_data.iloc[s - n_ctx:s].reset_index(drop=True).reset_index().melt(
                        id_vars="index", var_name="variable", value_name="value"
                    )
                    rolling_ctx = pd.Series(
                        rolling_ctx["value"].values,
                        index=rolling_ctx.apply(
                            lambda row: str(row["variable"]) + "-" + str(row["index"]), axis=1
                        ).values
                    )
                    full_ctx = pd.concat([
                        context_row.drop("_index"), rolling_ctx, pd.Series({"_index": f"{g}-{s}"})
                    ])
                    context_list.append(full_ctx)

                    rolling_next = g_data.iloc[s:s + base_size].copy()
                    rolling_next["_index"] = f"{g}-{s}"
                    child_list.append(rolling_next)

                return context_list, child_list

            results = Parallel(n_jobs=20)(
                delayed(process_group)(g, g_data, context_df.loc[g])
                for g, g_data in tqdm(child[["_index", *step_columns[0]]].groupby("_index"), "Construct next pred")
            )
            new_context_df = list(chain.from_iterable(r[0] for r in results))
            new_child = list(chain.from_iterable(r[1] for r in results))
            new_context_df = pd.DataFrame(new_context_df)
            new_child = pd.concat(new_child, axis=0, ignore_index=True)
            train_rtf(new_child, new_context_df, model_dir, **kwargs)
            del new_child, new_context_df, results
            gc.collect()
        else:
            base_size = -1
            train_rtf(child[["_index", *step_columns[0]]], context_df, model_dir, **kwargs)

        # learn extended columns
        context_df = context_df.set_index("_index").loc[child["_index"].values].reset_index(drop=True)
        child["_index"] = np.arange(child.shape[0])
        context_df["_index"] = np.arange(child.shape[0])
        context_df = pd.concat([context_df, child[step_columns[0]]], axis=1)
        for i in range(1, len(step_columns)):
            print(f"Fit step {i}/{len(step_columns)} ...")
            train_rtf(child[["_index", *step_columns[i]]], context_df, model_dir, f"step-{i}", **kwargs)
            context_df = pd.concat([context_df, child[step_columns[i]]], axis=1)
            child = child.drop(columns=step_columns[i])

    with open(os.path.join(model_dir, "model-info.json"), "w") as f:
        json.dump({
            "main_dim": main_dim, "step_columns": step_columns,
            "raw_columns": child.drop(columns=["_index"]).columns.tolist(),
            "base_size": base_size, "stride": stride, "n_ctx": n_ctx
        }, f, indent=2)


def train_rtf(child, context_df, model_dir, prefix="", **kwargs):
    if prefix and not prefix.endswith("-"):
        prefix = f"{prefix}-"
    cnt = 0
    learning_rate = kwargs.pop("learning_rate", 2e-4)
    while True:
        try:
            model = REaLTabFormer(
                model_type="relational", **kwargs,
                checkpoints_dir=os.path.join(model_dir, f"{prefix}ckpts"),
                samples_save_dir=os.path.join(model_dir, f"{prefix}samples"),
                freeze_parent_model=False, output_max_length=None, learning_rate=learning_rate
            )
            context_df = context_df.set_index("_index")
            context_df["_length"] = child.groupby("_index").size()
            context_df["_length"] = context_df["_length"].fillna(0)
            context_df = context_df.reset_index()
            model.fit(child, context_df, join_on="_index")
            if len([k for k, v in model.model.named_parameters() if not torch.isfinite(v).all()]) == 0:
                break
        except RuntimeError as e:
            if "diverge" not in str(e):
                raise e
        cnt += 1
        learning_rate /= 2
        if cnt > 5:
            raise RuntimeError("Model diverges.")
    model.save(os.path.join(model_dir, f"{prefix}final"))


def generate_actual_values(
        context: np.ndarray, length: np.ndarray, model_dir: str, max_cnt: int = 200, max_size: int = 5_000_000_000
) -> Tuple[np.ndarray, List[np.ndarray]]:
    x = np.concatenate([context, length.reshape((-1, 1))], axis=1)
    context_df = transform_rtf(x, model_dir, "ctx")
    length = pd.Series(length, dtype=np.int32)
    with open(os.path.join(model_dir, "model-info.json"), "r") as f:
        loaded = json.load(f)

    model = REaLTabFormer.load_from_dir(os.path.join(model_dir, "final"))
    all_data = []
    all_groups = []
    base_idx = 0
    chunk_size = int(
        round(max_size / (((loaded["n_ctx"] + loaded["base_size"]) if loaded["base_size"] > 0 else length.max())
                             * len(loaded["step_columns"][0])) + context_df.shape[-1])
    )
    for i, st in enumerate(range(0, context.shape[0], chunk_size)):
        g_data, g_groups, base_idx = generate_actual_values_chunk(
            context_df[st:st + chunk_size].reset_index(drop=True), length[st:st + chunk_size].reset_index(drop=True),
            max_cnt, model, model_dir, i, base_idx, loaded
        )
        all_data.append(g_data)
        all_groups.extend(g_groups)
    return np.concatenate(all_data, axis=0), all_groups


def generate_actual_values_chunk(
        context_df: pd.DataFrame, length: pd.Series, max_cnt: int, model: REaLTabFormer, model_dir: str,
        chunk_idx: int, base_idx: int, loaded: dict
) -> Tuple[np.ndarray, List[np.ndarray], int]:
    base_size = loaded["base_size"]
    n_ctx = loaded["n_ctx"]
    if base_size < 0:
        all_out, finished, violated, _ = generate_rtf(context_df, length, model, f"({chunk_idx})")
    else:
        base_model = REaLTabFormer.load_from_dir(os.path.join(model_dir, "base-final"))
        all_out, base_finished, violated, finished_length = generate_rtf(
            context_df, length.clip(upper=base_size), base_model, f"base ({chunk_idx})"
        )
        finished_length = np.array(finished_length)
        cnt = 0
        finished = pd.Series(finished_length >= length, index=context_df.index.values)
        long_max_cnt = max_cnt * np.ceil(length.max() / base_size)
        batch_size = 1024
        pbar = tqdm(
            desc=f"Other base ({chunk_idx}) [bs={batch_size}]", total=length.sum() - finished_length.sum()
        )
        while not finished.all():
            rest_context = context_df[~finished]
            prev = {
                g: all_out[g].iloc[-n_ctx:].reset_index(drop=True).reset_index().melt(id_vars="index")
                for g, f in finished.items() if not f
            }
            prev_ctx = pd.DataFrame([
                pd.Series(
                    p["value"].values,
                    index=p.apply(lambda row: str(row["variable"]) + "-" + str(row["index"]), axis=1).values, name=g
                ) for g, p in prev.items()
            ])
            rest_context = pd.concat([rest_context, prev_ctx], axis=1)
            rest_context["_length"] = pd.Series(
                (length - np.array(finished_length)), index=context_df.index
            )[~finished].clip(lower=0, upper=base_size)
            with disable_tqdm():
                next_child_samples = None
                ee = None
                while batch_size > 0:
                    try:
                        next_child_samples = model.sample(
                            input_unique_ids=rest_context.index,
                            input_df=rest_context,
                            gen_batch=batch_size,
                            related_num="_length"
                        ).groupby(level=0)
                        pbar.set_description(f"Other base ({chunk_idx}) [bs={batch_size}]")
                        break
                    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                        if isinstance(e, RuntimeError) and "out of memory" not in str(e):
                            raise e
                        batch_size //= 2
                        ee = e
                if next_child_samples is None:
                    raise ee
            next_sizes = next_child_samples.size()
            for g, g_data in next_child_samples:
                need = length[g] - finished_length[g]
                all_out[g] = pd.concat([all_out[g], g_data[:need]], axis=0, ignore_index=True)
                finished_length[g] += min(need, next_sizes[g])
                pbar.update(min(need, next_sizes[g]))
            finished = finished_length >= length
            cnt += 1
            if cnt > long_max_cnt:
                violated = True
                break
    if not finished.all():
        raise RuntimeError("Not all groups are generated.")
    groups = []
    idx = base_idx
    idx_to_grp = []
    for gi, go in enumerate(all_out):
        groups.append(np.arange(idx, idx + go.shape[0]))
        idx_to_grp.extend([gi] * go.shape[0])
        idx += go.shape[0]

    if violated:
        warnings.warn("Some lengths are violated.")

    out = pd.concat(all_out, ignore_index=True)

    step_columns = loaded["step_columns"]
    context_df = context_df.loc[idx_to_grp].set_axis(out.index, axis=0)
    context_df = pd.concat([context_df, out[step_columns[0]]], axis=1)
    out = out.reset_index(drop=True)
    context_df = context_df.reset_index(drop=True)
    for i in range(1, len(step_columns)):
        step_model = REaLTabFormer.load_from_dir(os.path.join(model_dir, f"step-{i}-final"))
        new_out, finished, violated, finished_length = generate_rtf(
            context_df, pd.Series(1, context_df.index), step_model, f"step-{i} ({chunk_idx})"
        )
        if violated or not finished.all() or (finished_length != 1).any():
            raise RuntimeError("Extended columns invalid.")
        new_out = pd.concat(new_out, ignore_index=True)
        out = pd.concat([out, new_out], axis=-1)
        context_df = pd.concat([context_df, new_out[step_columns[i]]], axis=1)
    out = out[loaded["raw_columns"]]
    return inverse_transform_rtf(out, model_dir, "core"), groups, idx


@contextmanager
def disable_tqdm():
    import tqdm
    import sys
    from contextlib import ExitStack
    from unittest.mock import patch
    from datasets.utils.logging import disable_progress_bar, enable_progress_bar
    orig_tqdm = tqdm.tqdm
    orig_auto_tqdm = sys.modules["tqdm.auto"].tqdm
    orig_std_tqdm = sys.modules["tqdm.std"].tqdm
    orig_notebook = sys.modules.get("tqdm.notebook", None)
    orig_notebook_tqdm = getattr(orig_notebook, "tqdm", None) if orig_notebook else None

    class SilentTqdm(orig_tqdm):
        def __init__(self, *args, **kwargs):
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)

    disable_progress_bar()

    # Force-update sys.modules
    sys.modules["tqdm"].tqdm = SilentTqdm
    sys.modules["tqdm.auto"].tqdm = SilentTqdm
    sys.modules["tqdm.std"].tqdm = SilentTqdm
    if orig_notebook:
        sys.modules["tqdm.notebook"].tqdm = SilentTqdm

    patches = [
        patch("tqdm.tqdm", SilentTqdm),
        patch("tqdm.auto.tqdm", SilentTqdm),
        patch("tqdm.std.tqdm", SilentTqdm),
    ]
    if orig_notebook_tqdm:
        patches.append(patch("tqdm.notebook.tqdm", SilentTqdm))

    import realtabformer
    for mod in [realtabformer.rtf_sampler]:
        patches.append(patch.object(mod, "tqdm", SilentTqdm))

    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        try:
            yield
        finally:
            sys.modules["tqdm"].tqdm = orig_tqdm
            sys.modules["tqdm.auto"].tqdm = orig_auto_tqdm
            sys.modules["tqdm.std"].tqdm = orig_std_tqdm
            if orig_notebook and orig_notebook_tqdm:
                sys.modules["tqdm.notebook"].tqdm = orig_notebook_tqdm
            enable_progress_bar()


def generate_rtf(context_df, length, model, descr):
    context_df = context_df.copy()
    context_df["_length"] = length
    all_out = [None] * context_df.shape[0]
    finished_length = [0] * context_df.shape[0]
    violated = False
    batch_size = 1024
    pbar = tqdm(desc=f"Generating {descr} [bs={batch_size}]", total=length.sum())
    with disable_tqdm():
        child_samples = None
        ee = None
        while batch_size > 0:
            try:
                pbar.set_description(f"Generating {descr} [bs={batch_size}, context={context_df.shape}]")
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    child_samples = model.sample(
                        input_unique_ids=context_df.index,
                        input_df=context_df,
                        gen_batch=batch_size,
                        related_num="_length"
                    ).groupby(level=0)
                break
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if isinstance(e, RuntimeError) and "out of memory" not in str(e):
                    raise e
                batch_size //= 2
                ee = e
        if child_samples is None:
            raise ee
    sizes = child_samples.size()
    for g, g_data in child_samples:
        if sizes[g] >= length[g]:
            if not violated and sizes[g] < length[g]:
                violated = True
            all_out[g] = g_data[:length[g]]
            finished_length[g] = all_out[g].shape[0]
            pbar.update(all_out[g].shape[0])
    return all_out, pd.Series(True, index=context_df.index.values), violated, finished_length
