import json
import os

import numpy as np
import pandas as pd

from .utils import (
    REaLTabFormer, fit_transform_rtf, inverse_transform_rtf, sort_column_importance, transform_rtf, update_epochs
)


def train_aggregated_information(
        context: np.ndarray, aggregated_info: np.ndarray, model_dir: str, max_main_dim: int = 300,
        **kwargs
):
    os.makedirs(model_dir, exist_ok=True)
    kwargs = update_epochs(context.shape[0], kwargs)
    context_df = fit_transform_rtf(context, model_dir, "ctx")
    agg_df = fit_transform_rtf(aggregated_info, model_dir, "agg")
    if not context_df.index.equals(agg_df.index):
        raise RuntimeError("Index mismatch.")
    main_dim = -1
    agg_df["_index"] = agg_df.index
    context_df["_index"] = context_df.index
    step_columns = [agg_df.drop(columns=["_index"]).columns.tolist()]
    if agg_df.shape[-1] - 1 > max_main_dim:
        main_dim = max_main_dim
        sorted_columns = sort_column_importance(agg_df.drop(columns=["_index"]))
        idx = main_dim
        step_columns = [sorted_columns[:idx]]
        while idx < len(sorted_columns):
            step_columns.append(sorted_columns[idx:idx + max_main_dim])
            idx += max_main_dim

    with open(os.path.join(model_dir, "model-info.json"), "w") as f:
        json.dump({
            "main_dim": main_dim, "step_columns": step_columns,
            "raw_columns": agg_df.drop(columns=["_index"]).columns.tolist(),
            "context_columns": context_df.columns.tolist()
        }, f, indent=2)
    model = REaLTabFormer(
        model_type="relational", **kwargs,
        checkpoints_dir=os.path.join(model_dir, "ckpts"), samples_save_dir=os.path.join(model_dir, "samples"),
        freeze_parent_model=False, output_max_length=None,
    )
    model.fit(agg_df[["_index", *step_columns[0]]], context_df, join_on="_index")
    model.save(os.path.join(model_dir, "final"))

    context_df = pd.concat([context_df, agg_df[step_columns[0]]], axis=1)
    for i in range(1, len(step_columns)):
        model = REaLTabFormer(
            model_type="relational", **kwargs,
            checkpoints_dir=os.path.join(model_dir, f"step-{i}-ckpts"),
            samples_save_dir=os.path.join(model_dir, f"step-{i}-samples"),
            freeze_parent_model=False, output_max_length=None,
        )
        model.fit(agg_df[["_index", *step_columns[i]]], context_df, join_on="_index")
        model.save(os.path.join(model_dir, f"step-{i}-final"))


def generate_aggregated_information(context: np.ndarray, model_dir: str, chunk_size: int = 50_000) -> np.ndarray:
    model = REaLTabFormer.load_from_dir(os.path.join(model_dir, "final"))
    with open(os.path.join(model_dir, "model-info.json"), "r") as f:
        loaded = json.load(f)
    context_df = transform_rtf(context, model_dir, "ctx")

    out = []
    for st in range(0, context.shape[0], chunk_size):
        out.append(generate_aggregated_information_chunk(
            context_df[st:st + chunk_size], model_dir, loaded, model
        ))
    return np.concatenate(out)


def generate_aggregated_information_chunk(
        context_df: pd.DataFrame, model_dir: str, loaded: dict, model: REaLTabFormer
) -> np.ndarray:
    context_df.index.name = "_index"
    context_df = context_df.reset_index()
    context_df = context_df[loaded["context_columns"]]

    batch_size = 1024
    while batch_size > 0:
        try:
            first_child_samples = model.sample(
                input_unique_ids=context_df["_index"],
                input_df=context_df.drop("_index", axis=1),
                gen_batch=batch_size,
            ).groupby(level=0).first()
            child_samples = pd.DataFrame(
                columns=first_child_samples.columns, index=pd.Index(context_df["_index"].values, name="_index")
            )
            first_sampled = pd.Series(False, index=pd.Index(context_df["_index"].values, name="_index"))
            child_samples.loc[first_child_samples.index] = first_child_samples
            first_sampled[first_child_samples.index] = True
            while not first_sampled.all():
                selected_context_df = context_df.set_index("_index")[~first_sampled]
                new_child_samples = model.sample(
                    input_unique_ids=selected_context_df.index,
                    input_df=selected_context_df,
                    gen_batch=batch_size
                ).groupby(level=0).first()
                first_sampled.loc[new_child_samples.index] = True
                child_samples.loc[new_child_samples.index] = new_child_samples
            break
        except Exception as e:
            if "memory" in str(e):
                batch_size //= 2
            else:
                raise e
    if batch_size == 0:
        raise RuntimeError("Out of memory.")

    step_columns = loaded["step_columns"]
    context_df = pd.concat([context_df, child_samples[step_columns[0]]], axis=1)
    out = child_samples
    for i in range(1, len(step_columns)):
        step_model = REaLTabFormer.load_from_dir(os.path.join(model_dir, f"step-{i}-final"))
        new_child_samples = step_model.sample(
            input_unique_ids=context_df["_index"], input_df=context_df.drop("_index", axis=1), gen_batch=64,
        ).groupby(level=0).head(1).set_index("_index").loc[context_df["_index"].values]
        out = pd.concat([out, new_child_samples], axis=1)
        context_df = pd.concat([context_df, new_child_samples[step_columns[i]]], axis=1)
    out = out[loaded["raw_columns"]]

    return inverse_transform_rtf(out, model_dir, "agg")
