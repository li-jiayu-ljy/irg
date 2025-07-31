import argparse
import json
import os
import shutil
import time
import warnings
from typing import Dict

import pandas as pd
import psutil
import torch
import yaml

from irg import TableConfig, IncrementalRelationalGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", default="config/sample.yaml", help="config file, default is a sample"
    )
    parser.add_argument(
        "--input-data-dir", "-i", required=True, help="input data directory, data are in TABLE_NAME.csv"
    )
    parser.add_argument("--output-path", "-o", default="./out", help="output directory")
    parser.add_argument("--actions", "-a", default=[], nargs="+", choices=["train", "gen"])
    return parser.parse_args()


def _validate_data_config(path: str, tables: Dict[str, TableConfig], descr: str):
    for tn, tc in tables.items():
        table = pd.read_csv(os.path.join(path, f"{tn}.csv"))
        if tc.primary_key is not None:
            if table[tc.primary_key].duplicated().any():
                raise ValueError(f"Primary key constraint {tc.primary_key} on {tn} is not fulfilled for {descr}.")
        for fk in tc.foreign_keys:
            parent = pd.read_csv(os.path.join(path, f"{fk.parent_table_name}.csv"))
            fk_str = f"{fk.child_table_name}{fk.child_column_names} -> {fk.parent_table_name}{fk.parent_column_names}"
            if parent[fk.parent_column_names].duplicated().any():
                raise ValueError(f"Foreign key {fk_str} uniqueness on parent is not fulfilled for {descr}.")
            if (parent.merge(
                table[fk.child_column_names].dropna(),
                    left_on=fk.parent_column_names, right_on=fk.child_column_names, how="outer", indicator="_merged"
            )["_merged"] == "right_only").any():
                raise ValueError(f"Foreign key {fk_str} validity is not fulfilled for {descr}.")
        for a, b in tc.inequality:
            if (table[a] == table[b].rename(columns={bb: aa for bb, aa in zip(b, a)})).all(axis=1).any():
                raise ValueError(f"Inequality [{a}, {b}] on {tn} is not fulfilled for {descr}.")


def main():
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["PYTHONWARNINGS"] = "ignore"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    mem = psutil.virtual_memory()
    if mem.available + mem.used < 0.95 * mem.total:
        raise RuntimeError(f"Memory not available: {mem.available:,}, {mem.free:,}, {mem.used:,}, {mem.total:,}")
    warnings.filterwarnings("ignore")
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    tables = {tn: TableConfig.from_dict(ta | {"name": tn}) for tn, ta in config["tables"].items()}

    if "train" in args.actions:
        config["tables"] = tables

        if os.path.exists(args.output_path):
            shutil.rmtree(args.output_path)
        _validate_data_config(args.input_data_dir, tables, "real")
        start_time = time.time()
        synthesizer = IncrementalRelationalGenerator(**config)
        table_paths = {
            tn: os.path.join(args.input_data_dir, f"{tn}.csv") for tn in tables
        }
        synthesizer.fit(table_paths, args.output_path)
        end_time = time.time()
        times = {"fit": end_time - start_time}
        with open(os.path.join(args.output_path, "timing.json"), "w") as f:
            json.dump(times, f, indent=2)
        torch.save(synthesizer, os.path.join(args.output_path, "synthesizer.pt"))
    else:
        synthesizer = torch.load(os.path.join(args.output_path, "synthesizer.pt"))
        with open(os.path.join(args.output_path, "timing.json"), "r") as f:
            times = json.load(f)

    if "gen" in args.actions:
        start_time = time.time()
        synthesizer.generate(os.path.join(args.output_path, "generated"), os.path.join(args.output_path, "model"))
        end_time = time.time()
        times["sample"] = end_time - start_time
        with open(os.path.join(args.output_path, "timing.json"), "w") as f:
            json.dump(times, f, indent=2)
        _validate_data_config(os.path.join(args.output_path, "generated"), tables, "synthetic")


if __name__ == '__main__':
    main()
