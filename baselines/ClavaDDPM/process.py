import argparse
import os
import re
import shutil
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import json_tricks as json

from preprocess_utils import topological_sort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdv-schema", "-s", type=str, required=True, help="SDV schema file")
    parser.add_argument("--output-dir", "-o", type=str, required=True, help="Output directory")
    subparsers = parser.add_subparsers(dest="op")

    pre_parser = subparsers.add_parser("pre")
    pre_parser.add_argument("--dataset-dir", "-d", type=str, required=True, help="Directory containing datasets")
    pre_parser.add_argument("--dataset-name", "-n", type=str, required=True, help="Dataset name")
    pre_parser.add_argument("--fast", "-f", default=False, action="store_true", help="Fast experiment or not")

    post_parser = subparsers.add_parser("post")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.op == "pre":
        preprocess(args)
    elif args.op == "post":
        postprocess(args)
    else:
        raise NotImplementedError(f"Unknown op: {args.op}")


def preprocess(args):
    with open(args.sdv_schema, "r") as f:
        schema = json.load(f)
    start_time = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "data"), exist_ok=True)
    n_copies = defaultdict(int)
    parents_fk = defaultdict(list)
    for fk in schema["relationships"]:
        n_copies[(fk["parent_table_name"], fk["child_table_name"])] += 1
        parents_fk[fk["child_table_name"]].append(fk)
    sum_n_copies = defaultdict(int)
    for (p, c), v in n_copies.items():
        sum_n_copies[p] = max(sum_n_copies[p], v)
    need_null_pk = defaultdict(list)
    renames = {}
    for table_name, table_args in schema["tables"].items():
        table = pd.read_csv(os.path.join(args.dataset_dir, f"{table_name}.csv"))
        table_renames = {}
        for c in table.columns:
            if (c.endswith("_id") and c not in [fk["child_foreign_key"] for fk in parents_fk[table_name]]
                    and c != schema["tables"][table_name].get("primary_key")):
                table_renames[c] = c.replace("_id", "_iidd")
        renames[table_name] = table_renames
        for fk in parents_fk[table_name]:
            fk["nullable"] = table[fk["child_foreign_key"]].isna().any()
            if fk["nullable"]:
                sizes = table[fk["child_foreign_key"]].value_counts(dropna=True)
                need_null_pk[fk["parent_table_name"]].append((
                    sizes.quantile(0.05), sizes.quantile(0.95), table[fk["child_foreign_key"]].isna().sum()
                ))
    parents = defaultdict(list)
    domains = {}
    n_na_all = {}
    for table_name, table_args in schema["tables"].items():
        domains[table_name] = {}
        table = pd.read_csv(os.path.join(args.dataset_dir, f"{table_name}.csv"))
        table_renames = renames[table_name]
        table = table.rename(columns=table_renames)
        for c, a in table_args["columns"].items():
            if a["sdtype"] == "id":
                continue
            domains[table_name][table_renames.get(c, c)] = {
                "size": len(table[table_renames.get(c, c)].unique()),
                "type": "discrete" if a["sdtype"] == "categorical" else "continuous",
            }
        primary_key = table_args.get("primary_key")
        if table_name in need_null_pk:
            min_n_na = 0
            max_n_na = np.inf
            for min_size, max_size, na_size in need_null_pk[table_name]:
                min_n_na = max(min_n_na, na_size / max_size)
                max_n_na = min(max_n_na, na_size / min_size)
            if max_n_na < min_n_na:
                raise RuntimeError("Number of NULLs cannot be inferred.")
            n_na = np.random.randint(np.floor(min_n_na), np.ceil(max_n_na) + 1)
            n_na_all[table_name] = n_na
            sampled_df = pd.DataFrame({
                c: table[c].sample(n=n_na).values for c in table.columns
            })
            sampled_df[primary_key] = np.char.add("NULL-KEY-", np.arange(n_na).astype(str))
            table["_isna_key"] = "notna"
            sampled_df["_isna_key"] = "isna"
            table = pd.concat([table, sampled_df[table.columns]], axis=0, ignore_index=True)
            domains[table_name]["_isna_key"] = {
                "size": 2, "type": "discrete"
            }
        index = defaultdict(int)
        for fk in parents_fk[table_name]:
            current_index = index[(fk["parent_table_name"], fk["child_table_name"])] + 1
            parent_name = fk["parent_table_name"]
            if current_index > 1:
                parent_name = f"{parent_name}{current_index}"
            if fk["nullable"]:
                isna = table[fk["child_foreign_key"]].isna()
                keys = pd.read_csv(os.path.join(args.output_dir, "data", f"{fk['parent_table_name']}.csv"))
                na_keys = keys[f"{fk['parent_table_name']}_id"][keys["_isna_key"] == "isna"]
                sampled_na_keys = na_keys.sample(n=isna.sum(), replace=True)
                table.loc[isna, fk["child_foreign_key"]] = sampled_na_keys.values
            table = table.rename(columns={fk["child_foreign_key"]: f"{parent_name}_id"})
            index[(fk["parent_table_name"], fk["child_table_name"])] += 1
            parents[table_name].append(parent_name)
        if primary_key is not None:
            if sum_n_copies[table_name] > 1:
                for i in range(sum_n_copies[table_name]):
                    if i == 0:
                        new_table = table.rename(columns={primary_key: f"{table_name}_id"})
                        new_table.to_csv(os.path.join(args.output_dir, "data", f"{table_name}.csv"), index=False)
                    else:
                        new_table = pd.DataFrame({
                            f"{table_name}_id": table[primary_key],
                            f"{table_name}{i + 1}_id": table[primary_key],
                        })
                        new_table.to_csv(os.path.join(args.output_dir, "data", f"{table_name}{i + 1}.csv"), index=False)
                        domains[f"{table_name}{i + 1}"] = {}
                        parents[f"{table_name}{i + 1}"].append(table_name)
            else:
                table = table.rename(columns={primary_key: f"{table_name}_id"})
                table.to_csv(os.path.join(args.output_dir, "data", f"{table_name}.csv"), index=False)
        else:
            for i in range(max(1, sum_n_copies[table_name])):
                if sum_n_copies[table_name] > 1 or i == 0:
                    new_table = table.copy()
                    new_table[f"{table_name}_id"] = np.arange(new_table.shape[0])
                    new_table.to_csv(os.path.join(args.output_dir, "data", f"{table_name}.csv"), index=False)
                else:
                    new_table = pd.DataFrame({
                        f"{table_name}_id": np.arange(table.shape[0]),
                        f"{table_name}{i + 1}_id": np.arange(table.shape[0]),
                    })
                    new_table.to_csv(os.path.join(args.output_dir, "data", f"{table_name}{i + 1}.csv"), index=False)
                    domains[f"{table_name}{i + 1}"] = {}
                    parents[f"{table_name}{i + 1}"].append(table_name)
    for table_name, table_domain in domains.items():
        with open(os.path.join(args.output_dir, "data", f"{table_name}_domain.json"), "w") as f:
            json.dump(table_domain, f, indent=2)
    children = defaultdict(list)
    for table_name, table_parents in parents.items():
        for parent in table_parents:
            children[parent].append(table_name)

    for table_name, child_tables in children.items():
        table = pd.read_csv(os.path.join(args.output_dir, "data", f"{table_name}.csv"))
        if not pd.api.types.is_numeric_dtype(table[f"{table_name}_id"].dtype):
            mapper = table[f"{table_name}_id"].reset_index().set_index(f"{table_name}_id")["index"].to_dict()
            table[f"{table_name}_id"] = table[f"{table_name}_id"].map(mapper)
            table.to_csv(os.path.join(args.output_dir, "data", f"{table_name}.csv"), index=False)
            for child in child_tables:
                child_table = pd.read_csv(os.path.join(args.output_dir, "data", f"{child}.csv"))
                child_table[f"{table_name}_id"] = child_table[f"{table_name}_id"].map(mapper)
                child_table.to_csv(os.path.join(args.output_dir, "data", f"{child}.csv"), index=False)

    all_tables = []
    for table_name in schema["tables"]:
        all_tables.append(table_name)
        if sum_n_copies[table_name] > 1:
            for i in range(1, sum_n_copies[table_name]):
                all_tables.append(f"{table_name}{i + 1}")
    meta = {"tables": {table_name: {"parents": [], "children": []} for table_name in all_tables}}
    for table_name, table_parents in parents.items():
        meta["tables"][table_name]["parents"] = table_parents
    for table_name, table_children in children.items():
        meta["tables"][table_name]["children"] = table_children
    sorted_order = topological_sort(meta["tables"])
    meta["relation_order"] = sorted_order
    with open(os.path.join(args.output_dir, "data", "dataset_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(__file__.replace("process.py", "configs/movie_lens.json")), "r") as f:
        configs = json.load(f)
    configs["general"]["data_dir"] = os.path.join(args.output_dir, "data")
    configs["general"]["exp_name"] = args.dataset_name
    configs["general"]["workspace_dir"] = os.path.join(args.output_dir, "workspace")
    if args.fast:
        configs["diffusion"]["iterations"] = 200
        configs["classifier"]["iterations"] = 200
        configs["diffusion"]["num_timesteps"] = 20
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(configs, f, indent=2)
    with open(os.path.join(args.output_dir, "process-config.json"), "w") as f:
        json.dump({
            "parent_fks": parents_fk,
            "has_null_pk": [*need_null_pk],
            "sum_n_copies": sum_n_copies,
            "renames": renames
        }, f, indent=2)

    end_time = time.time()
    with open(os.path.join(args.output_dir, "timing.json"), "w") as f:
        json.dump({"preprocess": end_time - start_time}, f, indent=2)


def postprocess(args):
    with open(args.sdv_schema, "r") as f:
        schema = json.load(f)
    start_time = time.time()
    with open(os.path.join(args.output_dir, "data", "dataset_meta.json"), "r") as f:
        dataset_meta = json.load(f)
    relation_order = dataset_meta["relation_order"]
    os.makedirs(os.path.join(args.output_dir, "generated"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "intermediate-generated"), exist_ok=True)
    for table_name in dataset_meta["tables"]:
        shutil.copyfile(
            os.path.join(args.output_dir, "workspace", table_name, "_final", f"{table_name}_synthetic.csv"),
            os.path.join(args.output_dir, "intermediate-generated", f"{table_name}.csv"),
        )

    for parent, child in relation_order:
        if parent is not None and re.fullmatch(r".*\d+", parent):
            base_parent = parent
            while re.fullmatch(r".*\d+", base_parent):
                base_parent = base_parent[:-1]
            child_table = pd.read_csv(os.path.join(args.output_dir, "intermediate-generated", f"{child}.csv"))
            parent_table = pd.read_csv(os.path.join(args.output_dir, "intermediate-generated", f"{parent}.csv"))
            base_parent_table = pd.read_csv(
                os.path.join(args.output_dir, "intermediate-generated", f"{base_parent}.csv")
            )
            child_table = child_table.merge(
                parent_table.rename(columns={f"{base_parent}_id": f"___{base_parent}_id"}),
                on=f"{parent}_id", how="left"
            )
            child_table = child_table.merge(
                base_parent_table[[f"{base_parent}_id"]].rename(columns={f"{base_parent}_id": f"___{base_parent}_id"}),
                on=f"___{base_parent}_id", how="left"
            ).drop(columns=[f"{parent}_id"]).rename(columns={f"___{base_parent}_id": f"{parent}_id"})
            child_table.to_csv(os.path.join(args.output_dir, "intermediate-generated", f"{child}.csv"), index=False)

    with open(os.path.join(args.output_dir, "process-config.json"), "r") as f:
        loaded = json.load(f)
    parents_fk = loaded["parent_fks"]
    has_null_pk = loaded["has_null_pk"]
    renames = loaded["renames"]
    for table_name, table_args in schema["tables"].items():
        table = pd.read_csv(os.path.join(args.output_dir, "intermediate-generated", f"{table_name}.csv"))
        primary_key = table_args.get("primary_key")
        if primary_key is not None:
            table = table.rename(columns={f"{table_name}_id": primary_key})
        if table_name in has_null_pk:
            table = table[table["_isna_key"] == "notna"].reset_index(drop=True).drop(columns=["_isna_key"])
        index = defaultdict(int)
        for fk in parents_fk[table_name]:
            current_index = index[(fk["parent_table_name"], fk["child_table_name"])] + 1
            parent_name = fk["parent_table_name"]
            if current_index > 1:
                parent_name = f"{parent_name}{current_index}"
            table = table.rename(columns={f"{parent_name}_id": fk["child_foreign_key"]})
            if fk["nullable"]:
                parent_na_key = pd.read_csv(
                    os.path.join(args.output_dir, "intermediate-generated", f"{fk['parent_table_name']}.csv")
                )
                parent_na_key = parent_na_key[parent_na_key["_isna_key"] == "isna"][f'{fk["parent_table_name"]}_id']
                table[fk["child_foreign_key"]] = table[fk["child_foreign_key"]].replace(parent_na_key.tolist(), np.nan)
            index[(fk["parent_table_name"], fk["child_table_name"])] += 1
        table = table.rename(columns={v: k for k, v in renames[table_name].items()})
        table.to_csv(os.path.join(args.output_dir, "generated", f"{table_name}.csv"), index=False)

    end_time = time.time()
    with open(os.path.join(args.output_dir, "timing.json"), "r") as f:
        timing = json.load(f)
    timing["postprocess"] = end_time - start_time
    with open(os.path.join(args.output_dir, "timing.json"), "w") as f:
        json.dump(timing, f, indent=2)


if __name__ == "__main__":
    main()
