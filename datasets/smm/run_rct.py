import argparse
import json
import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd
from rctgan import Metadata
from rctgan.relational import RCTGAN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", "-d", type=str, default="./simplified")
    parser.add_argument("--sdv-schema", "-s", type=str, default="./schema/sdv.json")
    parser.add_argument("--output-dir", "-o", type=str, default="./output")
    return parser.parse_args()


def main():
    args = parse_args()
    warnings.filterwarnings("ignore")
    os.makedirs(args.output_dir, exist_ok=True)
    table_names = ["players", "courses", "plays", "clears", "likes", "records", "course_meta"]
    all_tables = {t: pd.read_csv(os.path.join(args.dataset_dir, f"{t}.csv")) for t in table_names}
    start_time = time.time()
    null_player = pd.DataFrame([
        pd.Series({"id": f"NULL-KEY", "_isna_key": True} | {
            c: all_tables["players"][c].sample(1).values[0] for c in all_tables["players"].columns if c != "id"
        })
    ])
    all_tables["players"]["_isna_key"] = False
    all_tables["players"] = pd.concat([all_tables["players"], null_player], axis=0, ignore_index=True)
    all_tables["players"] = all_tables["players"].astype({"_isna_key": str})
    maker_na = all_tables["courses"]["maker"].isna()
    all_tables["courses"].loc[maker_na, "maker"] = "NULL-KEY"
    first_clear_na = all_tables["course_meta"]["firstClear"].isna()
    all_tables["course_meta"].loc[first_clear_na, "firstClear"] = "NULL-KEY"
    end_time = time.time()
    track_times = {"preprocess": end_time - start_time}
    with open(os.path.join(args.output_dir, "timing.json"), "w") as f:
        json.dump(track_times, f, indent=2)
    with open(args.sdv_schema, "r") as f:
        sdv_schema = json.load(f)
    meta = Metadata()
    for table in table_names:
        table_schema = sdv_schema["tables"][table]
        meta.add_table(table, all_tables[table], primary_key=table_schema.get("primary_key"))
    for fk in sdv_schema["relationships"]:
        meta.add_relationship(fk["parent_table_name"], fk["child_table_name"], fk["child_foreign_key"])
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(meta.to_dict(), f, indent=2)

    start_time = time.time()
    model = RCTGAN(meta)
    model.fit(all_tables)
    end_time = time.time()
    track_times["fit"] = end_time - start_time
    with open(os.path.join(args.output_dir, "timing.json"), "w") as f:
        json.dump(track_times, f, indent=2)
    with open(os.path.join(args.output_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    start_time = time.time()
    sampled = model.sample()
    end_time = time.time()
    track_times["sample"] = end_time - start_time
    with open(os.path.join(args.output_dir, "timing.json"), "w") as f:
        json.dump(track_times, f, indent=2)

    os.makedirs(os.path.join(args.output_dir, "generated"), exist_ok=True)
    postprocess_time = 0
    for table_name, sampled_table in sampled.items():
        start_time = time.time()
        if table_name == "players":
            sampled_table = sampled_table[
                sampled_table["_isna_key"] == "False"
            ].reset_index(drop=True).drop(columns=["_isna_key"])
        elif table_name == "courses":
            sampled_table["maker"] = sampled_table["maker"].replace(
                sampled["players"][sampled["players"]["_isna_key"] == "True"]["id"].tolist(), np.nan
            )
        elif table_name == "course_meta":
            sampled_table["firstClear"] = sampled_table["firstClear"].replace(
                sampled["players"][sampled["players"]["_isna_key"] == "True"]["id"].tolist(), np.nan
            )
        end_time = time.time()
        postprocess_time += end_time - start_time
        sampled_table.to_csv(os.path.join(args.output_dir, "generated", f"{table_name}.csv"), index=False)
    track_times["postprocess"] = postprocess_time
    with open(os.path.join(args.output_dir, "timing.json"), "w") as f:
        json.dump(track_times, f, indent=2)


if __name__ == "__main__":
    main()
