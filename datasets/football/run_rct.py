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
    table_names = ["teams", "players", "gamestats", "teamstats", "games", "appearances", "shots"]
    all_tables = {t: pd.read_csv(os.path.join(args.dataset_dir, f"{t}.csv")) for t in table_names}
    start_time = time.time()
    all_tables["games"]["placeholder"] = 1
    null_appearances = pd.DataFrame([
        pd.Series({"appearanceID": f"NULL-KEY-{i}", "_isna_key": True} | {
            c: all_tables["appearances"][c].sample(1).values[0]
            for c in all_tables["appearances"].columns if c != "appearanceID"
        }) for i in range(9000)
    ])
    all_tables["appearances"]["_isna_key"] = False
    all_tables["appearances"] = pd.concat([all_tables["appearances"], null_appearances], axis=0, ignore_index=True)
    all_tables["appearances"] = all_tables["appearances"].astype({"_isna_key": str})
    shot_na = all_tables["shots"]["assistAppearanceID"].isna()
    all_tables["shots"]["assistAppearanceID"][shot_na] = np.char.add(
        "NULL-KEY-", np.random.randint(0, 9000, shot_na.sum()).astype(str)
    )
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
        if table_name == "appearances":
            sampled_table = sampled_table[
                sampled_table["_isna_key"] == "False"
                ].reset_index(drop=True).drop(columns=["_isna_key"])
        elif table_name == "shots":
            sampled_table["assistAppearanceID"] = sampled_table["assistAppearanceID"].replace(
                sampled["appearances"][sampled["appearances"]["_isna_key"] == "True"]["appearanceID"].tolist(), np.nan
            )
        elif table_name == "games":
            sampled_table = sampled_table.drop(columns=["placeholder"])
        end_time = time.time()
        postprocess_time += end_time - start_time
        sampled_table.to_csv(os.path.join(args.output_dir, "generated", f"{table_name}.csv"), index=False)
    track_times["postprocess"] = postprocess_time
    with open(os.path.join(args.output_dir, "timing.json"), "w") as f:
        json.dump(track_times, f, indent=2)


if __name__ == "__main__":
    main()

