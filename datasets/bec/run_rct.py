import argparse
import json
import os
import pickle
import time
import warnings

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
    table_names = [
        "geolocation", "customers", "products", "sellers", "orders", "order_items", "order_payments", "order_reviews",
    ]
    all_tables = {t: pd.read_csv(os.path.join(args.dataset_dir, f"{t}.csv")) for t in table_names}
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
    track_times = {"fit": end_time - start_time}
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
    for table_name, sampled_table in sampled.items():
        sampled_table.to_csv(os.path.join(args.output_dir, "generated", f"{table_name}.csv"), index=False)


if __name__ == "__main__":
    main()
