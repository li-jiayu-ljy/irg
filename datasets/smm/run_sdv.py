import argparse
import json
import os
import time

import pandas as pd
from sdv.metadata import MultiTableMetadata
from sdv.multi_table import HMASynthesizer

try:
    from baselines.ind.synthesizer import IndependentSynthesizer
except (ModuleNotFoundError, ImportError):
    import importlib
    import sys
    base_dir = os.path.dirname(__file__)
    full_path = os.path.abspath(os.path.join(base_dir, "..", "..", "baselines", "ind", "synthesizer.py"))
    spec = importlib.util.spec_from_file_location("synthesizer", full_path)
    synthesizer = importlib.util.module_from_spec(spec)
    sys.modules["synthesizer"] = synthesizer
    spec.loader.exec_module(synthesizer)
    IndependentSynthesizer = synthesizer.IndependentSynthesizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", "-d", type=str, default="./simplified")
    parser.add_argument("--scale", "-s", type=float, default=1.0)
    parser.add_argument("--output-dir", "-o", type=str, default="./output")
    parser.add_argument("--model", "-m", choices=["hma", "ind"], default="hma")
    return parser.parse_args()


def main():
    args = parse_args()
    table_names = ["players", "courses", "course_maker", "plays", "clears", "likes", "records", "course_meta"]
    all_tables = {t: pd.read_csv(os.path.join(args.dataset_dir, f"{t}.csv")) for t in table_names}
    meta = MultiTableMetadata()
    meta.detect_from_dataframes(all_tables)
    meta.update_column("plays", "player", sdtype="id")
    meta.update_column("course_maker", "maker", sdtype="id")
    meta.update_column("course_meta", "firstClear", sdtype="id")
    meta.remove_primary_key("clears")
    meta.remove_primary_key("likes")
    meta.remove_primary_key("records")
    meta.remove_primary_key("course_maker")
    meta.remove_relationship("players", "plays")
    meta.add_relationship("players", "plays", "id", "player")
    meta.remove_relationship("players", "course_meta")
    meta.add_relationship("players", "course_maker", "id", "maker")
    meta.remove_relationship("courses", "course_maker")
    meta.add_relationship("courses", "course_maker", "id", "id")
    meta.add_relationship("plays", "clears", "playID", "playID")
    meta.add_relationship("plays", "likes", "playID", "playID")
    meta.add_relationship("plays", "records", "playID", "playID")
    meta.add_relationship("plays", "course_meta", "playID", "firstClear")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(meta.to_dict(), f, indent=2)

    synthesizer = HMASynthesizer if args.model == "hma" else IndependentSynthesizer
    if os.path.exists(os.path.join(args.output_dir, "model.pkl")):
        model = synthesizer.load(os.path.join(args.output_dir, "model.pkl"))
    else:
        start_time = time.time()
        model = synthesizer(meta)
        model.fit(all_tables)
        end_time = time.time()
        model.save(os.path.join(args.output_dir, "model.pkl"))
        with open(os.path.join(args.output_dir, "timing.json"), 'w') as f:
            json.dump({"fit": end_time - start_time}, f, indent=2)

    with open(os.path.join(args.output_dir, "timing.json"), 'r') as f:
        timing = json.load(f)
    if "sample" not in timing:
        start_time = time.time()
        sampled = model.sample(args.scale)
        os.makedirs(os.path.join(args.output_dir, "generated"), exist_ok=True)
        for k, v in sampled.items():
            v.to_csv(os.path.join(args.output_dir, "generated", f"{k}.csv"), index=False)
        end_time = time.time()
        timing["sample"] = end_time - start_time
        with open(os.path.join(args.output_dir, "timing.json"), 'w') as f:
            json.dump(timing, f, indent=2)


if __name__ == "__main__":
    main()
