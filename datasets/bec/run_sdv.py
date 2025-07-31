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
    table_names = [
        "geolocation", "customers", "products", "sellers", "orders", "order_items", "order_payments", "order_reviews",
    ]
    all_tables = {t: pd.read_csv(os.path.join(args.dataset_dir, f"{t}.csv")) for t in table_names}
    meta = MultiTableMetadata()
    meta.detect_from_dataframes(all_tables)
    meta.update_column("geolocation", "geolocation_zip_code_prefix", sdtype="id")
    meta.update_column("geolocation", "geolocation_state", sdtype="categorical")
    meta.update_column("customers", "customer_zip_code_prefix", sdtype="id")
    meta.update_column("customers", "customer_state", sdtype="categorical")
    meta.update_column("sellers", "seller_zip_code_prefix", sdtype="id")
    meta.update_column("sellers", "seller_state", sdtype="categorical")
    meta.add_relationship("geolocation", "customers", "geolocation_zip_code_prefix", "customer_zip_code_prefix")
    meta.add_relationship("geolocation", "sellers", "geolocation_zip_code_prefix", "seller_zip_code_prefix")
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
