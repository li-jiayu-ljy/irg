import argparse
import json
import os
import shutil

import pandas as pd

try:
    from preprocessor import DataTransformer
    from baselines.ClavaDDPM.preprocess_utils import topological_sort
except (ModuleNotFoundError, ImportError):
    import importlib
    import sys
    base_dir = os.path.dirname(__file__)
    full_path = os.path.abspath(os.path.join(base_dir, "..", "..", "preprocessor.py"))
    spec = importlib.util.spec_from_file_location("preprocessor", full_path)
    preprocessor = importlib.util.module_from_spec(spec)
    sys.modules["preprocessor"] = preprocessor
    spec.loader.exec_module(preprocessor)
    DataTransformer = preprocessor.DataTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='op')

    pre_parser = subparsers.add_parser('pre')
    pre_parser.add_argument("--dataset-dir", "-d", default=os.path.join("data"))
    pre_parser.add_argument("--out-dir", "-o", default=os.path.join("."))

    post_parser = subparsers.add_parser('desimplify')
    post_parser.add_argument("--dataset-dir", "-d", default=os.path.join("data"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.op == "pre":
        table_names = [
            "customers", "geolocation", "order_items", "order_payments", "order_reviews",
            "orders", "products", "sellers"
        ]
        tables = {
            table_name: pd.read_csv(os.path.join(args.dataset_dir, f"olist_{table_name}_dataset.csv"))
            for table_name in table_names
        }
        geolocation = tables["geolocation"].drop(columns=["geolocation_city"])
        grouped = geolocation.groupby("geolocation_zip_code_prefix")
        state = grouped[["geolocation_state"]].first()
        location = grouped[["geolocation_lat", "geolocation_lng"]].mean()
        geolocation = pd.concat([state, location], axis=1).reset_index(drop=False)
        customers = tables["customers"].drop(columns=["customer_unique_id", "customer_city"])
        customers = customers[
            customers["customer_zip_code_prefix"].isin(geolocation["geolocation_zip_code_prefix"])
        ].reset_index(drop=True)
        sellers = tables["sellers"].drop(columns=["seller_city"])
        sellers = sellers[
            sellers["seller_zip_code_prefix"].isin(geolocation["geolocation_zip_code_prefix"])
        ].reset_index(drop=True)
        orders = tables["orders"]
        orders = orders[orders["customer_id"].isin(customers["customer_id"])].reset_index(drop=True)
        products = tables["products"]
        order_items = tables["order_items"]
        order_items = order_items[order_items["order_id"].isin(orders["order_id"])]
        order_items = order_items[order_items["seller_id"].isin(sellers["seller_id"])].reset_index(drop=True)
        order_payments = tables["order_payments"]
        order_payments = order_payments[order_payments["order_id"].isin(orders["order_id"])].reset_index(drop=True)
        order_payments = order_payments.sort_values(["order_id", "payment_sequential"])
        order_reviews = tables["order_reviews"].drop(columns=["review_comment_title", "review_comment_message"])
        order_reviews = order_reviews[order_reviews["order_id"].isin(orders["order_id"])].reset_index(drop=True)
        order_reviews = order_reviews.groupby("review_id").head(1).reset_index(drop=True)

        processors = {
            table: DataTransformer() for table in table_names
        }
        if os.path.exists(os.path.join(args.out_dir, "processor.json")):
            with open(os.path.join(args.out_dir, "processor.json"), "r") as f:
                loaded = json.load(f)
                for table in table_names:
                    processors[table] = DataTransformer.from_dict(loaded[table])
        else:
            processors["geolocation"].fit(geolocation, ["geolocation_zip_code_prefix"])
            processors["products"].fit(products, ["product_id"])
            processors["customers"].fit(customers, ["customer_id"], {
                "customer_zip_code_prefix": processors["geolocation"].columns["geolocation_zip_code_prefix"]
            })
            processors["sellers"].fit(sellers, ["seller_id"], {
                "seller_zip_code_prefix": processors["geolocation"].columns["geolocation_zip_code_prefix"],
            })
            processors["orders"].fit(orders, ["order_id"], {
                "customer_id": processors["customers"].columns["customer_id"],
            })
            processors["order_items"].fit(order_items, ref_cols={
                "order_id": processors["orders"].columns["order_id"],
                "product_id": processors["products"].columns["product_id"],
                "seller_id": processors["sellers"].columns["seller_id"],
            })
            processors["order_payments"].fit(order_payments, ref_cols={
                "order_id": processors["orders"].columns["order_id"],
            })
            processors["order_reviews"].fit(order_reviews, ref_cols={
                "order_id": processors["orders"].columns["order_id"],
            })
            with open(os.path.join(args.out_dir, "processor.json"), "w") as f:
                json.dump({
                    t: p.to_dict() for t, p in processors.items()
                }, f, indent=2)

        os.makedirs(args.out_dir, exist_ok=True)
        os.makedirs(os.path.join(args.out_dir, "preprocessed"), exist_ok=True)
        for table in table_names:
            transformed = processors[table].transform(locals()[table])
            transformed.to_csv(os.path.join(args.out_dir, f"preprocessed/{table}.csv"), index=False)

        shutil.copytree(os.path.join(args.out_dir, "preprocessed"), os.path.join(args.out_dir, "simplified"))

    elif args.op == "desimplify":
        pass


if __name__ == "__main__":
    main()
