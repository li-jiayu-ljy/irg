import argparse
import json
import os
import shutil

import numpy as np
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
    # pre_parser.add_argument("--n-games", "-n", type=int, default=None)
    pre_parser.add_argument("--out-dir", "-o", default=os.path.join("."))

    post_parser = subparsers.add_parser('desimplify')
    post_parser.add_argument("--dataset-dir", "-d", default=os.path.join("data"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.op == "pre":
        table_names = ["players", "courses", "course_maker", "plays", "clears", "likes", "records", "course_meta"]
        players = pd.read_csv(os.path.join(args.dataset_dir, "players.csv"), sep="\t")
        players = players.drop(columns=["image", "name"])
        courses = pd.read_csv(os.path.join(args.dataset_dir, "courses.csv"), sep="\t")
        courses = courses.drop(columns=["title", "thumbnail", "image"])
        plays = pd.read_csv(os.path.join(args.dataset_dir, "plays.csv"), sep="\t")
        clears = pd.read_csv(os.path.join(args.dataset_dir, "clears.csv"), sep="\t")
        likes = pd.read_csv(os.path.join(args.dataset_dir, "likes.csv"), sep="\t")
        records = pd.read_csv(os.path.join(args.dataset_dir, "records.csv"), sep="\t")
        course_meta = pd.read_csv(os.path.join(args.dataset_dir, "course-meta.csv"), sep="\t")

        all_plays = plays[["id", "player"]].apply(lambda row: "$$".join(row.tolist()), axis=1)
        clears = clears[
            clears[["id", "player"]].apply(lambda row: "$$".join(row.tolist()), axis=1).isin(all_plays)
        ].reset_index(drop=True)
        likes = likes[
            likes[["id", "player"]].apply(lambda row: "$$".join(row.tolist()), axis=1).isin(all_plays)
        ].reset_index(drop=True)
        records = records[
            records[["id", "player"]].apply(lambda row: "$$".join(row.tolist()), axis=1).isin(all_plays)
        ].reset_index(drop=True)
        course_meta = course_meta[
            course_meta["firstClear"].isna() |
            course_meta[["id", "firstClear"]].astype(str).apply(
                lambda row: "$$".join(row.tolist()), axis=1
            ).isin(clears[["id", "player"]].apply(lambda row: "$$".join(row.tolist()), axis=1))
        ]
        courses = courses[courses.maker.isna() | courses.maker.isin(players["id"])].reset_index(drop=True)
        course_maker = courses[["id", "maker"]]
        courses = courses.drop(columns=["maker"])

        processors = {
            table: DataTransformer() for table in table_names
        }
        if os.path.exists(os.path.join(args.out_dir, "processor.json")):
            with open(os.path.join(args.out_dir, "processor.json"), "r") as f:
                loaded = json.load(f)
                for table in table_names:
                    processors[table] = DataTransformer.from_dict(loaded[table])
        else:
            processors["players"].fit(players, ["id"])
            processors["courses"].fit(courses, ["id"])
            processors["course_maker"].fit(course_maker, ["id"], ref_cols={
                "maker": processors["players"].columns["id"],
                "id": processors["courses"].columns["id"]
            })
            processors["plays"].fit(plays, ref_cols={
                "id": processors["courses"].columns["id"],
                "player": processors["players"].columns["id"],
            })
            processors["clears"].fit(clears, ref_cols={
                "id": processors["courses"].columns["id"],
                "player": processors["players"].columns["id"],
            })
            processors["likes"].fit(likes, ref_cols={
                "id": processors["courses"].columns["id"],
                "player": processors["players"].columns["id"],
            })
            processors["records"].fit(records, ref_cols={
                "id": processors["courses"].columns["id"],
                "player": processors["players"].columns["id"],
            })
            processors["course_meta"].fit(course_meta, ref_cols={
                "id": processors["courses"].columns["id"],
                "firstClear": processors["players"].columns["id"],
            })
            with open(os.path.join(args.out_dir, "processor.json"), "w") as f:
                json.dump({
                    t: p.to_dict() for t, p in processors.items()
                }, f, indent=2)

        players = processors["players"].transform(players)
        courses = processors["courses"].transform(courses)
        course_maker = processors["course_maker"].transform(course_maker)
        plays = processors["plays"].transform(plays)
        clears = processors["clears"].transform(clears)
        likes = processors["likes"].transform(likes)
        records = processors["records"].transform(records)
        course_meta = processors["course_meta"].transform(course_meta)

        os.makedirs(args.out_dir, exist_ok=True)
        os.makedirs(os.path.join(args.out_dir, "preprocessed"), exist_ok=True)
        for table in table_names:
            locals()[table].to_csv(os.path.join(args.out_dir, f"preprocessed/{table}.csv"), index=False)

        os.makedirs(os.path.join(args.out_dir, "simplified"), exist_ok=True)
        plays, course_meta, (clears, likes, records) = simplify_dataset(
            plays, course_meta, clears, likes, records
        )
        for table in table_names:
            locals()[table].to_csv(os.path.join(args.out_dir, f"simplified/{table}.csv"), index=False)

    elif args.op == "desimplify":
        desimplify_dataset(args.dataset_dir)


def simplify_dataset(plays, course_meta, *other_tables):
    plays["playID"] = plays[["id", "player"]].apply(lambda row: "$$".join(row.tolist()), axis=1)
    course_meta["firstClear"] = course_meta.apply(
        lambda row: np.nan if pd.isna(row["firstClear"]) else f"{row['id']}$${row['firstClear']}", axis=1
    )
    new_other_tables = []
    for table in other_tables:
        table["playID"] = table[["id", "player"]].apply(lambda row: "$$".join(row.tolist()), axis=1)
        table = table.drop(columns=["id", "player"])
        new_other_tables.append(table)
    return plays, course_meta, new_other_tables


def desimplify_dataset(generated_dir):
    plays = pd.read_csv(os.path.join(generated_dir, "plays.csv"))
    course_meta = pd.read_csv(os.path.join(generated_dir, "course_meta.csv"))

    def _process_df(df_name):
        df = pd.read_csv(os.path.join(generated_dir, f"{df_name}.csv"))
        df["index"] = df.index
        merged_clears = df.merge(
            plays[["id", "player", "playID"]], on="playID", how="left"
        ).set_index("index").loc[df.index].drop(columns=["playID"])
        df = merged_clears
        df.to_csv(os.path.join(generated_dir, f"{df_name}.csv"), index=False)
    _process_df("clears")
    _process_df("likes")
    _process_df("records")

    course_meta["index"] = course_meta.index
    merged_course_meta = course_meta.merge(
        plays[["id", "player", "playID"]].rename(columns={"id": "play_course_id"}),
        right_on="playID", left_on="firstClear", how="left"
    ).set_index("index").loc[course_meta.index].drop(columns=["playID", "play_course_id", "player"])
    merged_course_meta.to_csv(os.path.join(generated_dir, "course_meta.csv"), index=False)


if __name__ == "__main__":
    main()
