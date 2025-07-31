import argparse
import json
import os

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
    pre_parser.add_argument("--n-games", "-n", type=int, default=None)
    pre_parser.add_argument("--out-dir", "-o", default=os.path.join("."))

    post_parser = subparsers.add_parser('desimplify')
    post_parser.add_argument("--dataset-dir", "-d", default=os.path.join("data"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.op == "pre":
        table_names = ["teams", "players", "games", "gamestats", "appearances", "teamstats", "shots"]
        teams = pd.read_csv(os.path.join(args.dataset_dir, "teams.csv"))
        players = pd.read_csv(os.path.join(args.dataset_dir, "players.csv"), encoding="ISO-8859-1")
        games = pd.read_csv(os.path.join(args.dataset_dir, "games.csv"))
        gamestats = games.drop(columns=["homeTeamID", "awayTeamID"])
        games = games[["gameID", "homeTeamID", "awayTeamID"]]
        appearances = pd.read_csv(os.path.join(args.dataset_dir, "appearances.csv"))
        teamstats = pd.read_csv(os.path.join(args.dataset_dir, "teamstats.csv"))
        shots = pd.read_csv(os.path.join(args.dataset_dir, "shots.csv"))
        processors = {
            table: DataTransformer() for table in table_names
        }

        players = players.drop(columns=["name"])
        players["placeholder"] = 0
        teams = teams.drop(columns=["name"])
        teams["placeholder"] = 0
        joined_shots = shots.reset_index(drop=False).merge(
            appearances, left_on=["gameID", "assisterID"], right_on=["gameID", "playerID"], indicator="IND", how="left"
        ).set_index("index").loc[shots.index]
        shots["assisterID"] = joined_shots.apply(
            lambda row: row["assisterID"] if row["IND"] != "left_only" and row["assisterID"] != row["shooterID"]
            else np.nan, axis=1
        )
        if os.path.exists(os.path.join(args.out_dir, "processor.json")):
            with open(os.path.join(args.out_dir, "processor.json"), "r") as f:
                loaded = json.load(f)
                for table in table_names:
                    processors[table] = DataTransformer.from_dict(loaded[table])
        else:
            processors["teams"].fit(teams, ["teamID"])
            processors["players"].fit(players, ["playerID"])
            processors["gamestats"].fit(gamestats, ["gameID"])
            processors["teamstats"].fit(teamstats, ref_cols={
                "gameID": processors["gamestats"].columns["gameID"],
                "teamID": processors["teams"].columns["teamID"]
            })
            processors["games"].fit(games, ["gameID"], {
                "homeTeamID": processors["teamstats"].columns["teamID"],
                "awayTeamID": processors["teamstats"].columns["teamID"]
            })
            processors["appearances"].fit(appearances, ref_cols={
                "gameID": processors["games"].columns["gameID"],
                "playerID": processors["players"].columns["playerID"]
            })
            processors["shots"].fit(shots, ref_cols={
                "gameID": processors["appearances"].columns["gameID"],
                "shooterID": processors["appearances"].columns["playerID"],
                "assisterID": processors["appearances"].columns["playerID"]
            })
            with open(os.path.join(args.out_dir, "processor.json"), "w") as f:
                json.dump({
                    t: p.to_dict() for t, p in processors.items()
                }, f, indent=2)
        teams = processors["teams"].transform(teams)
        players = processors["players"].transform(players)
        gamestats = processors["gamestats"].transform(gamestats)
        games = processors["games"].transform(games)
        appearances = processors["appearances"].transform(appearances)
        teamstats = processors["teamstats"].transform(teamstats)
        shots = processors["shots"].transform(shots)

        os.makedirs(args.out_dir, exist_ok=True)
        os.makedirs(os.path.join(args.out_dir, "preprocessed"), exist_ok=True)
        for table in table_names:
            locals()[table].to_csv(os.path.join(args.out_dir, f"preprocessed/{table}.csv"), index=False)

        sampled_data = sample_dataset(
            args, table_names, appearances, games, gamestats, players, shots, teams, teamstats
        )

        os.makedirs(os.path.join(args.out_dir, "simplified"), exist_ok=True)
        simplified = simplify_dataset(
            args, table_names, appearances, games, gamestats, players, shots, teams, teamstats
        )
        for table in table_names:
            simplified[table].to_csv(os.path.join(args.out_dir, f"simplified/{table}.csv"), index=False)
        simplified = simplify_dataset(
            args, table_names,
            *[sampled_data[x] for x in ["appearances", "games", "gamestats", "players", "shots", "teams", "teamstats"]]
        )
        if args.n_games is not None:
            os.makedirs(os.path.join(args.out_dir, f"simplified-sample-{args.n_games}"), exist_ok=True)
            for table in table_names:
                simplified[table].to_csv(
                    os.path.join(args.out_dir, f"simplified-sample-{args.n_games}/{table}.csv"), index=False
                )

    elif args.op == "desimplify":
        desimplify_dataset(args.dataset_dir)


def sample_dataset(args, table_names, appearances, games, gamestats, players, shots, teams, teamstats):
    if args.n_games is not None:
        r_idle_players = 1 - players["playerID"].isin(appearances["playerID"]).mean()
        n_idle_players = max(1, round(r_idle_players * players.shape[0]))
        games = games.sample(n=args.n_games).reset_index(drop=True)
        game_ids = games["gameID"]
        gamestats = gamestats[gamestats["gameID"].isin(game_ids)]
        appearances = appearances[appearances["gameID"].isin(game_ids)].reset_index(drop=True)
        appeared = players["playerID"].isin(appearances["playerID"])
        idle_players = players[~appeared].sample(n=max(n_idle_players, (~appeared).sum()))
        players = pd.concat([players[appeared], idle_players], axis=0).reset_index(drop=True)
        teamstats = teamstats[teamstats["gameID"].isin(game_ids)].reset_index(drop=True)
        shots = shots[shots["gameID"].isin(game_ids)].reset_index(drop=True)
        os.makedirs(os.path.join(args.out_dir, f"preprocessed-sample-{args.n_games}"), exist_ok=True)
        for table in table_names:
            locals()[table].to_csv(
                os.path.join(args.out_dir, f"preprocessed-sample-{args.n_games}/{table}.csv"), index=False
            )
    table_vars = {}
    for table in table_names:
        table_vars[table] = locals()[table]
    return table_vars


def simplify_dataset(args, table_names, appearances, games, gamestats, players, shots, teams, teamstats):
    appearances["appearanceID"] = (appearances["gameID"].astype(int).astype("string")
                                   + "___" + appearances["playerID"].astype(int).astype("string"))
    shots = shots.merge(
        appearances[["gameID", "playerID", "appearanceID"]].rename(columns={"appearanceID": "shootAppearanceID"}),
        left_on=["gameID", "shooterID"], right_on=["gameID", "playerID"], how="left"
    ).drop(columns=["playerID"])
    shots = shots.merge(
        appearances[["gameID", "playerID", "appearanceID"]].rename(columns={"appearanceID": "assistAppearanceID"}),
        left_on=["gameID", "assisterID"], right_on=["gameID", "playerID"], how="left"
    ).drop(columns=["playerID"])
    shots = shots.drop(columns=["gameID", "shooterID", "assisterID"])

    teamstats["teamplayID"] = (teamstats["gameID"].astype(int).astype("string")
                               + "___" + teamstats["teamID"].astype(int).astype("string"))
    games = games.merge(
        teamstats[["gameID", "teamID", "teamplayID"]].rename(columns={"teamplayID": "homeTeamPlayID"}),
        left_on=["gameID", "homeTeamID"], right_on=["gameID", "teamID"], how="left"
    ).drop(columns=["teamID"])
    games = games.merge(
        teamstats[["gameID", "teamID", "teamplayID"]].rename(columns={"teamplayID": "awayTeamPlayID"}),
        left_on=["gameID", "awayTeamID"], right_on=["gameID", "teamID"], how="left"
    ).drop(columns=["teamID"])
    games = games.drop(columns=["homeTeamID", "awayTeamID"])

    table_vars = {}
    for table in table_names:
        table_vars[table] = locals()[table]
    return table_vars


def desimplify_dataset(generated_dir):
    shots = pd.read_csv(os.path.join(generated_dir, "shots.csv"))
    appearances = pd.read_csv(os.path.join(generated_dir, "appearances.csv"))
    shots["index"] = shots.index
    merged_shots = shots.merge(
        appearances[["gameID", "playerID", "appearanceID"]].rename(columns={"playerID": "shooterID"}),
        left_on="shootAppearanceID", right_on="appearanceID", how="left"
    ).set_index("index").loc[shots.index].drop(columns=["shootAppearanceID", "appearanceID"])
    shots = merged_shots
    shots["index"] = shots.index
    shots = shots.merge(
        appearances[["playerID", "appearanceID"]].rename(columns={"playerID": "assisterID"}),
        left_on="assistAppearanceID", right_on="appearanceID", how="left"
    ).set_index("index").loc[shots.index].drop(columns=["assistAppearanceID", "appearanceID"])
    shots.to_csv(os.path.join(generated_dir, "shots.csv"), index=False)
    appearances.drop(columns=["appearanceID"]).to_csv(os.path.join(generated_dir, "appearances.csv"), index=False)

    games = pd.read_csv(os.path.join(generated_dir, "games.csv"))
    teamstats = pd.read_csv(os.path.join(generated_dir, "teamstats.csv"))
    games["index"] = games.index
    merged_games = games.merge(
        teamstats[["teamID", "teamplayID"]].rename(columns={"teamID": "homeTeamID"}),
        left_on="homeTeamPlayID", right_on="teamplayID", how="left",
    ).set_index("index").loc[games.index].drop(columns=["homeTeamPlayID", "teamplayID"])
    games = merged_games
    games["index"] = games.index
    games = games.merge(
        teamstats[["teamID", "teamplayID"]].rename(columns={"teamID": "awayTeamID"}),
        left_on="awayTeamPlayID", right_on="teamplayID", how="left"
    ).set_index("index").loc[games.index].drop(columns=["awayTeamPlayID", "teamplayID"])
    games.to_csv(os.path.join(generated_dir, "games.csv"), index=False)
    teamstats.drop(columns=["teamplayID"]).to_csv(os.path.join(generated_dir, "teamstats.csv"), index=False)


if __name__ == "__main__":
    main()
