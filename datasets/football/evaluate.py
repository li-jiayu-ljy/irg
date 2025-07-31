import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from evaluator import RelationalEvaluator, format_by_rank
except (ModuleNotFoundError, ImportError):
    import importlib
    import sys
    base_dir = os.path.dirname(__file__)
    full_path = os.path.abspath(os.path.join(base_dir, "..", "..", "evaluator.py"))
    spec = importlib.util.spec_from_file_location("evaluator", full_path)
    evaluator = importlib.util.module_from_spec(spec)
    sys.modules["evaluator"] = evaluator
    spec.loader.exec_module(evaluator)
    RelationalEvaluator = evaluator.RelationalEvaluator
    format_by_rank = evaluator.format_by_rank


class FootballEvaluator(RelationalEvaluator):
    hue_order = ["Real", "IRG", "IND", "RCTGAN", "CLD"]
    renames = {
        # "sdv": "HMA",
        "ind": "IND",
        "rctgan": "RCTGAN",
        "clava": "CLD",
        "irg": "IRG"
    }

    def __init__(self):
        super().__init__("football", models=["ind", "rctgan", "clava", "irg"])

    def evaluate_game_goals(self):
        fig, axes = plt.subplots(
            nrows=1, ncols=len(self.renames) + 1, figsize=(3 * (len(self.renames) + 1), 3), sharex=True, sharey=True
        )
        inverse_renames = {v: k for k, v in self.renames.items()}
        cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])
        all_diffs = {}
        max_diff = -1
        for m in self.hue_order:
            diff = self._extract_goals(self.real_path if m == "Real" else self.model_paths[inverse_renames[m]])
            all_diffs[m] = diff
            max_diff = max(max_diff, diff.max().max())
        for i, m in enumerate(self.hue_order):
            ax = axes[i]
            sns.heatmap(
                all_diffs[m], ax=ax, cmap="PuRd", cbar=i == 0, cbar_ax=cbar_ax if i == 0 else None,
                vmin=0, vmax=min(max_diff, 3)
            )
            ax.set_title(m)
        plt.tight_layout(rect=(0, 0, 0.9, 1))
        fig.savefig(os.path.join(self.report_path, "goals.png"))
        fig.savefig(os.path.join(self.report_path, "goals.pdf"))

    @staticmethod
    def _extract_goals(path: str):
        games = pd.read_csv(os.path.join(path, "gamestats.csv")).set_index("gameID")
        appearances = pd.read_csv(os.path.join(path, "appearances.csv"))
        appearances["totalGoals"] = appearances["goals"] + appearances["ownGoals"]
        shots = pd.read_csv(os.path.join(path, "shots.csv"))
        goal_by_game = games["homeGoals"] + games["awayGoals"]
        goal_by_team = pd.read_csv(os.path.join(path, "teamstats.csv")).groupby("gameID")["goals"].sum()
        goal_by_player = appearances.groupby("gameID")["totalGoals"].sum()
        goal_by_shots = shots[shots["shotResult"].isin(["Goal", "OwnGoal"])].groupby("gameID").size()
        all_goals = pd.DataFrame({
            "G": goal_by_game, "T": goal_by_team, "A": goal_by_player, "S": goal_by_shots
        })
        all_goals["S"].fillna(0, inplace=True)
        diff = pd.DataFrame(0, index=all_goals.columns, columns=all_goals.columns)
        for a in diff.index:
            for b in diff.index:
                diff.loc[a, b] = np.log10((all_goals[a].fillna(1e3/2) - all_goals[b].fillna(-1e3/2)).abs().mean() + 1)
        return diff

    @staticmethod
    def _get_query_results(path: str):
        shots = pd.read_csv(os.path.join(path, "shots.csv"))
        appearances = pd.read_csv(os.path.join(path, "appearances.csv"))
        shooter_valid = shots.set_index(["gameID", "shooterID"]).index.isin(
            appearances.set_index(["gameID", "playerID"]).index
        )
        assister_valid = shots["assisterID"].isna() | shots.set_index(["gameID", "assisterID"]).index.isin(
            appearances.set_index(["gameID", "playerID"]).index
        )
        ineq_valid = shots["shooterID"] != shots["assisterID"]
        shots = shots[shooter_valid & assister_valid & ineq_valid].reset_index(drop=True)
        type_and_time_diff = []
        for _, g_data in shots.groupby("gameID"):
            is_open_play = g_data["situation"] == "OpenPlay"
            g_data = g_data.copy()
            g_data["open-play"] = ~is_open_play
            g_data = g_data.sort_values(["minute", "open-play"])
            diff = g_data["minute"].diff()
            diff.fillna(g_data.iloc[0]["minute"])
            situation = g_data["open-play"].replace({True: "non open-play", False: "open-play"})
            g_df = pd.DataFrame({
                "Time Diff. (min)": diff + 1, "Situation": situation,
            })
            type_and_time_diff.append(g_df)
        type_and_time_diff = pd.concat(type_and_time_diff, ignore_index=True)

        teamstats = pd.read_csv(os.path.join(path, "teamstats.csv"))
        games = pd.read_csv(os.path.join(path, "games.csv"))
        home_valid = games.set_index(["gameID", "homeTeamID"]).index.isin(
            teamstats.set_index(["gameID", "teamID"]).index
        )
        away_valid = games.set_index(["gameID", "awayTeamID"]).index.isin(
            teamstats.set_index(["gameID", "teamID"]).index
        )
        ineq_valid = games["homeTeamID"] != games["awayTeamID"]
        games = games[home_valid & away_valid & ineq_valid].reset_index(drop=True)
        home_results = games.merge(
            teamstats, left_on=["gameID", "homeTeamID"], right_on=["gameID", "teamID"], how="left"
        )
        away_results = games.merge(
            teamstats, left_on=["gameID", "awayTeamID"], right_on=["gameID", "teamID"], how="left"
        )

        total_goals_per_game = teamstats.groupby('gameID')['goals'].sum()
        top_games = total_goals_per_game[total_goals_per_game >= total_goals_per_game.quantile(0.75)].index
        shots_top_games = shots[shots['gameID'].isin(top_games)]
        player_shots = shots_top_games.groupby('shooterID').size()

        return {
            "OP diff.": type_and_time_diff[type_and_time_diff["Situation"] == "open-play"]["Time Diff. (min)"].rename(),
            "Non-OP diff.": type_and_time_diff[
                type_and_time_diff["Situation"] == "non open-play"
                ]["Time Diff. (min)"].rename(),
            "Home advantage": (home_results["shots"] - away_results["shots"]).rename(),
            "Shooters in top games": player_shots.rename(),
        }

    def _extract_ml(self, path: str):
        games = pd.read_csv(os.path.join(path, "games.csv"))
        gamestats = pd.read_csv(os.path.join(path, "gamestats.csv"))
        games = games.merge(gamestats, how="left", on="gameID").reset_index(drop=True)
        teamstats = pd.read_csv(os.path.join(path, "teamstats.csv")).set_index(["gameID", "teamID"])

        features = []
        outcomes = []

        for i, row in games.iterrows():
            gid = row['gameID']
            home_id = row['homeTeamID']
            away_id = row['awayTeamID']
            if (gid, home_id) not in teamstats.index or (gid, away_id) not in teamstats.index:
                continue
            if np.isnan([gid, home_id, away_id]).any() or (gid, home_id) not in teamstats.index:
                continue
            home_stats = teamstats.loc[(gid, home_id)]
            if len(home_stats.shape) > 1:
                home_stats = home_stats.iloc[0]
            outcomes.append(home_stats.result)

            all_features = {}

            # from gamestats
            game_stats = row.drop([
                "homeGoals", "awayGoals", "homeGoalsHalfTime", "awayGoalsHalfTime", "gameID", "homeTeamID", "awayTeamID"
            ])
            for k, v in game_stats.items():
                all_features[f"game-{k}"] = v

            # from teamstats
            curr_date = home_stats.date
            prev_home_stats = teamstats[
                (teamstats.index.get_level_values("teamID") == home_id) & (teamstats.date < curr_date).values
                ]
            prev_away_stats = teamstats[
                (teamstats.index.get_level_values("teamID") == away_id) & (teamstats.date < curr_date).values
                ]
            prev_home_stats = prev_home_stats.sort_values("date", ascending=True).drop(columns=["date"])
            prev_away_stats = prev_away_stats.sort_values("date", ascending=True).drop(columns=["date"])
            prev_home_stats[["W", "L", "D"]] = 0
            prev_away_stats[["W", "L", "D"]] = 0
            for s in ["W", "L", "D"]:
                prev_home_stats.loc[prev_home_stats.result == s, s] = 1
                prev_away_stats.loc[prev_away_stats.result == s, s] = 1
            prev_home_stats = prev_home_stats.drop(columns=["result"])
            prev_away_stats = prev_away_stats.drop(columns=["result"])
            for k, v in prev_home_stats.select_dtypes(include=np.number).mean().items():
                all_features[f"home-all-{k}-m"] = v
            for k, v in prev_home_stats.select_dtypes(include=np.number).std().items():
                all_features[f"home-all-{k}-s"] = v
            for k, v in prev_away_stats.select_dtypes(include=np.number).mean().items():
                all_features[f"away-all-{k}-m"] = v
            for k, v in prev_away_stats.select_dtypes(include=np.number).mean().items():
                all_features[f"away-all-{k}-s"] = v
            recent_home_stats = prev_home_stats[-3:]
            recent_away_stats = prev_away_stats[-3:]
            for k, v in recent_home_stats.select_dtypes(include=np.number).mean().items():
                all_features[f"home-recent-all-{k}-m"] = v
            for k, v in recent_away_stats.select_dtypes(include=np.number).std().items():
                all_features[f"away-recent-all-{k}-s"] = v
            home_loc_stats = prev_home_stats[prev_home_stats.location == "h"]
            away_loc_stats = prev_away_stats[prev_away_stats.location == "a"]
            for k, v in home_loc_stats.select_dtypes(include=np.number).mean().items():
                all_features[f"home-home-{k}-m"] = v
            for k, v in home_loc_stats.select_dtypes(include=np.number).std().items():
                all_features[f"home-home-{k}-s"] = v
            for k, v in away_loc_stats.select_dtypes(include=np.number).mean().items():
                all_features[f"away-home-{k}-m"] = v
            for k, v in away_loc_stats.select_dtypes(include=np.number).std().items():
                all_features[f"away-home-{k}-s"] = v

            features.append(all_features)

        X = pd.DataFrame(features)
        y = pd.Series(outcomes)

        return X, y


def parse_args() -> argparse.Namespace:
    full_parser = argparse.ArgumentParser()
    subparsers = full_parser.add_subparsers(dest="op")
    parser = subparsers.add_parser("all")
    parser.add_argument("--skip-shapes", dest="shape", action="store_false", default=True)
    parser.add_argument("--skip-schema", dest="schema", action="store_false", default=True)
    parser.add_argument("--skip-degrees", dest="degrees", action="store_false", default=True)
    parser.add_argument("--skip-deg-vis", dest="degrees_vis", action="store_false", default=True)
    parser.add_argument("--skip-time", dest="time", action="store_false", default=True)
    parser.add_argument("--skip-goals", dest="goals", action="store_false", default=True)
    parser.add_argument("--skip-queries", dest="queries", action="store_false", default=True)
    parser.add_argument("--skip-downstream", dest="downstream", action="store_false", default=True)

    parser = subparsers.add_parser("single")
    parser.add_argument("--do-shapes", dest="shape", action="store_true", default=False)
    parser.add_argument("--do-schema", dest="schema", action="store_true", default=False)
    parser.add_argument("--do-degrees", dest="degrees", action="store_true", default=False)
    parser.add_argument("--do-deg-vis", dest="degrees_vis", action="store_true", default=False)
    parser.add_argument("--do-time", dest="time", action="store_true", default=False)
    parser.add_argument("--do-goals", dest="goals", action="store_true", default=False)
    parser.add_argument("--do-queries", dest="queries", action="store_true", default=False)
    parser.add_argument("--do-downstream", dest="downstream", action="store_true", default=False)
    return full_parser.parse_args()


def main():
    args = parse_args()
    evaluator = FootballEvaluator()
    if args.shape:
        evaluator.evaluate_shapes()
    if args.schema:
        evaluator.evaluate_schema()
    if args.degrees:
        evaluator.evaluate_degrees(visualize=args.degrees_vis)
    if args.time:
        evaluator.evaluate_time()
    if args.goals:
        evaluator.evaluate_game_goals()
    if args.queries:
        evaluator.evaluate_query()
    if args.downstream:
        evaluator.evaluate_ml()


if __name__ == '__main__':
    main()
