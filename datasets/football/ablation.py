import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt

sys.path.insert(0, os.getcwd())
import irg

try:
    import datasets.football.evaluate as evaluate
except (ModuleNotFoundError, ImportError):
    import importlib
    import sys
    import os
    base_dir = os.path.dirname(__file__)
    spec = importlib.util.spec_from_file_location(
        "evaluate.py", os.path.join(base_dir, "evaluate.py")
    )
    evaluate = importlib.util.module_from_spec(spec)
    sys.modules["evaluate"] = evaluate


def ablation_reg_quantile():
    plt.rcParams["font.family"] = "DejaVu Serif"
    synthesizer = torch.load("datasets/football/out/irg/synthesizer.pt")
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.5))
    index = 0
    core_fks = {
        ("teamstats", 0): "T",
        ("appearances", 0): "P",
    }
    fk_repr_dict = {
        ("teamstats", 0): "T",
        ("teamstats", 1): "GT",
        ("games", 0): "G",
        ("games", 1): "HT",
        ("games", 2): "AT",
        ("appearances", 0): "P",
        ("appearances", 1): "G",
        ("shots", 0): "SS",
        ("shots", 1): "SA",
    }
    sum_deg = pd.DataFrame()
    for tn in synthesizer.transformer.order:
        foreign_keys = synthesizer.transformer.transformers[tn].config.foreign_keys
        for i, fk in enumerate(foreign_keys):
            fk_repr = fk_repr_dict[(tn, i)]
            deg_train_X, deg_train_y = synthesizer.transformer.degree_prediction_for(
                tn, i, "datasets/football/out/irg/data"
            )
            deg_test_X, deg_test_y = synthesizer.transformer.degree_prediction_for(
                tn, i, "datasets/football/out/irg/generated"
            )
            with open(f"datasets/football/out/irg/model/{tn}/degree{i}/info.json", "r") as f:
                info = json.load(f)
            zero_ratio = info["zero ratio"]
            real_sum = info["sum deg"]
            regressor = torch.load(f"datasets/football/out/irg/model/{tn}/degree{i}/reg.pt")
            predicted_degrees = regressor.predict(deg_test_X)
            predicted_degrees += np.random.normal(0, 1e-3, size=predicted_degrees.shape)  # avoid too many equal
            predicted_degrees_with_zeros = predicted_degrees
            is_zero = predicted_degrees_with_zeros < np.quantile(predicted_degrees_with_zeros, zero_ratio)
            predicted_degrees_with_zeros[is_zero] = 0
            min_val, max_val = synthesizer.deg_ranges[tn][i]
            raw_predicted_degrees_with_zeros = predicted_degrees_with_zeros.copy()
            sum_deg.loc["Exp. sum", fk_repr] = real_sum
            sum_deg.loc["IRG sum", fk_repr] = int(deg_test_y.sum())
            raw_predicted_degrees_with_zeros[(~is_zero) & (raw_predicted_degrees_with_zeros <= min_val)] = min_val
            if max_val is not None:
                raw_predicted_degrees_with_zeros = raw_predicted_degrees_with_zeros.clip(max=max_val)
            sum_deg.loc["w/o sum ctrl.", fk_repr] = int(raw_predicted_degrees_with_zeros.round().sum())

            if (tn, i) not in core_fks:
                continue
            predicted_degrees_with_zeros = irg.degree.round_degrees(
                real_sum, max_val, min_val, 0, predicted_degrees_with_zeros
            )
            predicted_degrees = irg.degree.round_degrees(
                real_sum, max_val, min_val, 0, predicted_degrees
            )
            all_min = min(
                deg_train_y.min(), deg_test_y.min(), predicted_degrees_with_zeros.min(), predicted_degrees.min()
            )
            all_max = max(
                deg_train_y.max(), deg_test_y.max(), predicted_degrees_with_zeros.max(), predicted_degrees.max()
            )
            ax = axes[index]
            palette = sns.color_palette(n_colors=4)
            linestyles = ["-", "--", "-.", ":"]
            for j, (label, deg_data) in enumerate({
                "Real": deg_train_y, "IRG": deg_test_y, "w/o Quantile": predicted_degrees_with_zeros,
                "w/o Quantile+Zeros": predicted_degrees
            }.items()):
                sns.kdeplot(
                    x=deg_data, label=label, ax=ax, color=palette[j],
                    clip=(all_min, all_max), bw_adjust=3, linestyle=linestyles[j],
                )
            plt.legend()
            index += 1
            ax.set_title(fk_repr)
    sum_deg.to_csv("datasets/football/reports/abl-sum-deg.csv")
    sum_deg.to_latex("datasets/football/reports/abl-sum-deg.tex")
    plt.tight_layout()
    fig.savefig("datasets/football/reports/abl-deg-q.png")
    fig.savefig("datasets/football/reports/abl-deg-q.pdf")


def ablation_na_ratio():
    plt.rcParams["font.family"] = "DejaVu Serif"
    synthesizer = torch.load("datasets/football/out/irg/synthesizer.pt")
    isna_train_X, isna_train_y = synthesizer.transformer.isna_indicator_prediction_for(
        "shots", 1, "datasets/football/out/irg/data"
    )
    isna_test_X, isna_test_y = synthesizer.transformer.isna_indicator_prediction_for(
        "shots", 1, "datasets/football/out/irg/generated"
    )
    isna_train_y = isna_train_y.astype(np.bool_)
    isna_test_y = isna_test_y.astype(np.bool_)
    classifier = torch.load("datasets/football/out/irg/model/shots/isna1/clf.pt")
    pred = classifier.predict(isna_test_X)
    if len(pred.shape) == 2:
        if pred.shape[-1] == 1:
            pred = pred[:, 0] >= 0.5
        else:
            pred = pred.argmax(axis=1) >= 0.5
    else:
        pred = pred >= 0.5
    all_isna = pd.concat([
        pd.DataFrame({"is NULL": isna_train_y, "Src.": "Real"}),
        pd.DataFrame({"is NULL": isna_test_y, "Src.": "IRG"}),
        pd.DataFrame({"is NULL": pred, "Src.": "w/o ratio ctrl."})
    ], ignore_index=True)
    # fig, ax = plt.subplots(1, 1, figsize=(5, 2.7))
    plt.figure(figsize=(5, 2.5))
    sns.histplot(
        all_isna.astype(str), x="is NULL", hue="Src.", palette="tab10", stat="probability", multiple="dodge",
        shrink=0.8, common_norm=False
    )
    plt.savefig("datasets/football/reports/abl-na-ratio.png")
    plt.savefig("datasets/football/reports/abl-na-ratio.pdf")


def ablation_seq():
    synthesizer = torch.load("datasets/football/out/irg/synthesizer.pt")
    context, length, values, groups = synthesizer.transformer.actual_generation_for(
        "teamstats", "datasets/football/out/irg/data"
    )
    expanded_context = np.zeros((values.shape[0], context.shape[-1]))
    for i, g in enumerate(groups):
        expanded_context[g, :] = context[i].reshape(1, -1)
    if not os.path.exists("datasets/football/out/irg/model/teamstats/original-aggregated"):
        shutil.copytree(
            "datasets/football/out/irg/model/teamstats/aggregated",
            "datasets/football/out/irg/model/teamstats/original-aggregated"
        )
        shutil.rmtree("datasets/football/out/irg/model/teamstats/aggregated")
    irg.aggregated.train_aggregated_information(
        expanded_context, values, "datasets/football/out/irg/model/teamstats/aggregated"
    )

    synthesizer.generate(
        "datasets/football/out/irg/abl-generated-seq",
        "datasets/football/out/irg/model",
    )
    shutil.copytree(
        "datasets/football/out/irg/model/teamstats/aggregated",
        "datasets/football/out/irg/model/teamstats/abl-seq-aggregated"
    )
    shutil.rmtree("datasets/football/out/irg/model/teamstats/aggregated")
    shutil.copytree(
        "datasets/football/out/irg/model/teamstats/original-aggregated",
        "datasets/football/out/irg/model/teamstats/aggregated",
    )
    evaluator = evaluate.FootballEvaluator()
    evaluator.model_paths = {
        "irg": "datasets/football/out/irg/generated",
        "abl": "datasets/football/out/abl-generated-seq"
    }
    evaluator.report_path = "datasets/football/abl-reports1"
    evaluator.renames = {
        "irg": "IRG",
        "abl": "w/o seq."
    }
    evaluator.evaluate_ml()


def ablation_component():
    synthesizer = torch.load("datasets/football/out/irg/synthesizer.pt")
    context, length, values, groups = synthesizer.transformer.actual_generation_for(
        "teamstats", "datasets/football/out/irg/data"
    )
    shutil.copytree(
        "datasets/football/out/model/teamstats/actual",
        "datasets/football/out/model/teamstats/original-actual"
    )
    shutil.rmtree("datasets/football/out/model/teamstats/actual")
    irg.actual.train_actual_values(
        context, length, values, groups, "datasets/football/out/model/teamstats/actual"
    )

    synthesizer.generate(
        "datasets/football/out/abl-generated-seq",
        "datasets/football/out/model",
    )
    shutil.copytree(
        "datasets/football/out/model/teamstats/actual",
        "datasets/football/out/model/teamstats/abl-comp-actual"
    )
    shutil.rmtree("datasets/football/out/model/teamstats/actual")
    shutil.copytree(
        "datasets/football/out/model/teamstats/original-actual",
        "datasets/football/out/model/teamstats/actual",
    )
    evaluator = evaluate.FootballEvaluator()
    evaluator.model_paths = {
        "irg": "datasets/football/out/irg/generated",
        "abl": "datasets/football/out/abl-generated-comp"
    }
    evaluator.report_path = "datasets/football/abl-reports2"
    evaluator.renames = {
        "irg": "IRG",
        "abl": "small comp."
    }
    evaluator.evaluate_ml()




if __name__ == "__main__":
    # ablation_na_ratio()
    # ablation_reg_quantile()
    ablation_seq()
