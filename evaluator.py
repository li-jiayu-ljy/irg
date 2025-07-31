import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib import pyplot as plt
from lightgbm import LGBMClassifier
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def format_by_rank(
        data: pd.DataFrame, maximize: bool = True, warning_threshold: float = None, stds: pd.DataFrame = None
):
    ranks = data.abs().rank(axis=1, method="min", ascending=not maximize)
    prefix = pd.DataFrame("", index=ranks.index, columns=ranks.columns)
    suffix = pd.DataFrame("", index=ranks.index, columns=ranks.columns)
    highlight = ranks <= 1.0
    suffix[highlight] = r"}"
    if stds is None:
        prefix[highlight] = r"\textbf{"
        result = prefix + data.applymap(lambda x: f"{x:.3f}") + suffix
    else:
        prefix[highlight] = r"\boldsymbol{"
        result = ("$" + prefix + data.applymap(lambda x: f"{x:.3f}") + r"_{\pm "
                  + stds.applymap(lambda x: f"{x:.3f}") + "}" + suffix + "$")
    if warning_threshold is not None:
        data = data.abs()
        warning = data < warning_threshold if maximize else data > warning_threshold
        warn_prefix = pd.DataFrame("", index=ranks.index, columns=ranks.columns)
        warn_suffix = pd.DataFrame("", index=ranks.index, columns=ranks.columns)
        warn_prefix[warning] = r"\textcolor{red}{"
        warn_suffix[warning] = r"}"
        result = warn_prefix + result + warn_suffix
    return result


class RelationalEvaluator:
    renames = {
        "sdv": "HMA",
        "ind": "IND",
        "rctgan": "RCT",
        "clava": "CLD",
        "irg": "IRG"
    }
    hue_order = ["Real", "IRG", "HMA", "IND", "RCTGAN", "CLD"]
    palette = "tab10"

    def __init__(
            self,
            dataset_name: str,
            models: list = ["sdv", "ind", "rctgan", "clava", "irg"],
            sdv_scale: float = 1.0
    ):
        self.dataset_name = dataset_name
        with open(os.path.join("datasets", dataset_name, "schema", "irg.yaml"), "r") as f:
            self.schema = yaml.safe_load(f)
        self.real_path = os.path.join("datasets", dataset_name, "preprocessed")
        self.model_paths = {
            m: os.path.join("datasets", dataset_name, "out", m, "generated") for m in models
        }
        self.report_path = os.path.join("datasets", dataset_name, "reports")
        os.makedirs(self.report_path, exist_ok=True)
        self.sdv_scale = sdv_scale
        plt.rcParams["font.family"] = "DejaVu Serif"

    def evaluate_schema(self):
        pk_uniqueness = pd.DataFrame()
        fk_validity = pd.DataFrame()
        ineq_validity = pd.DataFrame()
        for table_name, table_args in self.schema["tables"].items():
            primary_key = table_args.get("primary_key")
            foreign_keys = table_args.get("foreign_keys", [])
            inequalities = table_args.get("inequality", [])
            for m, p in self.model_paths.items():
                table = pd.read_csv(os.path.join(p, f"{table_name}.csv"))
                if primary_key is not None and not isinstance(primary_key, str):
                    pk_uniqueness.loc[table_name, m] = (1 -
                                                        table[primary_key].drop_duplicates().shape[0] / table.shape[0])
                for fk in foreign_keys:
                    parent_table_name, child_column_names, parent_column_names = (
                        fk["parent_table_name"], fk["child_column_names"], fk["parent_column_names"])
                    parent = pd.read_csv(os.path.join(p, f"{parent_table_name}.csv"))
                    fk_validity.loc[
                        f"{table_name}.{child_column_names} -> {parent_table_name}.{parent_column_names}", m
                    ] = table[
                        [child_column_names] if isinstance(child_column_names, str) else child_column_names
                    ].dropna().merge(
                        parent, left_on=child_column_names, right_on=parent_column_names, how="left",
                        indicator="__merge__"
                    )["__merge__"].value_counts(normalize=True).to_dict().get("left_only", 0.0)
                for ineq in inequalities:
                    l, r = ineq
                    ineq_validity.loc[f"{table_name}{ineq}", m] = (table[l] == table[r]).mean()

        pk_uniqueness.loc["Avg.", :] = pk_uniqueness.mean()
        fk_validity.loc["Avg.", :] = fk_validity.mean()
        ineq_validity.loc["Avg.", :] = ineq_validity.mean()
        pk_uniqueness.loc["# Vio.", :] = (pk_uniqueness.iloc[:-1] > 0).sum()
        fk_validity.loc["# Vio.", :] = (fk_validity.iloc[:-1] > 0).sum()
        ineq_validity.loc["# Vio.", :] = (ineq_validity.iloc[:-1] > 0).sum()
        pk_uniqueness.to_csv(os.path.join(self.report_path, "pk_uniqueness.csv"))
        fk_validity.to_csv(os.path.join(self.report_path, "fk_validity.csv"))
        ineq_validity.to_csv(os.path.join(self.report_path, "ineq_validity.csv"))
        formatted_pk_uniqueness = format_by_rank(pk_uniqueness, maximize=False, warning_threshold=0.)
        formatted_fk_validity = format_by_rank(fk_validity, maximize=False, warning_threshold=0.)
        formatted_ineq_validity = format_by_rank(ineq_validity, maximize=False, warning_threshold=0.)
        formatted_pk_uniqueness.to_latex(os.path.join(self.report_path, "pk_uniqueness.tex"), escape=False)
        formatted_fk_validity.to_latex(os.path.join(self.report_path, "fk_validity.tex"), escape=False)
        formatted_ineq_validity.to_latex(os.path.join(self.report_path, "ineq_validity.tex"), escape=False)
        pd.concat({
            "PK": formatted_pk_uniqueness, "FK": fk_validity, "INEQ": formatted_ineq_validity
        }, axis=0).to_latex(os.path.join(self.report_path, "schema.tex"), escape=False)

    def evaluate_shapes(self):
        sizes = pd.DataFrame()
        for table_name in self.schema["tables"]:
            real_size = pd.read_csv(os.path.join(self.real_path, f"{table_name}.csv")).shape[0]
            for m, p in self.model_paths.items():
                exp_size = real_size if m != "sdv" else real_size * self.sdv_scale
                model_size = pd.read_csv(os.path.join(p, f"{table_name}.csv")).shape[0]
                sizes.loc[table_name, m] = (model_size - exp_size) / exp_size
        sizes.loc["Avg.", :] = sizes.abs().mean()
        sizes.to_csv(os.path.join(self.report_path, "sizes.csv"))
        sizes = format_by_rank(sizes, maximize=False, warning_threshold=0.5)
        sizes.to_latex(os.path.join(self.report_path, "sizes.tex"), escape=False)

    def evaluate_degrees(self, visualize: bool = False):
        degrees = pd.DataFrame()
        isna = pd.DataFrame()
        n_fks = 0
        for table_name, table_args in self.schema["tables"].items():
            n_fks += len(table_args.get("foreign_keys", []))
        ncols = 3
        nrows = int(np.ceil(n_fks / ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 2.5 * nrows), squeeze=False)
        idx = 0
        handles, labels = None, None

        for table_name, table_args in self.schema["tables"].items():
            for fk in table_args.get("foreign_keys", []):
                parent_table_name, child_column_names, parent_column_names = (
                    fk["parent_table_name"], fk["child_column_names"], fk["parent_column_names"])
                fk_descr = f"{table_name}.{child_column_names} -> {parent_table_name}.{parent_column_names}"
                real_degrees, real_isna = self._extract_degrees(
                    self.real_path, parent_table_name, table_name, parent_column_names, child_column_names
                )
                if real_isna is not None:
                    real_isna = real_isna.mean()
                all_degrees = {
                    "Real": real_degrees
                }
                for m, p in self.model_paths.items():
                    syn_degrees, syn_isna = self._extract_degrees(
                        p, parent_table_name, table_name, parent_column_names, child_column_names
                    )
                    degrees.loc[fk_descr, m] = ks_2samp(real_degrees.values, syn_degrees.values).statistic
                    if real_isna is not None:
                        syn_isna = 0. if syn_isna is None else syn_isna.mean()
                        isna.loc[fk_descr, m] = syn_isna - real_isna
                    all_degrees[self.renames[m]] = syn_degrees
                if visualize:
                    combined_degrees = pd.concat(all_degrees).reset_index().rename(columns={
                        "level_0": "Model", 0: "Degrees"
                    })
                    ax: plt.Axes = axes[idx // ncols][idx % ncols]
                    combined_degrees["Degrees"] += np.random.normal(0, 0.1, combined_degrees.shape[0])
                    sns.kdeplot(
                        combined_degrees, x="Degrees", hue="Model", common_norm=True, ax=ax,
                        hue_order=self.hue_order, palette=self.palette, clip=(0, combined_degrees["Degrees"].max())
                    )
                    if handles is None:
                        legend = ax.get_legend()
                        handles, labels = legend.legendHandles, [text.get_text() for text in legend.texts]
                    ax.legend_.remove()
                    ax.set_ylabel("")
                    ax.set_xlabel("")
                    ax.set_title(f"FK {idx + 1}")
                idx += 1

        degrees.loc["Avg.", :] = degrees.mean()
        isna.loc["Avg.", :] = isna.abs().mean()
        degrees.to_csv(os.path.join(self.report_path, "degrees.csv"))
        degrees = format_by_rank(degrees)
        degrees.to_latex(os.path.join(self.report_path, "degrees.tex"), escape=False)
        isna.to_csv(os.path.join(self.report_path, "isna.csv"))
        isna = format_by_rank(isna, maximize=False)
        isna.to_csv(os.path.join(self.report_path, "isna.tex"))
        if visualize:
            for i in range(nrows):
                axes[i][0].set_ylabel("Density", va="center", fontsize=20, labelpad=10)
            fig.legend(
                handles, labels, columnspacing=5.5, loc="upper center", ncol=len(self.hue_order),
                bbox_to_anchor=(0.5, 1.08), frameon=False, fontsize=24
            )
            plt.tight_layout()
            fig.savefig(os.path.join(self.report_path, "degrees.png"), bbox_inches='tight')
            fig.savefig(os.path.join(self.report_path, "degrees.pdf"), bbox_inches='tight')

    @staticmethod
    def _extract_degrees(base_path: str, parent: str, child: str, parent_columns, child_columns):
        parent_table = pd.read_csv(os.path.join(base_path, f"{parent}.csv"))
        child_table = pd.read_csv(os.path.join(base_path, f"{child}.csv"))
        child_table["__index"] = child_table.index.astype(int)
        mi = parent_table.merge(
            child_table, left_on=parent_columns, right_on=child_columns, how="right", indicator="__merge"
        )
        mi = mi.groupby("__index").head(1).set_index("__index").loc[child_table.index]
        child_table = child_table[
            (mi["__merge"] == "both") | (
                child_table[child_columns].isna() if isinstance(child_columns, str)
                else child_table[child_columns].isna().any(axis=1)
            )
        ].drop(columns=["__index"])
        has_child_degrees = child_table.groupby(child_columns).size()
        if isinstance(parent_columns, str):
            parent_table = parent_table.rename(columns={parent_columns: child_columns})
        else:
            parent_table = parent_table.rename(columns={p: c for p, c in zip(parent_columns, child_columns)})
        degrees = pd.Series(0, index=parent_table.set_index(child_columns).index)
        degrees[has_child_degrees.index] = has_child_degrees
        isna = child_table[child_columns].isna()
        if np.any(isna.values):
            isna = isna if len(isna.shape) == 1 else isna.any(axis=1)
        else:
            isna = None
        return degrees, isna

    def evaluate_time(self):
        timing = pd.DataFrame()
        for m in self.model_paths:
            with open(os.path.join("datasets", self.dataset_name, "out", m, "timing.json"), "r") as f:
                model_timing = json.load(f)
            fit_time = model_timing["fit"] + model_timing.get("preprocess", 0)
            gen_time = model_timing["sample"] + model_timing.get("postprocess", 0)
            timing.loc["fit", m] = fit_time
            timing.loc["sample", m] = gen_time
        timing.to_csv(os.path.join(self.report_path, "timing.csv"))
        timing.to_latex(os.path.join(self.report_path, "timing.tex"), escape=False)

    def evaluate_ml(self):
        perf = pd.DataFrame(index=pd.MultiIndex.from_frame(pd.DataFrame(columns=["ML", "rep", "metric"])))
        real_x, real_y = self._extract_ml(self.real_path)
        raw_columns = real_x.columns
        real_dtypes = real_x.dtypes
        le = LabelEncoder()
        real_y = le.fit_transform(real_y)
        num_columns = [c for c in real_x.select_dtypes(include="number").columns if real_x[c].nunique() > 10]
        num_scaler = StandardScaler()
        real_num_x = num_scaler.fit_transform(real_x[num_columns].fillna(0))
        cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        real_cat_x = cat_encoder.fit_transform(real_x.drop(columns=num_columns).fillna("NA!").astype(str))
        real_x = np.concatenate([real_num_x, real_cat_x], axis=1)
        for i in range(3):
            train_x, test_x, train_y, test_y = train_test_split(real_x, real_y, stratify=real_y)
            self._eval_downstream(test_x, test_y, train_x, train_y, perf, "Real", i)

        for m, p in self.model_paths.items():
            syn_x, syn_y = self._extract_ml(p)
            if syn_x.shape[-1] == 0:
                syn_x = pd.DataFrame(columns=raw_columns)
                syn_num_x = syn_x[num_columns]
                syn_cat_x = syn_x.drop(columns=num_columns)
            else:
                syn_x = syn_x[raw_columns].astype(real_dtypes)
                syn_num_x = num_scaler.transform(syn_x[num_columns].fillna(0))
                syn_cat_x = cat_encoder.transform(syn_x.drop(columns=num_columns).fillna("NA!").astype(str))

            syn_y = le.transform(syn_y)
            syn_x = np.concatenate([syn_num_x, syn_cat_x], axis=1)
            for i in range(3):
                self._eval_downstream(real_x, real_y, syn_x, syn_y, perf, self.renames[m], i)

        perf.to_csv(os.path.join(self.report_path, "ml-perf.csv"))
        grouped = perf.groupby(level=(0, 2))
        format_by_rank(
            grouped.mean(), maximize=True, stds=grouped.std()
        ).to_latex(os.path.join(self.report_path, "ml-perf.tex"), escape=False)

    @staticmethod
    def _eval_downstream(test_x, test_y, train_x, train_y, perf: pd.DataFrame, key: str, rep: int):
        if train_x.shape[0] == 0:
            perf.loc[:, key] = np.nan
        else:
            acc, f1, auc = RelationalEvaluator._run_downstream(DecisionTreeClassifier, test_x, test_y, train_x, train_y)
            perf.loc[("DT", rep, "Acc."), key] = acc
            perf.loc[("DT", rep, "F1"), key] = f1
            perf.loc[("DT", rep, "AUC"), key] = auc
            acc, f1, auc = RelationalEvaluator._run_downstream(RandomForestClassifier, test_x, test_y, train_x, train_y)
            perf.loc[("RF", rep, "Acc."), key] = acc
            perf.loc[("RF", rep, "F1"), key] = f1
            perf.loc[("RF", rep, "AUC"), key] = auc
            acc, f1, auc = RelationalEvaluator._run_downstream(XGBClassifier, test_x, test_y, train_x, train_y)
            perf.loc[("XGB", rep, "Acc."), key] = acc
            perf.loc[("XGB", rep, "F1"), key] = f1
            perf.loc[("XGB", rep, "AUC"), key] = auc
            acc, f1, auc = RelationalEvaluator._run_downstream(LGBMClassifier, test_x, test_y, train_x, train_y)
            perf.loc[("LGBM", rep, "Acc."), key] = acc
            perf.loc[("LGBM", rep, "F1"), key] = f1
            perf.loc[("LGBM", rep, "AUC"), key] = auc

    @staticmethod
    def _run_downstream(model, test_x, test_y, train_x, train_y):
        model = model()
        le = LabelEncoder()
        train_y = le.fit_transform(train_y)
        model.fit(train_x, train_y)
        pred = model.predict(test_x)
        pred = le.inverse_transform(pred)
        proba = model.predict_proba(test_x)
        if len(proba.shape) == 1:
            proba = np.stack([1 - proba, proba]).T
        elif proba.shape[-1] == 1:
            proba = np.concatenate([1 - proba, proba], axis=1)
        if len(np.unique(test_y)) > len(np.unique(train_y)):
            placeholder_probs = np.zeros((test_x.shape[0], len(np.unique(test_y))))
            for i, j in enumerate(np.unique(train_y)):
                placeholder_probs[:, j] = proba[:, i]
            proba = placeholder_probs
        return accuracy_score(test_y, pred), f1_score(test_y, pred, average="weighted"), roc_auc_score(
                test_y, proba, average="weighted", multi_class="ovr", labels=sorted(np.unique(test_y))
            )


    def _extract_ml(self, path: str):
        pass

    def evaluate_query(self):
        all_queries = {
            "Real": self._get_query_results(self.real_path),
        }
        for m, p in self.model_paths.items():
            if m == "ind":
                continue
            all_queries[self.renames[m]] = self._get_query_results(p)
        swapped_queries = defaultdict(dict)
        for k, v in all_queries.items():
            for kk, vv in v.items():
                swapped_queries[kk][k] = vv
        n_rows = int(np.ceil(len(swapped_queries) / 4))
        fig, axes = plt.subplots(n_rows, 4, figsize=(16, 2 * n_rows), squeeze=False)
        axes = axes.flatten()
        handles, labels = None, None
        for i, (k, v) in enumerate(swapped_queries.items()):
            if (v["Real"].mean() - v["Real"].min()) * 10 < v["Real"].max() - v["Real"].mean():
                log_scale = True
            else:
                log_scale = False
            if pd.api.types.is_integer_dtype(v["Real"].dtype):
                vc = pd.concat(v).round().dropna().astype(int)
            else:
                vc = pd.concat(v).astype(np.float64).dropna()
            vc = vc.reset_index().rename(columns={"level_0": "Model", 0: "Value"})
            ax = axes[i]
            sns.kdeplot(
                vc, x="Value", hue="Model", hue_order=[h for h in self.hue_order if h != "IND"],
                common_norm=False, ax=ax, palette=self.palette,
                clip=(min(vc["Value"].min(), 0), vc["Value"].max()), bw_adjust=3, log_scale=log_scale, #cut=0
            )

            if handles is None:
                legend = ax.get_legend()
                handles, labels = legend.legendHandles, [text.get_text() for text in legend.texts]
            ax.legend_.remove()
            ax.set_ylabel("")
            ax.set_xlabel("")
            ax.set_title(k)

        axes[0].set_ylabel("Density", va="center", fontsize=20, labelpad=10)
        plt.tight_layout(rect=(0, 0, 1, 0.8))
        fig.legend(
            handles, labels, columnspacing=3.5, loc="upper center", ncol=5,
            bbox_to_anchor=(0.5, 1.05), frameon=False, fontsize=24
        )
        fig.savefig(os.path.join(self.report_path, "queries.png"))
        fig.savefig(os.path.join(self.report_path, "queries.pdf"))

    @staticmethod
    def _get_query_results(path: str):
        pass
