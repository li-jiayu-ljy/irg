import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .schema import RelationalTransformer, TableConfig
from .standalone import train_standalone, generate_standalone
from .degree import train_degrees, predict_degrees
from .isna_indicator import train_isna_indicator, predict_isna_indicator
from .aggregated import train_aggregated_information, generate_aggregated_information
from .actual import train_actual_values, generate_actual_values
from .match import match
from .utils import CacheBlock, log_resource_usage, resume_from_last


class IncrementalRelationalGenerator:
    def __init__(self, tables: Dict[str, TableConfig],
                 order: List[str],
                 max_ctx_dim: int = 100,
                 default_args: Dict[str, Any] = {}, table_specific_args: Dict[str, Dict[str, Any]] = {}):
        self.transformer = RelationalTransformer(tables, order, max_ctx_dim)
        self.model_args = {}
        for t in self.transformer.order:
            if self.transformer.transformers[t].config.foreign_keys:
                keys = ["degree", "isna", "aggregated", "actual"]
            else:
                keys = ["standalone"]
            all_keys = set(table_specific_args.get(t, {}).keys()) | set(default_args.keys())
            out_args = {}
            for k in all_keys:
                if k in keys:
                    out_args[k] = table_specific_args.get(t, {}).get(k, {}) | default_args.get(k, {})
                else:
                    value = table_specific_args.get(t, {}).get(k, default_args.get(k, {}))
                    if not isinstance(value, Dict):
                        out_args[k] = value
            for k in keys:
                if k not in out_args:
                    out_args[k] = {}
            self.model_args[t] = out_args
        self.deg_ranges = {}

    def fit(self, tables: Dict[str, str], out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        if not os.path.exists(out_dir):
            raise ValueError()
        data_cache_dir = os.path.join(out_dir, "data")
        model_cache_dir = os.path.join(out_dir, "model")
        resource_path = os.path.join(out_dir, "resource.csv")
        with open(resource_path, "w") as f:
            f.write("task,mem(MB),Gmem(MB),time(s)\n")
        self.transformer.fit(tables, data_cache_dir, resource_path)

        for tn in self.transformer.order:
            table_model_dir = os.path.join(model_cache_dir, tn)
            if not self.model_args[tn].get("synthesize", True):
                continue
            foreign_keys = self.transformer.transformers[tn].config.foreign_keys
            self.deg_ranges[tn] = []
            if foreign_keys:
                for i, fk in enumerate(foreign_keys):
                    with log_resource_usage(
                            resource_path, f"fit degrees {tn}.({'|'.join(fk.child_column_names)})[{i}]"
                    ):
                        deg_context, deg = self.transformer.degree_prediction_for(tn, i, data_cache_dir)
                        if deg.min() > 0:
                            fk.total_participate = True
                        if deg.max() <= 1:
                            fk.unique = True
                        self.deg_ranges[tn].append((deg.min(), deg.max()))
                        train_degrees(
                            deg_context, deg, os.path.join(table_model_dir, f"degree{i}"),
                            **self.model_args[tn]["degree"]
                        )

                    isnull = self.transformer.isna_indicator_prediction_for(tn, i, data_cache_dir)
                    if isnull is not None:
                        with log_resource_usage(
                                resource_path, f"fit isna {tn}.({'|'.join(fk.child_column_names)})[{i}]"
                        ):
                            isna_context, isna = isnull
                            train_isna_indicator(
                                isna_context, isna, os.path.join(table_model_dir, f"isna{i}"),
                                **self.model_args[tn]["isna"]
                            )
                with log_resource_usage(resource_path, f"fit aggregated {tn}"):
                    agg_context, agg = self.transformer.aggregated_generation_for(tn, data_cache_dir)
                    train_aggregated_information(
                        agg_context, agg, os.path.join(table_model_dir, "aggregated"),
                        **self.model_args[tn]["aggregated"]
                    )
                with log_resource_usage(resource_path, f"fit actual {tn}"):
                    context, length, values, groups = self.transformer.actual_generation_for(tn, data_cache_dir)
                    train_actual_values(
                        context, length, values, groups, os.path.join(table_model_dir, "actual"),
                        **self.model_args[tn]["actual"]
                    )
            else:
                with log_resource_usage(resource_path, f"fit standalone {tn}"):
                    encoded = self.transformer.standalone_encoded_for(tn, data_cache_dir)
                    train_standalone(encoded, table_model_dir, **self.model_args[tn]["standalone"])

    def generate(self, out_dir: str, trained_dir: str, table_sizes: Dict[str, int] = {}):
        cache_dir = os.path.join(out_dir, "cache")
        resource_path = os.path.join(os.path.dirname(trained_dir), "resource.csv")
        loaded_locals = resume_from_last(cache_dir)
        table_sizes = {
            t: table_sizes.get(t, self.transformer.fitted_size_of(t)) for t in self.transformer.order
        }
        with CacheBlock("init", cache_dir) as run:
            if run:
                with log_resource_usage(resource_path, "initialize generation"):
                    self.transformer.prepare_sampled_dir(out_dir)
                    resources = pd.read_csv(resource_path)
                    start_fit = resources["task"].str.startswith("fit ")
                    last_fit = resources[start_fit].index[-1]
                    resources[:last_fit + 1].to_csv(resource_path, index=False)
                    del resources, start_fit, last_fit, run

        for tn in self.transformer.order:
            table_model_dir = os.path.join(trained_dir, tn)
            if self.model_args[tn].get("synthesize", True):
                foreign_keys = self.transformer.transformers[tn].config.foreign_keys
                if foreign_keys:
                    fk = foreign_keys[0]
                    with CacheBlock(f"length-{tn}", cache_dir) as run:
                        if run:
                            with log_resource_usage(
                                    resource_path, f"gen degrees {tn}.({'|'.join(fk.child_column_names)})[0]"
                            ):
                                deg_context, _ = self.transformer.degree_prediction_for(tn, 0, out_dir)
                                min_val, max_val = self.deg_ranges[tn][0]
                                if fk.total_participate:
                                    min_val = 1
                                if fk.unique:
                                    max_val = 1
                                if min_val != max_val and not fk.unique:
                                    max_val = np.inf
                                degrees = predict_degrees(
                                    deg_context, os.path.join(table_model_dir, f"degree0"),
                                    expected_sum=table_sizes[tn], tolerance=0,
                                    min_val=min_val, max_val=max_val
                                )
                                self.transformer.save_degree_for(tn, 0, degrees, out_dir)
                                del deg_context, min_val, max_val, degrees, _, run

                    with CacheBlock(f"agg-{tn}", cache_dir) as run:
                        if run:
                            with log_resource_usage(resource_path, f"gen agg {tn}"):
                                agg_context, _ = self.transformer.aggregated_generation_for(tn, out_dir)
                                agg = generate_aggregated_information(
                                    agg_context, os.path.join(table_model_dir, "aggregated")
                                )
                                self.transformer.save_aggregated_info_for(tn, agg, out_dir)
                                del agg_context, _, agg, run
                    with CacheBlock(f"actual-{tn}", cache_dir) as run:
                        if run:
                            with log_resource_usage(resource_path, f"gen actual {tn}"):
                                context, length, _, _ = self.transformer.actual_generation_for(tn, out_dir)
                                values, groups = generate_actual_values(
                                    context, length, os.path.join(table_model_dir, "actual")
                                )
                                self.transformer.save_actual_values_for(tn, values, groups, out_dir)
                                n_rows = values.shape[0]
                                del context, length, _, values, run
                    
                    with CacheBlock(f"first-match-{tn}", cache_dir) as run:
                        if run:
                            with log_resource_usage(
                                    resource_path, f"prepare first match {tn}"
                            ):
                                _, deg = self.transformer.degree_prediction_for(tn, 0, out_dir)
                                match_to_orig = np.arange((deg > 0).sum()) + (deg == 0).cumsum()[deg > 0]
                                inverse_groups = np.full(n_rows, -1)
                                for j, g in enumerate(groups):
                                    inverse_groups[g] = match_to_orig[j]
                                self.transformer.save_matched_indices_for(
                                    tn, 0, inverse_groups, out_dir
                                )
                                del inverse_groups, run
                                
                    for i, fk in enumerate(foreign_keys[1:], 1):
                        with CacheBlock(f"isna{i}-{tn}", cache_dir) as run:
                            if run:
                                with log_resource_usage(
                                        resource_path, f"gen isna {tn}.({'|'.join(fk.child_column_names)})[{i}]"
                                ):
                                    isnull = self.transformer.isna_indicator_prediction_for(tn, i, out_dir)
                                    if isnull is not None:
                                        isna_context, _ = isnull
                                        isna = predict_isna_indicator(
                                            isna_context, os.path.join(table_model_dir, f"isna{i}")
                                        )
                                        self.transformer.save_isna_indicator_for(tn, i, isna, out_dir)
                                        del isna_context, isna, _
                                    del isnull, run
                                    
                        with CacheBlock(f"deg{i}-{tn}", cache_dir) as run:
                            if run:
                                with log_resource_usage(
                                        resource_path, f"gen degrees {tn}.({'|'.join(fk.child_column_names)})[{i}]"
                                ):
                                    deg_context, _ = self.transformer.degree_prediction_for(tn, i, out_dir)
                                    min_val, max_val = self.deg_ranges[tn][i]
                                    if fk.total_participate:
                                        min_val = 1
                                    if fk.unique:
                                        max_val = 1
                                    if min_val != max_val and not fk.unique:
                                        max_val = np.inf
                                    isnull = self.transformer.isna_indicator_prediction_for(
                                        tn, i, out_dir
                                    )
                                    if isnull is not None:
                                        expected_sum = n_rows - isnull[1].sum()
                                    else:
                                        expected_sum = n_rows
                                    degrees = predict_degrees(
                                        deg_context, os.path.join(table_model_dir, f"degree{i}"),
                                        expected_sum=expected_sum, tolerance=0,
                                        min_val=min_val, max_val=max_val
                                    )
                                    self.transformer.save_degree_for(tn, i, degrees, out_dir)
                                    del deg_context, min_val, max_val, _, degrees, run

                        with CacheBlock(f"match{i}-{tn}", cache_dir) as run:
                            if run:
                                with log_resource_usage(
                                        resource_path, f"match {tn}.({'|'.join(fk.child_column_names)})[{i}]"
                                ):
                                    (values, parent, degrees, isna,
                                     pools, non_overlapping_groups) = self.transformer.fk_matching_for(tn, i, out_dir)
                                    matched = match(values, parent, degrees, isna, pools, non_overlapping_groups)
                                    self.transformer.save_matched_indices_for(tn, i, matched, out_dir)
                                    del values, parent, degrees, isna, pools, non_overlapping_groups, matched, run
                else:
                    with CacheBlock(f"standalone-{tn}", cache_dir) as run:
                        if run:
                            with log_resource_usage(resource_path, f"gen standalone {tn}"):
                                encoded = generate_standalone(table_sizes[tn], table_model_dir)
                                self.transformer.save_standalone_encoded_for(tn, encoded, out_dir)
                                del encoded, run
            else:
                self.transformer.copy_fitted_for(tn, out_dir)

            with CacheBlock(f"next-{tn}", cache_dir) as run:
                if run:
                    with log_resource_usage(resource_path, f"prepare next {tn}"):
                        self.transformer.prepare_next_for(tn, out_dir)
