import hashlib
import json
import os
import shutil
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from typing_extensions import Self

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from .utils import load_from, log_resource_usage, save_to


class ForeignKey:
    def __init__(self,
                 child_table_name: str,
                 parent_table_name: str,
                 child_column_names: Union[str, Sequence[str]],
                 parent_column_names: Optional[Union[str, Sequence[str]]] = None,
                 unique: bool = False,
                 total_participate: bool = False):
        self.child_table_name = child_table_name
        self.parent_table_name = parent_table_name
        self.child_column_names = child_column_names if not isinstance(child_column_names, str) else [
            child_column_names]
        if parent_column_names is None:
            parent_column_names = self.child_column_names
        self.parent_column_names = parent_column_names if \
            not isinstance(parent_column_names, str) else [parent_column_names]
        self.unique = unique
        self.total_participate = total_participate

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ForeignKey):
            return False
        for k in ["child_table_name", "parent_table_name", "child_column_names"]:
            if getattr(self, k) != getattr(other, k):
                return False
        return True


class TableConfig:
    def __init__(self,
                 name: str,
                 primary_key: Optional[Union[str, Sequence[str]]] = None,
                 foreign_keys: Optional[Sequence[ForeignKey]] = None,
                 sortby: Optional[str] = None,
                 id_columns: Optional[Sequence[str]] = None,
                 inequality: Optional[Union[Tuple[str, str], Tuple[Tuple[str, ...], Tuple[str, ...]]]] = None):
        self.name = name
        self.primary_key = primary_key if not isinstance(primary_key, str) else [primary_key]
        self.foreign_keys = foreign_keys if foreign_keys is not None else []
        self.sortby = sortby
        self.id_columns = id_columns if id_columns is not None else []
        self.inequality = [
            ([a], [b]) if isinstance(a, str) else (a, b) for a, b in inequality
        ] if inequality is not None else []

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        foreign_keys = data.get("foreign_keys", [])
        foreign_keys = [ForeignKey(**x, child_table_name=data["name"]) for x in foreign_keys]
        data = data.copy()
        data["foreign_keys"] = foreign_keys
        return cls(**data)


class TableTransformer:
    def __init__(self, config: TableConfig):
        self.config = config
        self.columns = []
        self.categorical_columns = []
        self.numeric_columns = []
        self.agg_columns = None
        self.top_cat_values = {}
        self.agg_transformer = StandardScaler() if self.config.foreign_keys else None
        self.cat_transformer = OrdinalEncoder()
        self.num_transformer = StandardScaler()
        self.count_null = []
        self.split_dim = 0

    def fit(self, table: pd.DataFrame):
        for c in self.config.id_columns:
            if table[c].isna().any():
                self.count_null.append(c)
        self.columns = table.columns
        numeric_columns = table.select_dtypes(include=np.number).columns
        categorical_columns = table.drop(columns=numeric_columns.tolist()).columns
        self.categorical_columns = [
            c for c in categorical_columns if c not in self.config.id_columns
        ]
        self.numeric_columns = [
            c for c in numeric_columns if c not in self.config.id_columns
        ]
        for c in self.categorical_columns:
            self.top_cat_values[c] = table[c].value_counts().iloc[:3].values.tolist()
        if self.config.foreign_keys:
            aggregated, table = self.aggregate(table)
            aggregated = aggregated.bfill().ffill()
            self.agg_columns = aggregated.columns
            if aggregated.shape[-1] > 0:
                self.agg_transformer.fit(aggregated.values)
        table = table.bfill().ffill()
        if self.categorical_columns:
            cat = self.cat_transformer.fit_transform(table[self.categorical_columns].values)
            self.split_dim = cat.shape[1]
        else:
            self.split_dim = 0
        if self.numeric_columns:
            self.num_transformer.fit(table[self.numeric_columns].values)

    def aggregate(self, table: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not self.config.foreign_keys:
            raise RuntimeError(f"Table {self.config.name} has no FK, so aggregate is not a valid operation.")
        groupby_columns = self.config.foreign_keys[0].child_column_names
        groupby = table.groupby(groupby_columns)
        if self.config.sortby:
            first_sortby: pd.Series = groupby[self.config.sortby].head(1)
            first_sortby.index = pd.MultiIndex.from_frame(
                table.loc[first_sortby.index, groupby_columns]
            )
            sorby_diff: pd.Series = groupby[self.config.sortby].diff()
            sorby_diff = sorby_diff.fillna(sorby_diff.mean())
            table = pd.concat([
                table.drop(columns=[self.config.sortby]),
                sorby_diff.to_frame(self.config.sortby)
            ], axis=1)[table.columns]
            groupby = table.groupby(groupby_columns)
            out = self._aggregate_values(groupby)
            out = pd.concat([
                out, pd.concat({self.config.sortby: first_sortby.to_frame("first")}, axis=1)
            ], axis=1)
        else:
            out = self._aggregate_values(groupby)
        out.columns = pd.Index([f"{a}${b}" for a, b in out.columns])
        return out, table

    def _aggregate_values(self, groupby):
        if self.numeric_columns:
            num_groupby = groupby[self.numeric_columns]
            out = num_groupby.aggregate(["mean", "median", "std"]).fillna(0)
        else:
            out = pd.concat({"": groupby.size().to_frame()[[]]}, axis=1)
        if len(self.categorical_columns) > 0:
            cat_groupby = groupby[self.categorical_columns]
            out = pd.concat([out, self._aggregate_categorical(cat_groupby)], axis=1)
        if len(self.count_null) > 0:
            null_groupby = groupby[self.count_null].aggregate(lambda group: group.isna().mean())
            null_groupby = pd.concat({"null-ratio": null_groupby}, axis=1).swaplevel(0, 1, axis=1)
            out = pd.concat([out, null_groupby], axis=1)
        if out.index.nlevels <= 1:
            out.index = pd.MultiIndex.from_arrays([out.index], names=[out.index.name])
        return out

    def _aggregate_categorical(self, grouped: pd.core.groupby.generic.DataFrameGroupBy) -> pd.DataFrame:
        df = grouped.obj
        group_keys = grouped.grouper.names
        results = {}
        sizes = grouped.size()
        for col, values in self.top_cat_values.items():
            ctab = pd.crosstab(index=[df[k] for k in group_keys], columns=df[col])
            ctab = ctab.reindex(columns=values, fill_value=0)
            ctab_ratio = ctab / sizes.loc[ctab.index].values.reshape((-1, 1))
            results[col] = ctab_ratio
        final = pd.concat(results, axis=1)
        return final

    def transform(self, table: pd.DataFrame) -> Tuple[
        np.ndarray, Optional[Dict[Tuple, np.ndarray]], Optional[np.ndarray], Optional[pd.Index]
    ]:
        table = table.reset_index(drop=True)
        if self.config.foreign_keys:
            groups = table.groupby(self.config.foreign_keys[0].child_column_names).groups
            groups = {
                k: v.values for k, v in groups.items()
            }
            aggregated, table = self.aggregate(table)
            if aggregated.index.nlevels <= 1:
                groups = {(k,): v for k, v in groups.items()}
            agg_index = aggregated.index
            if aggregated.shape[-1] > 0:
                aggregated = self.agg_transformer.transform(aggregated.values)
            else:
                aggregated = aggregated.values
        else:
            groups = None
            agg_index = None
            aggregated = None
        if self.categorical_columns:
            cat = self.cat_transformer.transform(
                table[self.categorical_columns].values
            ) / np.array([len(x) for x in self.cat_transformer.categories_]).reshape((1, -1))
        else:
            cat = np.zeros((table.shape[0], 0))
        if self.numeric_columns:
            num = self.num_transformer.transform(table[self.numeric_columns].values)
        else:
            num = np.zeros((table.shape[0], 0))
        transformed = np.concatenate([cat, num], axis=1)
        return transformed, groups, aggregated, agg_index

    def inverse_transform(self, transformed: np.ndarray, groups: Optional[Dict[Tuple, np.ndarray]] = None,
                          aggregated: Optional[np.ndarray] = None, agg_index: Optional[pd.Index] = None
                          ) -> pd.DataFrame:
        if self.categorical_columns:
            cat = transformed[:, :self.split_dim]
            cat = np.clip(cat, 0, np.array([x.shape[0] for x in self.cat_transformer.categories_]) - 1).round()
            cat = self.cat_transformer.inverse_transform(cat)
            cat = pd.DataFrame(cat, columns=self.categorical_columns)
        else:
            cat = pd.DataFrame(index=np.arange(transformed.shape[0]), columns=[])
        if self.numeric_columns:
            num = self.num_transformer.inverse_transform(transformed[:, self.split_dim:])
            num = pd.DataFrame(num, columns=self.numeric_columns)
        else:
            num = pd.DataFrame(index=np.arange(transformed.shape[0]), columns=[])
        table = pd.concat([cat, num], axis=1)
        for c in self.config.id_columns:
            table[c] = np.arange(table.shape[0])
        table = table[self.columns]

        if self.config.foreign_keys:
            groupby_columns = self.config.foreign_keys[0].child_column_names
            for vals, idx in groups.items():
                table.loc[idx, groupby_columns] = pd.Series(
                    {c: v for c, v in zip(groupby_columns, vals)}
                ).to_frame().T.loc[[0] * idx.shape[0]].set_axis(idx, axis=0)
            if self.config.sortby:
                aggregated = self.agg_transformer.inverse_transform(aggregated)
                aggregated = pd.DataFrame(aggregated, index=agg_index, columns=self.agg_columns)
                first_sortby = aggregated[f"{self.config.sortby}$first"]
                head = table.groupby(groupby_columns)[groupby_columns].head(1)
                agg_idx_to_table_idx = {
                    tuple(row[groupby_columns]): i for i, row in head.iterrows()
                }
                first_sortby.index = [agg_idx_to_table_idx[x] for x in first_sortby.index]
                table.loc[head.index, self.config.sortby] = first_sortby
                table[self.config.sortby] = table.groupby(groupby_columns)[self.config.sortby].cumsum()
        return table

    @classmethod
    def load(cls, path: str) -> Self:
        return load_from(path)

    def save(self, path: str):
        save_to(self, path)


class RelationalTransformer:
    def __init__(self,
                 tables: Dict[str, TableConfig],
                 order: List[str],
                 max_ctx_dim: int = 100):
        self.order = order
        self.transformers = {}
        self.children: Dict[str, List[ForeignKey]] = defaultdict(list)
        for tn in order:
            config = tables[tn]
            self.transformers[tn] = TableTransformer(config)
            for fk in config.foreign_keys:
                self.children[fk.parent_table_name].append(fk)
        self.max_ctx_dim = max_ctx_dim
        self._fitted_cache_dir = None
        self._sizes_of = {}
        self._nullable = {}
        self._parent_dims = {}
        self._core_dims = {}

    def fit(self, tables: Dict[str, str], cache_dir: str = "./cache", resource_path: str = "./cache/resource.csv"):
        self._fitted_cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        for tn in self.order:
            table = pd.read_csv(tables[tn])
            self._sizes_of[tn] = table.shape[0]
            table.to_csv(os.path.join(cache_dir, f"{tn}.csv"), index=False)
            with log_resource_usage(resource_path, f"fit table {tn} transformer"):
                transformer = self.transformers[tn]
                if len(set(table.columns)) != len(table.columns):
                    raise ValueError(f"Same column name repeated in one table ({tn}).")
                transformer.fit(table)
                transformer.save(os.path.join(cache_dir, f"{tn}-transformer.pkl"))

            foreign_keys = self.transformers[tn].config.foreign_keys
            if foreign_keys:
                self._nullable[tn] = []
                with log_resource_usage(resource_path, f"transform {tn} FK"):
                    encoded, groups, aggregated, agg_index = transformer.transform(table)
                    save_to({
                        "actual": (None, None, encoded, None)
                    }, os.path.join(cache_dir, f"{tn}.pkl"))
                with log_resource_usage(resource_path, f"extend {tn}"):
                    key, context, new_encoded = self._extend_till(tn, tn, table.columns.tolist(), cache_dir)
                float_cols = [
                    c for c in key.select_dtypes(include="float").columns
                    if c not in self.transformers[tn].config.id_columns
                ]
                if np.abs(encoded - new_encoded[:, self._core_dims[tn]]).mean() > 1e-5 or not (
                        key.equals(table) or ((len(float_cols) == 0 or
                                               (key[float_cols] - table[float_cols]).abs().values.mean() <= 1e-5)
                                              and key.drop(columns=float_cols).equals(table.drop(columns=float_cols)))
                ):
                    raise RuntimeError(
                        f"Error when extending: {np.abs(encoded - new_encoded[:, self._core_dims[tn]]).mean()}, "
                        f"{key.equals(table)}, {len(float_cols)}, "
                        f"{(key[float_cols] - table[float_cols]).abs().values.mean()}, "
                        f"{key.drop(columns=float_cols).equals(table.drop(columns=float_cols))}."
                    )

                agg_context = np.zeros((aggregated.shape[0], 0))
                actual_context = np.zeros((aggregated.shape[0], 0))
                transformed_context = np.zeros((encoded.shape[0], 0))
                length = np.zeros(aggregated.shape[0])
                all_fk_info = []
                for fi, fk in enumerate(foreign_keys):
                    fk_info = {}
                    with log_resource_usage(
                            resource_path, f"get degrees {tn}.({'|'.join(fk.child_column_names)})[{fi}]"
                    ):
                        parent_key, parent_context, parent_encoded = self._extend_till(
                            fk.parent_table_name, tn, fk.parent_column_names, cache_dir, fitting=False, queue=[fk]
                        )
                        degree_x = np.concatenate([parent_context, parent_encoded], axis=1)
                        degree_y = table[fk.child_column_names].groupby(fk.child_column_names).size()
                        if degree_y.index.nlevels <= 1:
                            degree_y.index = pd.MultiIndex.from_arrays([degree_y.index], names=[degree_y.index.name])
                        if fi == 0:
                            raw_degree = degree_y[agg_index]
                        else:
                            raw_degree = None
                        parent_key_as_child = parent_key.rename(columns={
                            p: c for p, c in zip(fk.parent_column_names, fk.child_column_names)
                        })
                        y_order = pd.MultiIndex.from_frame(parent_key_as_child)
                        placeholder_degree_y = pd.Series(0, index=y_order)
                        placeholder_degree_y.loc[degree_y.index] = degree_y
                        degree_y = placeholder_degree_y.values
                    if fi == 0:
                        with log_resource_usage(resource_path, f"get context {tn}"):
                            non_zero_degree_x = pd.DataFrame(
                                degree_x, columns=[f"_dim{i:02d}" for i in range(degree_x.shape[-1])],
                                index=parent_key.index
                            )
                            non_zero_degree_x = pd.concat([parent_key_as_child, non_zero_degree_x], axis=1)
                            agg_context = agg_index.to_frame().reset_index(drop=True)
                            agg_context = agg_context.merge(
                                non_zero_degree_x, how="left", on=agg_index.names
                            )
                            agg_context = agg_context.set_index(agg_index.names)
                            if agg_context.index.nlevels <= 1:
                                agg_context.index = pd.MultiIndex.from_arrays(
                                    [agg_context.index], names=agg_index.names
                                )
                            length = raw_degree.values

                            agg_context = agg_context.loc[agg_index].values
                            actual_context = np.concatenate([agg_context, aggregated], axis=1)
                            actual_context = pd.DataFrame(actual_context, index=agg_index)
                            transformed_context = np.empty((encoded.shape[0], actual_context.shape[-1]))
                            for g, idx in groups.items():
                                transformed_context[idx] = actual_context.loc[g]
                            actual_context = actual_context.values
                    fk_info["degree"] = degree_x, degree_y

                    if table[fk.child_column_names].isna().any().any():
                        with log_resource_usage(
                                resource_path, f"get isna {tn}.({'|'.join(fk.child_column_names)})[{fi}]"
                        ):
                            isna_y = table[fk.child_column_names].isna().any(axis=1)
                            fk_info["isna"] = np.concatenate([transformed_context, new_encoded], axis=1), isna_y.values
                        self._nullable[tn].append(True)
                    else:
                        self._nullable[tn].append(False)
                    all_fk_info.append(fk_info)

                out = {
                    "aggregated": (agg_context, aggregated),
                    "actual": (
                        actual_context, length, new_encoded,
                        [groups[tuple(x) if isinstance(x, tuple) else (x,)] for x in agg_index]
                    ),
                    "foreign_keys": all_fk_info,
                }
            else:
                encoded, _, _, _ = transformer.transform(table)
                out = {
                    "encoded": encoded
                }
            save_to(out, os.path.join(cache_dir, f"{tn}.pkl"))

    def _extend_till(self, table: str, till: str, keys: Sequence[str], cache_dir: str,
                     fitting: bool = True, queue: List[ForeignKey] = []) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        allowed_tables = self.order[:self.order.index(till)]
        raw = pd.read_csv(os.path.join(cache_dir, f"{table}.csv"))
        if self.transformers[table].config.foreign_keys:
            _, _, encoded, _ = self.actual_generation_for(table, cache_dir)
        else:
            encoded = self.standalone_encoded_for(table, cache_dir)
        core_columns = [f"_dim{i:02d}" for i in range(encoded.shape[-1])]
        core = pd.DataFrame(encoded, columns=core_columns, index=raw.index)
        core = pd.concat([raw.index.to_frame(False, "_id"), raw, core], axis=1)
        for fi, fk in enumerate(self.transformers[table].config.foreign_keys):
            if fk in queue:
                continue
            parent_raw, parent_context, parent_encoded = self._extend_till(
                fk.parent_table_name, till, fk.parent_column_names, cache_dir, fitting, queue + [fk]
            )
            parent_encoded = np.concatenate([parent_context, parent_encoded], axis=1)
            if table == till:
                parent_encoded = self._reduce_dims(
                    parent_encoded, fk.parent_table_name, fitting, queue + [fk], cache_dir, allowed_tables
                )
            parent_encoded = pd.DataFrame(
                parent_encoded, columns=[f"_dim{i:02d}_p{fi}" for i in range(parent_encoded.shape[-1])],
                index=np.arange(parent_encoded.shape[0])
            )
            parent_idx_df = parent_raw[fk.parent_column_names].rename(columns={
                p: c for p, c in zip(fk.parent_column_names, fk.child_column_names)
            })
            parent_encoded = pd.concat([parent_idx_df, parent_encoded], axis=1)
            core = core.merge(parent_encoded, on=fk.child_column_names, how="left").fillna(-1)

        for fi, fk in enumerate(self.children[table]):
            if fk.child_table_name not in allowed_tables or fk in queue:
                continue
            sibling_raw, sibling_context, sibling_encoded = self._extend_till(
                fk.child_table_name, till, fk.child_column_names, cache_dir, fitting, queue + [fk]
            )
            sibling_encoded = np.concatenate([sibling_context, sibling_encoded], axis=1)
            sibling_encoded = self._reduce_dims(
                sibling_encoded, fk.child_table_name, fitting, queue + [fk], cache_dir, allowed_tables
            )
            encoded_columns = [f"_dim{i:02d}_c{fi}" for i in range(sibling_encoded.shape[-1])]
            sibling_encoded = pd.DataFrame(
                sibling_encoded, columns=encoded_columns, index=np.arange(sibling_encoded.shape[0])
            )
            sibling_idx_df = sibling_raw[fk.child_column_names].rename(columns={
                c: p for c, p in zip(fk.child_column_names, fk.parent_column_names)
            })
            sibling_encoded = pd.concat([sibling_idx_df, sibling_encoded], axis=1)
            sibling_encoded_aggregated = sibling_encoded.groupby(fk.parent_column_names).aggregate(["mean", "std"])
            sibling_encoded_aggregated = sibling_encoded_aggregated.reset_index()
            sibling_encoded_aggregated.columns = pd.Index([
                f"{a}${b}" if b else a for a, b in sibling_encoded_aggregated.columns
            ])
            core = core.merge(
                sibling_encoded_aggregated, on=fk.parent_column_names, how="left"
            ).fillna(0)

        core = core.set_index("_id").loc[raw.index]
        raw_keys = raw[keys]
        context_columns = [c for c in core.columns if c.startswith("_dim") and c.endswith("_p0")]
        context = core[context_columns]
        encoded = core.drop(columns=context_columns + raw.columns.tolist())

        if fitting and table == till:
            parent_dims = []
            name_to_id = {
                c: i for i, c in enumerate(encoded.columns)
            }
            for fi in range(0, len(self.transformers[table].config.foreign_keys)):
                parent_dims.append([
                    name_to_id[n] for n in encoded.columns if n.endswith(f"_p{fi}") and n.startswith("_dim")
                ])
            self._parent_dims[table] = parent_dims
            self._core_dims[table] = [name_to_id[n] for n in core_columns]
        if raw_keys.shape[0] != encoded.values.shape[0]:
            raise RuntimeError(f"Extended table shape changed: {raw_keys.shape, raw.shape, encoded.shape}")  # TODO: remove
        return raw_keys, context.values, encoded.values

    def _reduce_dims(self, parent_encoded: np.ndarray, table: str, fitting: bool, queue: List[ForeignKey],
                     cache_dir: str, allowed_tables: List[str]) -> np.ndarray:
        if parent_encoded.shape[-1] > self.max_ctx_dim:
            queue_str = json.dumps([
                f"parent={qfk.parent_table_name}, child={qfk.child_table_name}, "
                f"columns={qfk.child_column_names}" for qfk in queue
            ])
            pca_name = f"{table}_{len(allowed_tables)}_{hashlib.sha1(queue_str.encode()).hexdigest()}"
            os.makedirs(os.path.join(cache_dir, "pca"), exist_ok=True)
            pca_path = os.path.join(cache_dir, "pca", f"{pca_name}.pkl")
            if fitting:
                if os.path.exists(pca_path):
                    raise FileExistsError(f"File for PCA already exists: {table} {allowed_tables[-1]} {queue}.")
                pca = PCA(n_components=self.max_ctx_dim)
                parent_encoded = pca.fit_transform(parent_encoded)
                save_to(pca, pca_path)
            else:
                pca = load_from(pca_path)
                parent_encoded = pca.transform(parent_encoded)
        return parent_encoded

    def fitted_size_of(self, table_name: str) -> int:
        return self._sizes_of[table_name]

    @classmethod
    def standalone_encoded_for(cls, table_name: str, cache_dir: str = "./cache") -> np.ndarray:
        return load_from(os.path.join(cache_dir, f"{table_name}.pkl"))["encoded"]

    @classmethod
    def degree_prediction_for(cls, table_name: str, fk_idx: int, cache_dir: str = "./cache") -> Tuple[
        np.ndarray, Optional[np.ndarray]
    ]:
        return load_from(os.path.join(cache_dir, f"{table_name}.pkl"))["foreign_keys"][fk_idx]["degree"]

    @classmethod
    def isna_indicator_prediction_for(cls, table_name: str, fk_idx: int, cache_dir: str = "./cache") -> Optional[Tuple[
        np.ndarray, Optional[np.ndarray]
    ]]:
        return load_from(os.path.join(cache_dir, f"{table_name}.pkl"))["foreign_keys"][fk_idx].get("isna")

    @classmethod
    def aggregated_generation_for(cls, table_name: str, cache_dir: str = "./cache") -> Tuple[
        np.ndarray, Optional[np.ndarray]
    ]:
        return load_from(os.path.join(cache_dir, f"{table_name}.pkl"))["aggregated"]

    @classmethod
    def actual_generation_for(cls, table_name: str, cache_dir: str = "./cache") -> Tuple[
        np.ndarray, np.ndarray, Optional[np.ndarray], Optional[List[np.ndarray]]
    ]:
        return load_from(os.path.join(cache_dir, f"{table_name}.pkl"))["actual"]

    def fk_matching_for(self, table_name: str, fk_idx: int, sampled_dir: str = "./cache") -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Optional[np.ndarray]], List[np.ndarray]
    ]:
        loaded = load_from(os.path.join(sampled_dir, f"{table_name}.pkl"))
        _, _, values, groups = loaded["actual"]
        values = values[:, self._parent_dims[table_name][fk_idx]]
        parent, degrees = loaded["foreign_keys"][fk_idx]["degree"]
        fk = self.transformers[table_name].config.foreign_keys[fk_idx]
        parent = self._reduce_dims(
            parent, fk.parent_table_name,
            False, [fk], self._fitted_cache_dir, self.order[:self.order.index(table_name)]
        )
        if values.shape[-1] != parent.shape[-1]:
            raise RuntimeError(f"The sizes to be matched are different: {values.shape}, {parent.shape}.")
        isnull = loaded["foreign_keys"][fk_idx]["isna"]
        if isnull is None:
            return_isna = np.zeros(values.shape[0], dtype=np.bool_)
        else:
            _, return_isna = isnull

        # collect prev FK values
        key_df = pd.DataFrame(index=pd.RangeIndex(values.shape[0]))
        for i, fk in enumerate(self.transformers[table_name].config.foreign_keys[:fk_idx]):
            parent_match = loaded["foreign_keys"][i]["match"]
            existing_vals = pd.read_csv(
                os.path.join(sampled_dir, f"{fk.parent_table_name}.csv")
            ).rename(
                columns={p: c for p, c in zip(fk.parent_column_names, fk.child_column_names)}
            )[fk.child_column_names]
            isna = np.isnan(parent_match.astype(np.float32))
            if isna.any():
                dummy_idx = existing_vals.shape[0]
                existing_vals.loc[dummy_idx, existing_vals.columns] = np.nan
                existing_vals = existing_vals.iloc[
                    np.where(isna, dummy_idx, parent_match)
                ].reset_index(drop=True)
            else:
                existing_vals = existing_vals.iloc[parent_match].reset_index(drop=True)
            if set(key_df.columns) & set(existing_vals.columns):
                same_cols = [*set(key_df.columns) & set(existing_vals.columns)]
                if key_df[same_cols][~return_isna].equals(
                        existing_vals[same_cols][~return_isna].astype(key_df[same_cols].dtypes)
                ):
                    new_cols = [*set(existing_vals.columns) - set(key_df.columns)]
                    key_df[new_cols] = existing_vals[new_cols]
                else:
                    raise RuntimeError(f"Overlapping FKs in previous FKs invalid ({table_name})[{fk_idx}].")
            else:
                key_df[fk.child_column_names] = existing_vals

        pools = [None] * values.shape[0]
        # overlapping FKs result in limited pools
        curr_fk = self.transformers[table_name].config.foreign_keys[fk_idx]
        prev_fk_cols = set()
        this_parent_raw = pd.read_csv(os.path.join(sampled_dir, f"{curr_fk.parent_table_name}.csv")).rename(
            columns={p: c for p, c in zip(curr_fk.parent_column_names, curr_fk.child_column_names)}
        )
        for i, fk in enumerate(self.transformers[table_name].config.foreign_keys[:fk_idx]):
            set1_cols = set(curr_fk.child_column_names) & set(fk.child_column_names)
            if set1_cols:
                set1_cols = [*set1_cols]
                existing_vals = key_df[set1_cols]
                this_parent_to_overlap_grouped = this_parent_raw[set1_cols].groupby(set1_cols)
                for ov, rows in existing_vals.groupby(set1_cols):
                    try:
                        this_parent_rows = this_parent_to_overlap_grouped.get_group(ov)
                        allowed_choices = this_parent_rows.index.values
                        for r in rows.index:
                            if pools[r] is None:
                                pools[r] = allowed_choices
                            else:
                                pools[r] = np.intersect1d(pools[r], allowed_choices)
                    except KeyError:
                        pass
            prev_fk_cols |= set(fk.child_column_names)
        curr_fk_cols = set(curr_fk.child_column_names)
        all_fk_cols = prev_fk_cols | curr_fk_cols

        # inequality results in limited pools
        for (a, b) in self.transformers[table_name].config.inequality:
            this_ineq_cols = set(a) | set(b)
            if this_ineq_cols <= all_fk_cols and not this_ineq_cols <= prev_fk_cols:
                for i, fk in enumerate(self.transformers[table_name].config.foreign_keys[:fk_idx]):
                    set1_cols = set(fk.child_column_names) & this_ineq_cols
                    set2_cols = this_ineq_cols - prev_fk_cols
                    if set1_cols:
                        if set1_cols & set(a):
                            set1_cols = [x for x in a if x in set1_cols]
                            set2_cols = [x for x in b if x in set2_cols]
                        else:
                            set1_cols = [x for x in b if x in set1_cols]
                            set2_cols = [x for x in a if x in set2_cols]
                        existing_vals = key_df[set1_cols]
                        this_parent_to_overlap_grouped = this_parent_raw[set2_cols].groupby(set2_cols)
                        for ov, rows in existing_vals.groupby(set1_cols):
                            try:
                                this_parent_rows = this_parent_to_overlap_grouped.get_group(ov)
                                disallowed_choices = this_parent_rows.index.values
                                for r in rows.index:
                                    if pools[r] is None:
                                        pools[r] = np.setdiff1d(np.arange(this_parent_raw.shape[0]), disallowed_choices)
                                    else:
                                        pools[r] = np.setdiff1d(pools[r], disallowed_choices)
                            except KeyError:
                                pass

        # uniqueness constraints of uniqueness groups
        uniqueness_groups = []
        if self.transformers[table_name].config.primary_key:
            pk_cols = set(self.transformers[table_name].config.primary_key)
            if pk_cols <= (curr_fk_cols | prev_fk_cols) and not pk_cols <= prev_fk_cols:
                core_cols = [*pk_cols & prev_fk_cols]
                for g, d in key_df.groupby(core_cols):
                    uniqueness_groups.append(d.index.values)

        return values, parent, degrees, return_isna, pools, uniqueness_groups

    def prepare_sampled_dir(self, sampled_dir: str):
        if os.path.exists(sampled_dir):
            shutil.rmtree(sampled_dir)
        os.makedirs(sampled_dir, exist_ok=True)
        if os.path.exists(os.path.join(self._fitted_cache_dir, "pca")):
            shutil.copytree(os.path.join(self._fitted_cache_dir, "pca"), os.path.join(sampled_dir, "pca"))

    @classmethod
    def save_standalone_encoded_for(cls, table_name: str, encoded: np.ndarray, sampled_dir: str = "./sampled"):
        save_to({"encoded": encoded}, os.path.join(sampled_dir, f"{table_name}.pkl"))

    @classmethod
    def save_degree_for(cls, table_name: str, fk_idx: int, degree: np.ndarray, sampled_dir: str = "./sampled"):
        loaded = load_from(os.path.join(sampled_dir, f"{table_name}.pkl"))
        x, _ = loaded["foreign_keys"][fk_idx]["degree"]
        loaded["foreign_keys"][fk_idx]["degree"] = x, degree

        if fk_idx == 0:
            a, b, c, d = loaded.get("actual", (None, None, None, None))
            non_zero_deg = degree > 0
            loaded["actual"] = a, degree[non_zero_deg], c, d
            non_zero_x = x[non_zero_deg]
            loaded["aggregated"] = non_zero_x, None

        save_to(loaded, os.path.join(sampled_dir, f"{table_name}.pkl"))

    def save_isna_indicator_for(self, table_name: str, fk_idx: int, isna: np.ndarray, sampled_dir: str = "./sampled"):
        loaded = load_from(os.path.join(sampled_dir, f"{table_name}.pkl"))
        x, _ = loaded["foreign_keys"][fk_idx]["isna"]
        loaded["foreign_keys"][fk_idx]["isna"] = x, isna
        a, b, encoded, d = loaded["actual"]
        encoded[np.ix_(isna, self._parent_dims[table_name][fk_idx])] = 0
        loaded["actual"] = a, b, encoded, d

        save_to(loaded, os.path.join(sampled_dir, f"{table_name}.pkl"))

    @classmethod
    def save_aggregated_info_for(cls, table_name: str, aggregated: np.ndarray, sampled_dir: str = "./sampled"):
        loaded = load_from(os.path.join(sampled_dir, f"{table_name}.pkl"))
        agg_context, _ = loaded["aggregated"]
        loaded["aggregated"] = agg_context, aggregated
        actual_context = np.concatenate([agg_context, aggregated], axis=1)
        _, length, _, _ = loaded["actual"]
        loaded["actual"] = actual_context, length, None, None

        save_to(loaded, os.path.join(sampled_dir, f"{table_name}.pkl"))

    @classmethod
    def save_actual_values_for(
            cls, table_name: str, values: np.ndarray, groups: List[np.ndarray], sampled_dir: str = "./sampled"
    ):
        loaded = load_from(os.path.join(sampled_dir, f"{table_name}.pkl"))
        context, length, _, _ = loaded["actual"]
        length = np.array([len(x) for x in groups])
        loaded["actual"] = context, length, values, groups
        for i, fk in enumerate(loaded["foreign_keys"]):
            isnull = fk["isna"]
            if isnull is not None:
                cids = np.repeat(np.arange(context.shape[0]), length.astype(int))
                loaded["foreign_keys"][i]["isna"] = np.concatenate([context[cids], values], axis=1), None
                break
        save_to(loaded, os.path.join(sampled_dir, f"{table_name}.pkl"))

    def save_matched_indices_for(self, table_name: str, fk_idx: int,
                                 indices: np.ndarray, sampled_dir: str = "./sampled"):
        loaded = load_from(os.path.join(sampled_dir, f"{table_name}.pkl"))
        loaded["foreign_keys"][fk_idx]["match"] = indices
        context, length, encoded, d = loaded["actual"]
        parent, _ = loaded["foreign_keys"][fk_idx]["degree"]
        isna = np.isnan(indices.astype(np.float32))

        if self._parent_dims[table_name][fk_idx]:
            fk = self.transformers[table_name].config.foreign_keys[fk_idx]
            encoded[np.ix_(np.nonzero(~isna)[0], self._parent_dims[table_name][fk_idx])] = self._reduce_dims(
                parent[indices[~isna].astype(np.int32)], fk.parent_table_name, False, [fk],
                self._fitted_cache_dir, self.order[:self.order.index(table_name)]
            )
        loaded["actual"] = context, length, encoded, d
        for i, fk in enumerate(loaded["foreign_keys"]):
            if i <= fk_idx:
                continue
            isnull = fk["isna"]
            if isnull is not None:
                cids = np.repeat(np.arange(context.shape[0]), length.astype(int))
                loaded["foreign_keys"][i]["isna"] = np.concatenate([context[cids], encoded], axis=1), None
                break
        save_to(loaded, os.path.join(sampled_dir, f"{table_name}.pkl"))

    def copy_fitted_for(self, table_name: str, sampled_dir: str = "./sampled"):
        shutil.copyfile(os.path.join(self._fitted_cache_dir, f"{table_name}.pkl"),
                        os.path.join(sampled_dir, f"{table_name}.pkl"))
        shutil.copyfile(os.path.join(self._fitted_cache_dir, f"{table_name}.csv"),
                        os.path.join(sampled_dir, f"{table_name}.csv"))

    def prepare_next_for(self, table_name: str, sampled_dir: str = "./cache"):
        if self.transformers[table_name].config.foreign_keys:
            _, aggregated = self.aggregated_generation_for(table_name, sampled_dir)
            _, _, encoded, indices = self.actual_generation_for(table_name, sampled_dir)
            _, deg = self.degree_prediction_for(table_name, 0, sampled_dir)
            foreign_keys = self.transformers[table_name].config.foreign_keys
            fk = foreign_keys[0]
            parent = pd.read_csv(os.path.join(sampled_dir, f"{fk.parent_table_name}.csv"))
            parent_idx = pd.MultiIndex.from_frame(parent[fk.parent_column_names].rename({
                p: c for p, c in zip(fk.parent_column_names, fk.child_column_names)
            }))[deg > 0]
            groups = {
                pi: idx for pi, idx in zip(parent_idx, indices)
            }
            recovered = self.transformers[table_name].inverse_transform(
                encoded[:, self._core_dims[table_name]], groups, aggregated, parent_idx
            )

            occurred_cols = set()
            if len(foreign_keys) > 1:
                loaded = load_from(os.path.join(sampled_dir, f"{table_name}.pkl"))
            else:
                loaded = None
            for i, fk in enumerate(foreign_keys):
                if i == 0:
                    occurred_cols |= set(fk.child_column_names)
                    continue
                new_cols = [c for c in fk.child_column_names if c not in occurred_cols]
                match_indices = loaded["foreign_keys"][i]["match"]
                parent_table = pd.read_csv(os.path.join(sampled_dir, f"{fk.parent_table_name}.csv"))
                dummy_index = parent_table.shape[0]
                parent_table.loc[dummy_index, parent_table.columns] = np.nan
                recovered.loc[:, new_cols] = parent_table.iloc[np.where(
                    np.isnan(match_indices.astype(np.float32)), dummy_index, match_indices
                )].rename(
                    columns={p: c for p, c in zip(fk.parent_column_names, fk.child_column_names)}
                )[new_cols].set_axis(recovered.index, axis=0)
                occurred_cols |= set(fk.child_column_names)
        else:
            encoded = self.standalone_encoded_for(table_name, sampled_dir)
            recovered = self.transformers[table_name].inverse_transform(encoded)
        recovered.to_csv(os.path.join(sampled_dir, f"{table_name}.csv"), index=False)

        table_idx = self.order.index(table_name)
        if table_idx >= len(self.order) - 1:
            return
        next_table_name = self.order[table_idx + 1]
        degrees = []
        for i, fk in enumerate(self.transformers[next_table_name].config.foreign_keys):
            parent_raw, parent_context, parent_encoded = self._extend_till(
                fk.parent_table_name, next_table_name, fk.parent_column_names, sampled_dir, False, [fk]
            )
            parent_extend_till = np.concatenate([parent_context, parent_encoded], axis=1)
            degrees.append(parent_extend_till)
        save_to({
            "foreign_keys": [{
                "degree": (x, None), "isna": (None, None) if y else None
            } for x, y in zip(degrees, self._nullable.get(next_table_name, []))]
        }, os.path.join(sampled_dir, f"{next_table_name}.pkl"))
