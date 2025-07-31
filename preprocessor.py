import pandas as pd
from tqdm import tqdm


class DataTransformer:
    def __init__(self):
        self.columns = {}
        self.col_order = []

    def fit(self, data: pd.DataFrame, id_cols: list = [], ref_cols: dict = {}, name: str = None):
        self.col_order = data.columns.tolist()
        for c in tqdm(data.columns, desc=f"Table {name if name is not None else ''} columns:"):
            col = data[c]
            if c in ref_cols:
                self.columns[c] = ref_cols[c]
            elif c in id_cols:
                self.columns[c] = {
                    "type": "id",
                    "is_int": pd.api.types.is_integer_dtype(col.dtype) or (pd.api.types.is_numeric_dtype(col.dtype)
                                                                           and (col - col.round()).abs().mean() < 1e-6)}
            else:
                col_descr = {}
                if (col.nunique() > 5 and pd.to_numeric(col.dropna(), errors="coerce").isna().all()
                        and not pd.to_datetime(col.dropna(), errors="coerce").isna().any()):
                    col = pd.to_datetime(col, errors="coerce")
                    col_descr["type"] = "datetime"
                    min_col = col.min()
                    col_descr["min"] = str(min_col)
                    col = (col - min_col).dt.total_seconds()
                if col.isna().any():
                    if pd.api.types.is_numeric_dtype(col.dtype):
                        col_descr["fillna"] = col.min() - 1
                    else:
                        col_descr["fillna"] = "<special-for-null>"
                self.columns[c] = col_descr

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        for c in self.col_order:
            col_descr = self.columns[c]
            col_type = col_descr.get("type", "normal")
            col = data[c]
            if col_type == "id":
                if col_descr["is_int"]:
                    col = col.astype("Int64")
            elif col_type == "datetime":
                col = pd.to_datetime(col, errors="coerce")
                col = (col - pd.to_datetime(col_descr["min"])).dt.total_seconds()
            if "fillna" in col_descr:
                col = col.fillna(col_descr["fillna"])
            data[c] = col
        return data[self.col_order]

    def to_dict(self):
        return {
            "columns": self.columns,
            "order": self.col_order
        }

    @classmethod
    def from_dict(cls, data):
        transformer = cls()
        transformer.columns = data["columns"]
        transformer.col_order = data["order"]
        return transformer
