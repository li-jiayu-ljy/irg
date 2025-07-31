from sdv.metadata import MultiTableMetadata
from sdv.multi_table.base import BaseMultiTableSynthesizer
from sdv.single_table import CopulaGANSynthesizer
from sdv.sampling.independent_sampler import BaseIndependentSampler


class IndependentSynthesizer(BaseIndependentSampler, BaseMultiTableSynthesizer):
    def __init__(self, metadata: MultiTableMetadata, epochs = {}):
        BaseMultiTableSynthesizer.__init__(self, metadata)
        self._table_sizes = {}
        self._table_synthesizers = {
            t: CopulaGANSynthesizer(c, epochs=epochs.get(t, 300)) for t, c in metadata.tables.items()
        }
        BaseIndependentSampler.__init__(self, metadata, self._table_synthesizers, self._table_sizes)
        self._fitted = False

    def fit(self, tables):
        for t, s in self._table_synthesizers.items():
            s.fit(tables[t])
            self._table_sizes[t] = tables[t].shape[0]
        self._fitted = True

    def _add_foreign_key_columns(self, child_table, parent_table, child_name, parent_name):
        fk = None
        for rel in self.metadata.relationships:
            if rel["parent_table_name"] == parent_name and rel["child_table_name"] == child_name:
                fk = rel
                break
        if fk is None:
            raise ValueError()
        ccol = fk["child_foreign_key"]
        if ccol not in child_table.columns:
            child_table[ccol] = parent_table[fk["parent_primary_key"]].sample(
                n=child_table.shape[0], replace=True
            ).reset_index(drop=True)
