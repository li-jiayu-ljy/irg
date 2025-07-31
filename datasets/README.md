# General Information about Datasets

## General Data Processing Principles

1. We make sure that the relational schema is 100% accurate. Invalid data may be removed or converted to N/A.
2. Missing data are removed except for FK.
3. Date time columns are converted to numeric data.

Information of this processing is found in `processor.json` under each folder.

As for data processing on baseline models,

1. Composite PKs are ignored.
2. Composite FKs are converted to singular FKs by inserting auxiliary singular corresponding PK or candidate key.
3. NULL FKs are removed by inserting NULL parent.

## Implementation of Baselines

The data schema used for each baseline, including our model, IRG, can be found in `schema/` directory under the folder
for each dataset.