# ClavaDDPM

This is a modified version of ClavaDDPM that runs as our baselines.
The original README file is found in [README-original.md](./README-original.md).
Code is adapted from the [GitHub repository of the paper](https://github.com/weipang142857/ClavaDDPM).

## Modification

1. Create `process.py` to do automatic preparation of config files.
2. The preprocessing also prepares data whose primary key names and foreign key names follow the naming convention of 
   ClavaDDPM's constraints (must be `<table_name>_id`). All ID columns should be numeric in ClavaDDPM, which will also
   be processed.
3. For tables with multiple foreign keys sharing the same parent table, which is not inherently manageable by the 
   original implementation of ClavaDDPM but theoretically manageable by the algorithm, we implement the feature 
   engineering and corresponding pipeline changes. In order for the processing to work, one should make sure that no 
   table name ends with numbers.
4. In the pipeline, evaluation is skipped.
5. For FKs with NULL values, a dummy NULL parent is created. This column will have primary ID `NULL-KEY`, so this value
   should not be a real primary key value.
6. ClavaDDPM original code has a bug on `handle_multi_parent` on categorical values. We apply one-hot encoding on 
   these categorical values.