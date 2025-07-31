# RCTGAN

This is a modified version of RCTGAN that runs as our baselines.
The original README file is found in [README-original.md](./README-original.md).
Code is adapted from the [GitHub repository of the paper](https://github.com/croesuslab/RCTGAN).

## Modification

1. Version alignment with more recent `rdt` versions.
2. NULL keys will be handled on dataset basis in `run-rct.py` under each dataset.
3. Tables with only ID columns are not valid in RCTGAN, so we insert a placeholder unary column in it.
