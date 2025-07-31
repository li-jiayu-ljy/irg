import argparse
import os

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

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


class BECEvaluator(RelationalEvaluator):
    def __init__(self):
        super().__init__("bec", sdv_scale=0.2, models=["ind", "rctgan", "clava", "irg"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-shapes", dest="shape", action="store_false", default=True)
    parser.add_argument("--skip-schema", dest="schema", action="store_false", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    evaluator = BECEvaluator()
    if args.shape:
        evaluator.evaluate_shapes()
    if args.schema:
        evaluator.evaluate_schema()


if __name__ == '__main__':
    main()

