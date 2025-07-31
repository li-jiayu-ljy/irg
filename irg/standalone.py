import os

import numpy as np

from .utils import REaLTabFormer, fit_transform_rtf, inverse_transform_rtf, update_epochs


def train_standalone(data: np.ndarray, model_dir: str, **kwargs):
    os.makedirs(model_dir, exist_ok=True)
    kwargs = update_epochs(data.shape[0], kwargs)
    model = REaLTabFormer(
        model_type="tabular", **kwargs,
        checkpoints_dir=os.path.join(model_dir, "ckpts"), samples_save_dir=os.path.join(model_dir, "samples")
    )
    data = fit_transform_rtf(data, model_dir)
    model.fit(data)
    model.save(os.path.join(model_dir, "final"))


def generate_standalone(n: int, model_dir: str, ) -> np.ndarray:
    model = REaLTabFormer.load_from_dir(os.path.join(model_dir, "final"))
    generated = model.sample(n)
    return inverse_transform_rtf(generated, model_dir)

