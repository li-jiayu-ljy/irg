import contextlib
import copy
import gc
import glob
import inspect
import itertools
import json
import logging
import os
import pickle
import random
import shutil
import threading
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cupy as cp
import dill
import numpy as np
import pandas as pd
import psutil
import torch
from datasets import Dataset
from dython.nominal import associations, cramers_v, identify_nominal_columns
from joblib import Parallel, delayed
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder
from realtabformer import REaLTabFormer as BaseREaLTabFormer, data_utils
from realtabformer.data_utils import (
    ModelFileName, ModelType, SpecialTokens, TEACHER_FORCING_PRE, encode_partition_numeric_col, encode_processed_column,
    process_datetime_data
)
from realtabformer.realtabformer import _validate_get_device
from realtabformer.rtf_datacollator import RelationalDataCollator
from realtabformer.rtf_trainer import ResumableTrainer
from tqdm import tqdm
from transformers import (
    EncoderDecoderConfig, EncoderDecoderModel, EarlyStoppingCallback, GPT2Config, GPT2LMHeadModel,
    LEDConfig, LEDForConditionalGeneration, PretrainedConfig,
    Seq2SeqTrainer, Seq2SeqTrainingArguments, Trainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
)
from transformers.models.led import modeling_led


data_utils.SPECIAL_COL_SEP = "%"


@dataclass(frozen=True)
class ColDataType:
    NUMERIC: str = "NUM"
    DATETIME: str = "DATETIME"
    CATEGORICAL: str = "CAT"

    @staticmethod
    def types():
        return [field.default for field in fields(ColDataType)]


data_utils.ColDataType = ColDataType


class LEDSeq2SeqModelOutput(modeling_led.LEDSeq2SeqModelOutput):
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


modeling_led.LEDSeq2SeqModelOutput = LEDSeq2SeqModelOutput


def get_positions(config: PretrainedConfig, encoder=True) -> int:
    if hasattr(config, "n_positions"):
        return config.n_positions
    if hasattr(config, "max_position_embeddings"):
        return config.max_position_embeddings
    key = f"max_{'en' if encoder else 'de'}coder_position_embeddings"
    if hasattr(config, key):
        return getattr(config, key)
    raise NotImplementedError(f"Config {config} positions isn't recognized.")


def set_positions(config: PretrainedConfig, val: int, encoder=True):
    try:
        config.n_positions = val
    except:
        pass
    if hasattr(config, "max_position_embeddings"):
        config.max_position_embeddings = val
    led_name = f"max_{'en' if encoder else 'de'}coder_position_embeddings"
    if hasattr(config, led_name):
        setattr(config, led_name, val)
    return config


class AdditionalEarlyStopCallback(TrainerCallback):
    def __init__(
            self, loss_threshold: float = 1e-6, early_stopping_patience: int = 1,
            early_stopping_threshold: Optional[float] = 0.0
    ):
        self.loss_threshold = loss_threshold
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.early_stopping_patience_counter = 0
        self.best_loss = torch.inf

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        loss_value = metrics.get("eval_loss")
        if loss_value is not None and self.loss_threshold > loss_value:
            control.should_training_stop = True

    def on_log(self, args, state, control, logs=None, **kwargs):
        loss = logs.get("loss")
        if loss is not None:
            if not np.isfinite(loss):
                raise RuntimeError("Training has diverged.")
            if loss < self.loss_threshold:
                control.should_training_stop = True
            elif state.global_step >= min(1000, args.warmup_steps):
                if loss < self.best_loss - self.early_stopping_threshold:
                    self.early_stopping_patience_counter = 0
                    self.best_loss = loss
                else:
                    self.early_stopping_patience_counter += 1
                if self.early_stopping_patience_counter >= self.early_stopping_patience:
                    control.should_training_stop = True

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        prev_losses = []
        if state.log_history:
            for entry in reversed(state.log_history):
                if "loss" in entry:
                    prev_losses.append(entry["loss"])
                if len(prev_losses) > self.early_stopping_patience:
                    break
        if prev_losses:
            self.best_loss = min(prev_losses)
            if not np.isfinite(self.best_loss):
                raise RuntimeError("Training has diverged.")
            if self.best_loss < self.loss_threshold:
                warnings.warn(f"Stop training due to best loss {self.best_loss}")
                control.should_training_stop = True
            elif (state.global_step >= min(1000, args.warmup_steps) and
                  len(prev_losses) > self.early_stopping_patience and all(
                loss >= prev_losses[-1] - self.early_stopping_threshold for loss in prev_losses[:-1]
            )):
                warnings.warn(f"Stop training due to patience reached {prev_losses}")
                control.should_training_stop = True


def make_relational_dataset(
    in_columns: list,
    out_columns: list,
    cache_path: str,
    vocab: dict,
    in_out_idx: dict,
    mask_rate=0,
    output_max_length: Optional[int] = None,
    return_token_type_ids: bool = False,
) -> Dataset:
    with open(os.path.join(cache_path, "in.csv"), 'r') as f:
        in_rows = sum(1 for _ in f) - 1
    encoder_dataset = Dataset.from_csv(
        os.path.join(cache_path, "in.csv"), num_proc=max(5, min(20, in_rows // 1000)), keep_in_memory=False,
        cache_dir=os.path.join(cache_path, "in-cache")
    )
    with open(os.path.join(cache_path, "out.csv"), 'r') as f:
        out_rows = sum(1 for _ in f) - 1
    decoder_dataset = Dataset.from_csv(
        os.path.join(cache_path, "out.csv"), num_proc=max(5, min(20, out_rows // 1000)), keep_in_memory=False,
        cache_dir=os.path.join(cache_path, "out-cache")
    )
    # Do not add [BOS] and [EOS] here. This will be handled
    # in the creation of the training_dataset in `get_relational_input_ids`.
    decoder_dataset = decoder_dataset.map(
        lambda example: get_input_ids_batched(
            example,
            vocab["decoder"],
            out_columns,
            mask_rate=mask_rate,
            return_label_ids=False,
            return_token_type_ids=return_token_type_ids,
            affix_bos=False,
            affix_eos=False,
        ),
        remove_columns=decoder_dataset.column_names,
        num_proc=max(5, min(20, out_rows // 1000)), desc="Decoder dataset", keep_in_memory=False, batched=True,
        batch_size=min(1000, out_rows // max(5, min(20, out_rows // 1000)))
    )

    training_dataset = encoder_dataset.map(
        lambda example: get_input_ids_batched(
            example,
            vocab["encoder"],
            in_columns,
            return_label_ids=False,
            return_token_type_ids=return_token_type_ids,
            affix_bos=True,
            affix_eos=True,
        ),
        remove_columns=encoder_dataset.column_names,
        num_proc=max(5, min(20, in_rows // 1000)), desc="Encoder dataset", keep_in_memory=False, batched=True,
        batch_size=min(1000, in_rows // max(5, min(20, in_rows // 1000)))
    )
    training_dataset = training_dataset.map(
        lambda example, idx: get_relational_input_ids_from_ids(
            example,
            idx,
            vocab,
            decoder_dataset,
            in_out_idx,
            output_max_length,
        ),
        with_indices=True,
        num_proc=max(5, min(20, in_rows // 1000)), desc="Encoder dataset connect", keep_in_memory=False
    )

    # If the output_max_length variable is specified, filter
    # observations that exceed this length. The
    # `get_relational_input_ids` should have set the
    # `labels` to None if the output exceeds `output_max_length`.
    if output_max_length:
        init_data_length = training_dataset.shape[0]

        training_dataset = training_dataset.filter(
            lambda example: example["labels"] is not None
        )

        removed_count = init_data_length - training_dataset.shape[0]
        if removed_count > 0:
            warnings.warn(
                f"A total of {removed_count} out of {init_data_length} has been removed from the training data because they exceeded the `output_max_length` of {output_max_length}."
            )

    return training_dataset


def process_numeric_data(
    series: pd.Series,
    max_len: int = 10,
    numeric_precision: int = 4,
    transform_data: Dict = None,
) -> Tuple[pd.Series, Dict]:
    is_transform = True

    if transform_data is None:
        transform_data = dict()
        is_transform = False

    if is_transform:
        warnings.warn(
            "Default values will be overridden because transform_data was passed..."
        )
        max_len = transform_data["max_len"]
        numeric_precision = transform_data["numeric_precision"]
    else:
        transform_data["max_len"] = max_len
        transform_data["numeric_precision"] = numeric_precision

    has_neg = series.min() < 0
    if pd.api.types.is_integer_dtype(series.dtype):
        is_int = True
    else:
        is_int = False

    # Get the most significant digit
    if is_transform:
        mx_sig = transform_data["mx_sig"]
        max_abs = None
    else:
        abs_num_series = series.abs()
        max_abs = abs_num_series.max()
        if max_abs == 0:
            max_abs = 1 if is_int else 1e-6
        mx_sig = int(-1 if is_int else (1 if max_abs < 10 else np.ceil(np.log10(max_abs))) + has_neg)
        transform_data["mx_sig"] = mx_sig

    if mx_sig <= 0:
        # The data has no decimal point.
        # Pad the data with leading zeros if not
        # aligned to the largest value.
        # We also don't apply the max_len to integral
        # valued data because it will basically
        # remove important information.
        if is_transform:
            zfill = transform_data["zfill"]
        else:
            zfill = int(max(1, np.ceil(np.log10(max_abs))) + has_neg)  # +1 for sign
            transform_data["zfill"] = zfill
        series = pd.Series(np.char.zfill(series.to_numpy(dtype='U'), zfill), index=series.index)
    else:
        # Make sure that we don't exessively truncate the data.
        # The max_len should be greater than the mx_sig.
        # Add a +1 to generate a minimum of tenth place resolution
        # for this data.
        assert max_len > (
            mx_sig + 1
        ), f"The target length {max_len} of the data doesn't include the numeric precision at {mx_sig}. Increase max_len to at least {max_len + (mx_sig + 2 - max_len)}."

        # Left align first based on the magnitude of the values.
        # We compute the difference in the most significant digits
        # of all values with respect to the largest value.
        # We then pad a leading zero to values with lower most significant
        # digits.
        # For example we have the values 1029.61 and 4.269. This will
        # determine that 1029.61 has the largest magnitude, with most significant
        # digit of 4. It will pad the value 4.269 with three zeros and convert it
        # to 0004.269.
        total_len = int(numeric_precision + 1 + mx_sig)
        series = pd.Series(
            np.char.mod(f'%0{total_len}.{numeric_precision}f', series.values), index=series.index
        ).str[:max_len]

        # We additionally apply left justify to align based on the trailing precision.
        # For example, we have 1029.61 and 0004.269 as values. This time we transform the first
        # value to become 1029.610 to align with the precision of the second value.
        if is_transform:
            ljust = transform_data["ljust"]
        else:
            ljust = min(max_len, total_len)
            transform_data["ljust"] = int(ljust)

    return series, transform_data


def tokenize_numeric_col(series: pd.Series, nparts=2, col_zfill=2):
    # After normalizing the numeric values, we then segment
    # them based on a fixed partition size (nparts).
    col = series.name
    lengths = series.str.len()
    max_len = lengths.max()

    if nparts > max_len > 2:
        # Allow minimum of 0-99 as acceptable singleton range.
        raise ValueError(
            f"Partition size {nparts} is greater than the value length {max_len}. Consider reducing the number of partitions..."
        )
    mx = lengths.min()

    tr = pd.concat([series.str[i : i + nparts] for i in range(0, mx, nparts)], axis=1)

    tr.columns = encode_partition_numeric_col(col, tr, col_zfill)

    return tr


def process_data(
    df: pd.DataFrame,
    numeric_max_len=10,
    numeric_precision=4,
    numeric_nparts=2,
    first_col_type=None,
    col_transform_data: Dict = None,
    target_col: str = None,
    base_idx=None
) -> Tuple[pd.DataFrame, Dict]:
    # This should receive a dataframe with dtypes that have already been
    # properly categorized between numeric and categorical.
    # Date type can be converted as UNIX timestamps.
    assert first_col_type in [None, ColDataType.CATEGORICAL, ColDataType.NUMERIC]

    df = df.copy()

    # Unify the variable for missing data
    df = df.fillna(pd.NA)

    # Force cast integral values to Int64Dtype dtype
    # to save precision if they are represented as float.
    for c in df:
        try:
            if pd.api.types.is_datetime64_any_dtype(df[c].dtype):
                # Don't cast datetime types.
                continue

            if pd.api.types.is_numeric_dtype(df[c].dtype):
                # Only cast if the column is explicitly numeric type.
                df[c] = df[c].astype(pd.Int64Dtype())
        except TypeError:
            pass
        except ValueError:
            pass

    if target_col is not None:
        assert (
            first_col_type is None
        ), "Implicit ordering of columns when teacher-forcing of target is used is not supported yet!"
        tf_col_name = f"{TEACHER_FORCING_PRE}_{target_col}"
        assert (
            tf_col_name not in df.columns
        ), f"The column name ({tf_col_name}) must not be in the raw data. Found instead..."

        target_ser = df[target_col].copy()
        target_ser.name = tf_col_name
        df = pd.concat([target_ser, df], axis=1)

    # Rename the columns to encode the original order by adding a suffix of increasing
    # integer values.
    num_cols = len(str(len(df.columns)))
    if base_idx is None:
        base_idx = 0
    col_idx = {col: f"{str(i + base_idx).zfill(num_cols)}" for i, col in enumerate(df.columns)}

    # Create a dataframe that will hold the processed data
    processed_series = []

    # Process numerical data
    numeric_cols = df.select_dtypes(include=np.number).columns

    if col_transform_data is None:
        col_transform_data = dict()

    for c in numeric_cols:
        col_name = encode_processed_column(col_idx[c], ColDataType.NUMERIC, c)
        _col_transform_data = col_transform_data.get(c)
        series, transform_data = process_numeric_data(
            df[c],
            max_len=numeric_max_len,
            numeric_precision=numeric_precision,
            transform_data=_col_transform_data,
        )
        if _col_transform_data is None:
            # This means that no transform data is available
            # before the processing.
            col_transform_data[c] = transform_data
        series.name = col_name
        processed_series.append(series)

    # Process datetime data
    datetime_cols = df.select_dtypes(include="datetime").columns

    for c in datetime_cols:
        col_name = encode_processed_column(col_idx[c], ColDataType.DATETIME, c)

        _col_transform_data = col_transform_data.get(c)
        series, transform_data = process_datetime_data(
            df[c],
            transform_data=_col_transform_data,
        )
        if _col_transform_data is None:
            # This means that no transform data is available
            # before the processing.
            col_transform_data[c] = transform_data
        series.name = col_name
        processed_series.append(series)

    processed_df = pd.concat([pd.DataFrame()] + processed_series, axis=1)

    if not processed_df.empty:
        # Combine the processed numeric and datetime data.
        processed_df = pd.concat(
            [
                tokenize_numeric_col(processed_df[col], nparts=numeric_nparts)
                for col in processed_df.columns
            ],
            axis=1,
        )

    # NOTE: The categorical data should be the last to be processed!
    categorical_cols = df.columns.difference(numeric_cols).difference(datetime_cols)

    if not categorical_cols.empty:
        # Process the rest of the data, assumed to be categorical values.
        processed_cat = df[categorical_cols].astype(str)
        cat_col_idx = pd.Series([col_idx[c] for c in categorical_cols])
        new_cat_col_names = (cat_col_idx + data_utils.SPECIAL_COL_SEP + f"{ColDataType.CATEGORICAL}" +
                             data_utils.SPECIAL_COL_SEP + pd.Series(categorical_cols))
        processed_df = pd.concat(
            [
                processed_df,
                processed_cat.set_axis(new_cat_col_names.tolist(), axis=1)
            ],
            axis=1,
        )

    # Get the different sets of column types
    is_cat = processed_df.columns.str.contains(ColDataType.CATEGORICAL)
    cat_cols = processed_df.columns[is_cat]
    numeric_cols = processed_df.columns[~is_cat]

    if first_col_type == ColDataType.CATEGORICAL:
        df = processed_df[cat_cols.union(numeric_cols, sort=False)]
    elif first_col_type == ColDataType.NUMERIC:
        df = processed_df[numeric_cols.union(cat_cols, sort=False)]
    else:
        # Reorder columns to the original order
        df = processed_df[np.sort(processed_df.columns)]

    df = df.columns.values.reshape((1, -1)) + data_utils.SPECIAL_COL_SEP + df

    return df, col_transform_data


def build_large_vocab(cache_path, chunk_size, columns, part):
    id2token = {}
    curr_id = 0
    special_tokens = SpecialTokens.tokens()
    if special_tokens:
        id2token.update(dict(enumerate(special_tokens)))
        curr_id = max(id2token) + 1
    column_token_ids = {}

    def obtain_unique(series):
        return series.name, series.unique()
    col_uniques = {col: set() for col in columns}
    for chunk in pd.read_csv(os.path.join(cache_path, f"{part}.csv"), chunksize=10_000_000 // len(columns)):
        chunk_uniques = Parallel(n_jobs=20)(
            delayed(obtain_unique)(chunk[col]) for col in columns
        )
        for col, vals in chunk_uniques:
            col_uniques[col].update(vals)

    def collect_unique(col, uniques, dtype):
        return col, np.sort(np.array([*uniques], dtype=dtype))
    unique_values = Parallel(n_jobs=20)(
        delayed(collect_unique)(col, col_uniques[col], chunk[col].dtype)
        for col in tqdm(columns, desc="Sorting unique values")
    )

    pbar = tqdm(desc=f"Preparing {part}put vocabulary", total=len(columns))
    for col, values in unique_values:
        new_id2token = dict(enumerate(values, curr_id))
        id2token.update(new_id2token)
        new_curr_id = max(new_id2token) + 1
        column_token_ids[col] = list(range(curr_id, new_curr_id))
        curr_id = new_curr_id
        pbar.update()
    token2id = {v: k for k, v in id2token.items()}
    return dict(
        id2token=id2token,
        token2id=token2id,
        column_token_ids=column_token_ids,
    )


def get_input_ids_batched(
    batch,
    vocab: Dict,
    columns: list,
    mask_rate: float = 0,
    return_label_ids: Optional[bool] = True,
    return_token_type_ids: Optional[bool] = False,
    affix_bos: Optional[bool] = True,
    affix_eos: Optional[bool] = True,
) -> Dict:
    # Raise an assertion error while the implementation
    # is not yet ready.
    assert return_token_type_ids is False
    input_ids: list[np.ndarray] = []
    token_type_ids: list[int] = []
    batch_size = len(batch[columns[0]])

    if affix_bos:
        input_ids.append(np.full(batch_size, vocab["token2id"][SpecialTokens.BOS], dtype=np.int32))
        if return_token_type_ids:
            token_type_ids.append(vocab["token2id"][SpecialTokens.SPTYPE])

    vocab_token2id = vocab["token2id"]
    masked = np.random.random((batch_size, len(columns))) < mask_rate
    col_names = pd.Series(columns).str.extract(
        f"([0-9]+{data_utils.SPECIAL_COL_SEP}({'|'.join(ColDataType.types())}))"
    )[0]
    for j, k in enumerate(columns):
        token_ids = [vocab_token2id.get(token, vocab_token2id[SpecialTokens.UNK]) for token in batch[k]]
        token_ids = np.array(token_ids, dtype=np.int32)
        if mask_rate > 0:
            token_ids[masked[:, j]] = vocab_token2id[SpecialTokens.RMASK]
        input_ids.append(token_ids)
        if return_token_type_ids:
            col_name = col_names[j]
            token_type_ids.append(vocab["token2id"][col_name])

    if affix_eos:
        input_ids.append(np.full(batch_size, vocab["token2id"][SpecialTokens.EOS], dtype=np.int32))
        if return_token_type_ids:
            token_type_ids.append(vocab["token2id"][SpecialTokens.SPTYPE])

    input_ids = np.stack(input_ids).T.tolist()
    data = dict(input_ids=input_ids)

    if return_label_ids:
        data["label_ids"] = input_ids

    if return_token_type_ids:
        data["token_type_ids"] = np.array(token_type_ids).reshape((1, -1)).repeat(batch_size, axis=0).tolist()

    return data


def get_relational_input_ids_from_ids(
    example,
    input_idx,
    vocab,
    output_dataset,
    in_out_idx,
    output_max_length: Optional[int] = None,
) -> dict:
    # Start with 2 to take into account the [BOS] and [EOS] tokens
    sequence_len = 2

    # Build the input_ids for the encoder
    input_ids = example["input_ids"]
    token_type_ids = example.get("token_type_ids")

    # Build the label_ids for the decoder
    output_idx = in_out_idx[input_idx]

    valid = True

    label_ids = [vocab["decoder"]["token2id"][SpecialTokens.BOS]]
    if len(output_idx) > 0:
        for ids in output_dataset.select(output_idx)["input_ids"]:
            # Pad each observation with the [BMEM] and [EMEM] tokens

            tmp_label_ids = [vocab["decoder"]["token2id"][SpecialTokens.BMEM]]
            tmp_label_ids.extend(ids)
            tmp_label_ids.append(vocab["decoder"]["token2id"][SpecialTokens.EMEM])

            if output_max_length:
                if (sequence_len + len(tmp_label_ids)) > output_max_length:
                    # This exceeds the expected limit.
                    # Drop this observation.
                    valid = False
                    break

            label_ids.extend(tmp_label_ids)
            sequence_len += len(tmp_label_ids)

    label_ids.append(vocab["decoder"]["token2id"][SpecialTokens.EOS])

    payload = dict(
        input_ids=input_ids,
        # The variable `labels` is used in the EncoderDecoder model
        # instead of `label_ids`.
        labels=label_ids if valid else None,
    )

    if token_type_ids is not None:
        payload["token_type_ids"] = token_type_ids

    return payload


def relational_to_led_config(relational_config):
    combined_config: LEDConfig = copy.deepcopy(relational_config.encoder)
    combined_config.is_encoder_decoder = True
    combined_config.decoder_layers = relational_config.decoder.decoder_layers
    combined_config.decoder_attention_heads = relational_config.decoder.decoder_attention_heads
    combined_config.decoder_ffn_dim = relational_config.decoder.decoder_ffn_dim
    combined_config.decoder_start_token_id = relational_config.decoder_start_token_id
    combined_config.max_decoder_position_embeddings = (
        relational_config.decoder.max_decoder_position_embeddings)
    combined_config.add_cross_attention = True
    combined_config.bos_token_id = relational_config.bos_token_id
    combined_config.eos_token_id = relational_config.eos_token_id
    combined_config.pad_token_id = relational_config.pad_token_id
    combined_config.vocab_size = max(combined_config.vocab_size, relational_config.decoder.vocab_size)
    return combined_config


class REaLTabFormer(BaseREaLTabFormer):
    # allow LED relational config

    def __init__(
            self, *args, early_stopping_threshold: float = 1e-3, n_critic: int = 5,
            learning_rate: float = 2e-4, lr_scheduler_type: str = "cosine", adam_epsilon: float = 1e-6, **kwargs
    ):
        self.n_critic = n_critic
        super().__init__(
            *args, **kwargs, early_stopping_threshold=early_stopping_threshold, learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type, adam_epsilon=adam_epsilon
        )

    def _split_train_eval_dataset(self, dataset: Dataset):
        test_size = 1 - self.train_size
        if test_size > 0:
            dataset = dataset.train_test_split(
                test_size=test_size, seed=self.random_state
            )
            dataset["train_dataset"] = dataset.pop("train")
            dataset["eval_dataset"] = dataset.pop("test")
            # Override `metric_for_best_model` from "loss" to "eval_loss"
            self.training_args_kwargs["metric_for_best_model"] = "eval_loss"
            # Make this explicit so that no assumption is made on the
            # direction of the metric improvement.
            self.training_args_kwargs["greater_is_better"] = False
        else:
            dataset = dict(train_dataset=dataset)
            self.training_args_kwargs["evaluation_strategy"] = "no"
            self.training_args_kwargs["load_best_model_at_end"] = False

        steps_per_epoch = len(dataset["train_dataset"]) // (self.batch_size * torch.cuda.device_count())
        self.training_args_kwargs["eval_steps"] = min(
            max(self.training_args_kwargs["eval_steps"], steps_per_epoch), 1000
        )
        self.training_args_kwargs["save_steps"] = self.training_args_kwargs["eval_steps"]
        self.training_args_kwargs["warmup_steps"] = min(
            10000, min(5, self.training_args_kwargs.get("num_train_epochs", 5)) * steps_per_epoch
        )
        return dataset

    def _set_up_relational_coder_configs(self) -> None:
        def _get_coder(coder_name) -> Union[GPT2Config, LEDConfig]:
            return getattr(self.relational_config, coder_name)

        for coder_name in ["encoder", "decoder"]:
            coder = _get_coder(coder_name)

            coder.bos_token_id = self.vocab[coder_name]["token2id"][SpecialTokens.BOS]
            coder.eos_token_id = self.vocab[coder_name]["token2id"][SpecialTokens.EOS]
            coder.pad_token_id = self.vocab[coder_name]["token2id"][SpecialTokens.PAD]
            coder.vocab_size = len(self.vocab[coder_name]["id2token"])

            if coder_name == "decoder":
                self.relational_config.bos_token_id = coder.bos_token_id
                self.relational_config.eos_token_id = coder.eos_token_id
                self.relational_config.pad_token_id = coder.pad_token_id
                self.relational_config.decoder_start_token_id = coder.eos_token_id

            # Make sure that we have at least the number of
            # columns in the transformed data as positions.
            # This will prevent runtime error.
            # `RuntimeError: CUDA error: device-side assert triggered`
            assert self.relational_max_length
            if (
                coder_name == "decoder" and
                get_positions(coder, encoder=False) < self.relational_max_length
            ):
                coder = set_positions(coder, 128 + self.relational_max_length, encoder=False)
            elif coder_name == "encoder" and get_positions(coder) < len(self.vocab[coder_name]["column_token_ids"]):
                positions = 128 + len(self.vocab[coder_name]["column_token_ids"])
                coder = set_positions(coder, positions)

        # This must be set to True for the EncoderDecoderModel to work at least
        # with GPT2 as the decoder.
        self.relational_config.decoder.add_cross_attention = True

    def _fit_relational(
        self, out_df: pd.DataFrame, in_df: pd.DataFrame, join_on: str, device="cuda", resume_from_checkpoint=None
    ):
        # bert2bert.config.decoder_start_token_id = tokenizer.cls_token_id
        # bert2bert.config.eos_token_id = tokenizer.sep_token_id
        # bert2bert.config.pad_token_id = tokenizer.pad_token_id
        # bert2bert.config.vocab_size = bert2bert.config.encoder.vocab_size

        # All join values in the out_df must be present in the in_df.
        assert len(set(out_df[join_on].unique()).difference(in_df[join_on])) == 0

        # Get the list of index of observations that are related based on
        # the join_on variable.
        common_out_idx = (
            out_df.reset_index(drop=True)
            .groupby(join_on)
            .apply(lambda x: x.index.to_list())
        )

        # Track the mapping of index from input to the list of output indices.
        in_out_idx = pd.Series(
            # Reset the index so that we are sure that the index ids are set properly.
            dict(in_df[join_on].reset_index(drop=True).items())
        ).map(lambda x: common_out_idx.get(x, []))

        # Remove the unique id column from the in_df and the out_df
        in_df = in_df.drop(join_on, axis=1)
        out_df = out_df.drop(join_on, axis=1)
        cache_path = str(self.checkpoints_dir).replace("ckpts", "cache-data")
        os.makedirs(cache_path, exist_ok=True)

        self._extract_column_info(out_df)
        max_size = min(10_000_000, out_df.shape[0] * 20)
        if out_df.size > max_size:
            self.col_transform_data, chunk_size, self.processed_columns, out_df = self._process_large_data(
                out_df, cache_path, max_size, "out"
            )
            self.vocab["decoder"] = build_large_vocab(cache_path, chunk_size, self.processed_columns, "out")
        else:
            out_df, self.col_transform_data = process_data(
                out_df,
                numeric_max_len=self.numeric_max_len,
                numeric_precision=self.numeric_precision,
                numeric_nparts=self.numeric_nparts,
            )
            self.processed_columns = out_df.columns.to_list()
            self.vocab["decoder"] = self._generate_vocab(out_df)
            out_df.to_csv(os.path.join(cache_path, "out.csv"), index=False)
        self.relational_col_size = len(self.processed_columns)
        out_shape = out_df.shape[0], self.relational_col_size
        del out_df
        gc.collect()

        # NOTE: the index starts at zero, but should be adjusted
        # to account for the special tokens. For relational data,
        # the index should start at 3 ([[EOS], [BOS], [BMEM]]).
        self.col_idx_ids = {
            ix: self.vocab["decoder"]["column_token_ids"][col]
            for ix, col in enumerate(self.processed_columns)
        }

        # Add these special tokens at specific key values
        # which are used in `REaLSampler._get_relational_col_idx_ids`
        self.col_idx_ids[-1] = [
            self.vocab["decoder"]["token2id"][SpecialTokens.BMEM],
            self.vocab["decoder"]["token2id"][SpecialTokens.EOS],
        ]
        self.col_idx_ids[-2] = [self.vocab["decoder"]["token2id"][SpecialTokens.EMEM]]

        max_size = min(1_000_000, in_df.shape[0] * 20)
        if in_df.size > max_size:
            self.in_col_transform_data, chunk_size, in_columns, in_df = self._process_large_data(
                in_df, cache_path, max_size, "in"
            )
            if self.parent_vocab is None:
                self.vocab["encoder"] = build_large_vocab(cache_path, chunk_size, in_columns, "in")
            else:
                self.vocab["encoder"] = self.parent_vocab
            in_shape = in_df.shape[0], len(in_columns)
        else:
            in_df, self.in_col_transform_data = process_data(
                in_df,
                numeric_max_len=self.numeric_max_len,
                numeric_precision=self.numeric_precision,
                numeric_nparts=self.numeric_nparts,
                col_transform_data=self.parent_col_transform_data,
            )
            if self.parent_vocab is None:
                self.vocab["encoder"] = self._generate_vocab(in_df)
            else:
                self.vocab["encoder"] = self.parent_vocab
            in_df.to_csv(os.path.join(cache_path, "in.csv"), index=False)
            in_columns = in_df.columns.tolist()
            in_shape = in_df.shape
        del in_df
        gc.collect()

        # Load the dataframe into a HuggingFace Dataset
        # torch.save(dict(vocab=self.vocab,
        #     in_out_idx=in_out_idx,
        #     output_max_length=self.output_max_length,
        #     mask_rate=self.mask_rate,
        #     return_token_type_ids=False,), os.path.join(cache_path, "init-dataset.pkl"))
        dataset = make_relational_dataset(
            # in_df=in_df,
            # out_df=out_df,
            in_columns=in_columns,
            out_columns=self.processed_columns,
            cache_path=cache_path,
            vocab=self.vocab,
            in_out_idx=in_out_idx,
            output_max_length=self.output_max_length,
            mask_rate=self.mask_rate,
            return_token_type_ids=False,
        )

        # Compute the longest sequence of labels in the dataset and add a buffer of 1.
        self.relational_max_length = (
            max(
                dataset.map(
                    lambda example: dict(length=len(example["labels"])),
                    num_proc=min(10, max(1, len(dataset) // 1000))
                )[
                    "length"
                ]
            )
            + 1
        )

        # Create train-eval split if specified
        dataset = self._split_train_eval_dataset(dataset)

        enc_configs = [
            GPT2Config(n_layer=6), LEDConfig(encoder_layers=6),
            GPT2Config(n_layer=6, n_embd=256, n_inner=1024, n_head=8),
            LEDConfig(
                encoder_layers=6, encoder_ffn_dim=1024, encoder_attention_heads=8, d_model=256,
                attention_window=256
            ),
            GPT2Config(n_layer=4, n_embd=128, n_inner=384, n_head=8),
            LEDConfig(
                encoder_layers=4, d_model=128, encoder_ffn_dim=384, encoder_attention_heads=8,
                attention_window=128
            )
        ]
        dec_configs = [
            GPT2Config(n_layer=6),
            LEDConfig(is_decoder=True, add_cross_attention=True, decoder_layers=6),
            GPT2Config(n_layer=6, n_embd=256, n_inner=1024, n_head=8),
            LEDConfig(
                decoder_layers=6, decoder_ffn_dim=1024, decoder_attention_heads=8, d_model=256,
                is_decoder=True, add_cross_attention=True, attention_window=256
            ),
            GPT2Config(n_layer=4, n_embd=128, n_inner=384, n_head=8),
            LEDConfig(
                decoder_layers=4, d_model=128, decoder_ffn_dim=384, decoder_attention_heads=8,
                is_decoder=True, add_cross_attention=True, attention_window=128
            )
        ]
        ee = None
        for dec_config in dec_configs:
            for enc_config in enc_configs:
                if dec_config.model_type != enc_config.model_type:
                    continue
                if len(in_columns) > 1024 and dec_config.model_type != "led":
                    continue
                self.relational_config = EncoderDecoderConfig(
                    encoder=enc_config.to_dict().copy(), decoder=dec_config.to_dict().copy()
                )
                try:
                    trainer = self._run_relational_trainer(device, dataset, resume_from_checkpoint)
                    shutil.rmtree(cache_path)
                    return trainer
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    ee = e
                    if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e) or "batch size" in str(
                        e):
                        if os.path.exists(self.checkpoints_dir):
                            shutil.rmtree(self.checkpoints_dir)
                        gc.collect()
                        torch.cuda.empty_cache()
                    else:
                        raise e
        raise ee

    def _process_large_data(self, df, cache_path, max_size, part):
        chunk_size = int(np.ceil(max_size / df.shape[0]))
        csv_paths = []
        in_columns = []
        chunks = []
        for i in range(int(np.ceil(df.shape[-1] / chunk_size))):
            selected_chunk = df.iloc[:, :chunk_size]
            df = df.drop(columns=selected_chunk.columns)
            chunks.append(selected_chunk)
        all_in_col_transform_data = {}
        def process_chunk(i, selected_chunk):
            processed_in_df, this_in_col_transform = process_data(
                selected_chunk,
                numeric_max_len=self.numeric_max_len,
                numeric_precision=self.numeric_precision,
                numeric_nparts=self.numeric_nparts,
                col_transform_data=self.parent_col_transform_data,
                base_idx=i * chunk_size
            )
            csv_path = os.path.join(cache_path, f"temp-{part}-{i}.csv")
            processed_in_df.to_csv(csv_path, index=False)
            return this_in_col_transform, csv_path, processed_in_df.columns.tolist()

        pbar = tqdm(
            enumerate(chunks), desc=f"Processing {part}put data [chunksize={chunk_size} x {len(chunks)}]",
            total=len(chunks)
        )
        results = Parallel(n_jobs=min(20, len(chunks)), verbose=10)(
            delayed(process_chunk)(i, selected_chunk)
            for i, selected_chunk in pbar
        )
        for t, p, c in results:
            all_in_col_transform_data.update(t)
            csv_paths.append(p)
            in_columns.extend(c)
        row_chunk_size = 10000
        input_files = [open(path, 'r') for path in csv_paths]
        headers = [f.readline().strip() for f in input_files]
        merged_header = ",".join(headers)
        output_file = open(os.path.join(cache_path, f"{part}.csv"), 'w')
        output_file.write(merged_header + '\n')
        total_lines = df.shape[0]
        pbar = tqdm(total=total_lines, desc="Combining chunks")
        caches = []
        for i in range(total_lines):
            lines = [f.readline() for f in input_files]
            combined = ",".join(line.strip() for line in lines)
            caches.append(combined)
            if (i + 1) % row_chunk_size == 0:
                output_file.write("\n".join([*caches, ""]))
                caches = []
            pbar.update()

        if caches:
            output_file.write("\n".join([*caches, ""]))

        # Clean up
        for f in input_files:
            f.close()
        output_file.close()
        for path in csv_paths:
            os.remove(path)
        return all_in_col_transform_data, chunk_size, in_columns, df

    @classmethod
    def load_from_dir(cls, path: Union[str, Path]):
        if not os.path.exists(os.path.join(path, "rtf_config.json")):
            path = glob.glob(f"{path}/*")[0]
        if isinstance(path, str):
            path = Path(path)

        config_file = path / ModelFileName.rtf_config_json
        model_file = path / ModelFileName.rtf_model_pt

        assert path.is_dir(), f"Directory {path} does not exist."
        assert config_file.exists(), f"Config file {config_file} does not exist."
        assert model_file.exists(), f"Model file {model_file} does not exist."

        # Load the saved attributes
        rtf_attrs = json.loads(config_file.read_text())

        # Create new REaLTabFormer model instance
        try:
            realtf = cls(model_type=rtf_attrs["model_type"])
        except KeyError:
            # Back-compatibility for saved models
            # before the support for relational data
            # was implemented.
            realtf = cls(model_type="tabular")

        # Set all attributes and handle the
        # special case for the GPT2Config.
        for k, v in rtf_attrs.items():
            if k == "gpt_config":
                # Back-compatibility for saved models
                # before the support for relational data
                # was implemented.
                v = GPT2Config.from_dict(v)
                k = "tabular_config"

            elif k == "tabular_config":
                v = GPT2Config.from_dict(v)

            elif k == "relational_config":
                v = EncoderDecoderConfig.from_dict(v)

            elif k in ["checkpoints_dir", "samples_save_dir"]:
                v = Path(v)

            elif k == "vocab":
                if realtf.model_type == ModelType.tabular:
                    # Cast id back to int since JSON converts them to string.
                    v["id2token"] = {int(ii): vv for ii, vv in v["id2token"].items()}
                elif realtf.model_type == ModelType.relational:
                    v["encoder"]["id2token"] = {
                        int(ii): vv for ii, vv in v["encoder"]["id2token"].items()
                    }
                    v["decoder"]["id2token"] = {
                        int(ii): vv for ii, vv in v["decoder"]["id2token"].items()
                    }
                else:
                    raise ValueError(f"Invalid model_type: {realtf.model_type}")

            elif k == "col_idx_ids":
                v = {int(ii): vv for ii, vv in v.items()}

            setattr(realtf, k, v)

        # Implement back-compatibility for REaLTabFormer version < 0.0.1.8.2
        # since the attribute `col_idx_ids` is not implemented before.
        if "col_idx_ids" not in rtf_attrs:
            if realtf.model_type == ModelType.tabular:
                realtf.col_idx_ids = {
                    ix: realtf.vocab["column_token_ids"][col]
                    for ix, col in enumerate(realtf.processed_columns)
                }
            elif realtf.model_type == ModelType.relational:
                # NOTE: the index starts at zero, but should be adjusted
                # to account for the special tokens. For relational data,
                # the index should start at 3 ([[EOS], [BOS], [BMEM]]).
                realtf.col_idx_ids = {
                    ix: realtf.vocab["decoder"]["column_token_ids"][col]
                    for ix, col in enumerate(realtf.processed_columns)
                }

                # Add these special tokens at specific key values
                # which are used in `REaLSampler._get_relational_col_idx_ids`
                realtf.col_idx_ids[-1] = [
                    realtf.vocab["decoder"]["token2id"][SpecialTokens.BMEM],
                    realtf.vocab["decoder"]["token2id"][SpecialTokens.EOS],
                ]
                realtf.col_idx_ids[-2] = [
                    realtf.vocab["decoder"]["token2id"][SpecialTokens.EMEM]
                ]

        # Load model weights
        if realtf.model_type == ModelType.tabular:
            realtf.model = GPT2LMHeadModel(realtf.tabular_config)
        elif realtf.model_type == ModelType.relational:
            if realtf.relational_config.encoder.model_type == "led":
                combined_config = relational_to_led_config(realtf.relational_config)
                realtf.model = LEDForConditionalGeneration(combined_config)
            else:
                realtf.model = EncoderDecoderModel(realtf.relational_config)
        else:
            raise ValueError(f"Invalid model_type: {realtf.model_type}")

        realtf.model.load_state_dict(
            torch.load(model_file.as_posix(), map_location="cpu")
        )

        return realtf

    def _run_relational_trainer(self, device, dataset, resume_from_checkpoint):

        # Set up the config and the model
        self._set_up_relational_coder_configs()

        # Build the model.
        if self.relational_config.encoder.model_type == "led" and self.relational_config.decoder.model_type == "led":
            combined_config = relational_to_led_config(self.relational_config)
            self.model = LEDForConditionalGeneration(combined_config)
        else:
            self.model = EncoderDecoderModel(self.relational_config)
        if self.parent_gpt2_state_dict is not None:
            pretrain_load = self.model.encoder.load_state_dict(
                self.parent_gpt2_state_dict, strict=False
            )
            assert (
                not pretrain_load.missing_keys
            ), "There should be no missing_keys after loading the pretrained GPT2 state!"

            if self.freeze_parent_model:
                # We freeze the weights if we use the pretrained
                # parent table model.
                for param in self.model.encoder.parameters():
                    param.requires_grad = False

        # Tell pytorch to run this model on the GPU.
        device = torch.device(device)
        if device == torch.device("cuda"):
            self.model.cuda()

        # Set TrainingArguments and the Seq2SeqTrainer
        training_args_kwargs = dict(self.training_args_kwargs)

        default_args_kwargs = dict(
            # predict_with_generate=True,
            # warmup_steps=2000,
            fp16=(
                device == torch.device("cuda")
            ),  # Use fp16 by default if using cuda device
            auto_find_batch_size=True
        )

        for k, v in default_args_kwargs.items():
            if k not in training_args_kwargs:
                training_args_kwargs[k] = v

        callbacks = None
        if training_args_kwargs["load_best_model_at_end"]:
            callbacks = [
                EarlyStoppingCallback(
                    self.early_stopping_patience, self.early_stopping_threshold
                ),
                AdditionalEarlyStopCallback(
                    early_stopping_patience=self.early_stopping_patience,
                    early_stopping_threshold=self.early_stopping_threshold
                )
            ]
        else:
            callbacks = [AdditionalEarlyStopCallback(
                early_stopping_patience=self.early_stopping_patience,
                early_stopping_threshold=self.early_stopping_threshold
            )]

        # instantiate trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=Seq2SeqTrainingArguments(**training_args_kwargs),
            callbacks=callbacks,
            data_collator=RelationalDataCollator(),
            **dataset,
        )
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        return trainer

    def _build_tabular_trainer(
        self,
        device="cuda",
        num_train_epochs: int = None,
        target_epochs: int = None,
    ) -> Trainer:
        device = torch.device(device)

        # Set TrainingArguments and the Trainer
        logging.info("Set up the TrainingArguments and the Trainer...")
        training_args_kwargs: Dict[str, Any] = dict(self.training_args_kwargs)

        default_args_kwargs = dict(
            fp16=(
                device == torch.device("cuda")
            ),  # Use fp16 by default if using cuda device
        )

        for k, v in default_args_kwargs.items():
            if k not in training_args_kwargs:
                training_args_kwargs[k] = v

        if num_train_epochs is not None:
            training_args_kwargs["num_train_epochs"] = num_train_epochs

        # # NOTE: The `ResumableTrainer` will default to its original
        # # behavior (Trainer) if `target_epochs`` is None.
        # # Set the `target_epochs` to `num_train_epochs` if not specified.
        # if target_epochs is None:
        #     target_epochs = training_args_kwargs.get("num_train_epochs")

        callbacks = None
        if training_args_kwargs["load_best_model_at_end"]:
            callbacks = [
                EarlyStoppingCallback(
                    self.early_stopping_patience, self.early_stopping_threshold
                ),
                AdditionalEarlyStopCallback(
                    early_stopping_patience=self.early_stopping_patience,
                    early_stopping_threshold=self.early_stopping_threshold
                )
            ]
        else:
            callbacks = [AdditionalEarlyStopCallback(
                early_stopping_patience=self.early_stopping_patience,
                early_stopping_threshold=self.early_stopping_threshold
            )]

        assert self.dataset
        trainer = ResumableTrainer(
            target_epochs=target_epochs,
            save_epochs=None,
            model=self.model,
            args=TrainingArguments(**training_args_kwargs),
            data_collator=None,  # Use the default_data_collator
            callbacks=callbacks,
            **self.dataset,
        )
        return trainer

    def fit(
        self,
        df: pd.DataFrame,
        in_df: Optional[pd.DataFrame] = None,
        join_on: Optional[str] = None,
        resume_from_checkpoint: Union[bool, str] = False,
        device="cuda",
        n_critic: Optional[int] = None,
        **kwargs
    ):
        if n_critic is None:
            n_critic = self.n_critic
        kwargs["n_critic"] = n_critic
        device = _validate_get_device(device)
        if self.model_type == ModelType.relational:
            assert (
                    in_df is not None
            ), "The REaLTabFormer for relational data requires two tables for training."
            assert join_on is not None, "The column to join the data must not be None."

            trainer = self._fit_relational(
                df, in_df, join_on=join_on, device=device, resume_from_checkpoint=resume_from_checkpoint
            )
            try:
                self.experiment_id = f"id{int((time.time() * 10 ** 10)):024}"
                torch.cuda.empty_cache()

                return trainer
            except Exception as exception:
                if device == torch.device("cuda"):
                    del self.model
                    torch.cuda.empty_cache()
                    self.model = None

                raise exception
        else:
            return super().fit(df, in_df, join_on, resume_from_checkpoint, device, **kwargs)

    @torch.no_grad()
    def sample(self, *args, **kwargs):
        # Faster sampling
        return super().sample(*args, **kwargs)


def fit_transform_rtf(data: np.ndarray, model_dir: str, prefix: str = None) -> pd.DataFrame:
    prefix = "" if prefix is None else f"{prefix}-"
    dim_prefix = prefix if prefix.endswith("-") else "dim"
    data = pd.DataFrame(data, columns=[f"{dim_prefix}{i:02d}" for i in range(data.shape[1])])
    raw_columns = data.columns.tolist()
    unique_threshold = 500 if prefix == "ctx" else 20
    def fast_nunique(col):
        return col.nunique()
    results = Parallel(n_jobs=20)(
        delayed(fast_nunique)(data[col]) for col in data.columns
    )
    uniques = pd.Series(results, index=data.columns)
    num_columns = data[:10].select_dtypes(include=[np.number]).columns.tolist()
    unary_columns = {
        c: data[c].iloc[0] for c, n in uniques.items() if n <= 1
    }
    data = data.drop(columns=[*unary_columns])
    num_columns = [
        c for c in num_columns if c not in unary_columns and
                                  (uniques[c] > unique_threshold or not (
                                          (data[c] - data[c].round()).abs().mean() < 1e-6 and
                                          data[c].max() - data[c].min() == uniques[c] - 1))
    ]
    cat_columns = [c for c in data.drop(columns=num_columns).columns]
    placeholder_columns = [] if data.shape[-1] > 0 else ["placeholder"]
    if (prefix == "ctx" and data.shape[-1] > 100) or (uniques[num_columns] > unique_threshold).any():
        bin_columns = [c for c in num_columns if uniques[c] > unique_threshold]
        bins = KBinsDiscretizer(n_bins=min(data.shape[0], 200), encode="ordinal", strategy="kmeans")
        num_columns = [c for c in num_columns if c not in bin_columns]
        data[bin_columns] = bins.fit_transform((data[bin_columns])).astype(np.int32)
        torch.save(bins, os.path.join(model_dir, f"{prefix}bins.pkl"))
    else:
        bin_columns = []
    info = {
        "num_columns": num_columns,
        "cat_columns": cat_columns,
        "bin_columns": bin_columns,
        "placeholders": placeholder_columns,
        "unary": unary_columns,
        "raw_columns": raw_columns
    }
    oe = OrdinalEncoder()
    data[cat_columns] = oe.fit_transform(data[cat_columns])
    data[cat_columns] = data[cat_columns].applymap(lambda x: f"C{int(x)}")
    data[num_columns] = data[num_columns].round(6)
    data[placeholder_columns] = 0
    with open(os.path.join(model_dir, f"{prefix}info.json"), "w") as f:
        json.dump(info, f, indent=2)
    torch.save(oe, os.path.join(model_dir, f"{prefix}oe.pkl"))
    return data


def transform_rtf(data: np.ndarray, model_dir: str, prefix: str = None) -> pd.DataFrame:
    prefix = "" if prefix is None else f"{prefix}-"
    with open(os.path.join(model_dir, f"{prefix}info.json"), "r") as f:
        info = json.load(f)
    oe = torch.load(os.path.join(model_dir, f"{prefix}oe.pkl"))
    dim_prefix = prefix if prefix.endswith("-") else "dim"
    data = pd.DataFrame(data, columns=[f"{dim_prefix}{i:02d}" for i in range(data.shape[1])])
    num_columns, cat_columns, bin_columns = info["num_columns"], info["cat_columns"], info["bin_columns"]
    data = data.drop(columns=[*info["unary"]])
    data[cat_columns] = oe.transform(data[cat_columns])
    data[cat_columns] = data[cat_columns].applymap(lambda x: f"C{int(x)}")
    data[num_columns] = data[num_columns].round(6)
    if bin_columns:
        bins = torch.load(os.path.join(model_dir, f"{prefix}bins.pkl"))
        data[bin_columns] = bins.transform(data[bin_columns]).astype(np.int32)
    data[info["placeholders"]] = 0
    return data


def inverse_transform_rtf(generated: pd.DataFrame, model_dir: str, prefix: str = None) -> np.ndarray:
    prefix = "" if prefix is None else f"{prefix}-"
    with open(os.path.join(model_dir, f"{prefix}info.json"), "r") as f:
        info = json.load(f)
    oe = torch.load(os.path.join(model_dir, f"{prefix}oe.pkl"))
    cat_columns = info["cat_columns"]
    if cat_columns:
        generated[cat_columns] = oe.inverse_transform(
            generated[cat_columns].applymap(lambda x: x[len("C"):]).astype(np.int32).values
        )
    bin_columns = info["bin_columns"]
    if bin_columns:
        bins = torch.load(os.path.join(model_dir, f"{prefix}bins.pkl"))
        generated[bin_columns] = bins.inverse_transform(generated[bin_columns].clip(
            0, upper={c: max(0, n) for c, n in zip(bin_columns, bins.n_bins_ - 1)}, axis=1).values
                                                        )
    generated = generated.drop(columns=info["placeholders"])
    for k, v in info["unary"].items():
        generated[k] = v
    generated = generated[info["raw_columns"]]
    return generated.values


def update_epochs(size, kwargs):
    if kwargs.get("max_steps", kwargs.get("num_train_epochs", kwargs.get("epochs"))) is None and size < 200:
        kwargs = kwargs.copy()
        kwargs["epochs"] = 500
    return kwargs


def fast_pearson_corr_gpu(df_num: pd.DataFrame) -> pd.DataFrame:
    data_gpu = cp.asarray(df_num.to_numpy(dtype=cp.float32))
    means = cp.mean(data_gpu, axis=0)
    stds = cp.std(data_gpu, axis=0)
    standardized = (data_gpu - means) / stds
    corr_gpu = cp.dot(standardized.T, standardized) / (data_gpu.shape[0] - 1)
    return pd.DataFrame(cp.asnumpy(cp.abs(corr_gpu)), index=df_num.columns, columns=df_num.columns)


def fast_correlation_ratio(
    categories: pd.Series,
    measurements: pd.DataFrame,
) -> pd.Series:
    columns = measurements.columns.tolist()
    measurements = measurements.values
    fcat, _ = pd.factorize(categories)  # type: ignore
    cat_num = np.max(fcat) + 1
    y_avg_array = np.zeros((cat_num, len(columns)))
    n_array = np.bincount(fcat)
    for i in range(0, cat_num):
        cat_measures = measurements[fcat == i]
        for j in range(measurements.shape[-1]):
            y_avg_array[i, j] = np.average(cat_measures[:, j])
    n_array = n_array.reshape((-1, 1))
    y_total_avg = np.sum(y_avg_array * n_array, axis=0) / np.sum(n_array)
    numerator = np.sum(n_array * (y_avg_array - y_total_avg) ** 2, axis=0)
    denominator = np.sum((measurements - y_total_avg) ** 2, axis=0)
    etas = np.sqrt(np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0))
    etas = np.clip(etas, 0.0, 1.0)
    result = pd.Series(etas, index=columns)
    return result


def wide_associations(
    df: pd.DataFrame,
) -> pd.DataFrame:
    # identifying categorical columns
    columns = df.columns
    nominal_columns = identify_nominal_columns(df)
    numerical_columns = df.drop(columns=nominal_columns).columns.tolist()

    # will be used to store associations values
    pbar = tqdm(total=len(columns) ** 2, desc="Compute corr")
    corr = pd.DataFrame(np.eye(len(columns)), index=columns, columns=columns, dtype=np.float64)
    pbar.update(len(columns))
    if len(numerical_columns) > 1:
        pearson_corr = fast_pearson_corr_gpu(df[numerical_columns])
        corr.loc[pearson_corr.index, pearson_corr.columns] = pearson_corr
        pbar.update(len(numerical_columns) ** 2 - len(numerical_columns))
        del pearson_corr
        gc.collect()

    cat_num_corr = []
    pbar = tqdm(
        desc=f"Cat-num correlations (#cat={len(nominal_columns)}, #num={len(numerical_columns)})",
        total=len(nominal_columns)
    )
    chunk_size = 20
    for i in range(0, len(nominal_columns), chunk_size):
        this_cat_num_corr = Parallel(n_jobs=chunk_size)(
            delayed(fast_correlation_ratio)(df[cat], df[numerical_columns])
            for cat in nominal_columns[i:i + chunk_size]
        )
        cat_num_corr.extend(this_cat_num_corr)
        pbar.update(len(nominal_columns[i:i + chunk_size]))
    for cat, corr_values in zip(nominal_columns, cat_num_corr):
        corr.loc[cat, corr_values.index] = corr_values.values
        corr.loc[corr_values.index, cat] = corr_values.values

    cat_pairs = itertools.combinations(nominal_columns, 2)
    cat_results = Parallel(n_jobs=20)(
        delayed(cramers_v)(df[col1], df[col2], False) for col1, col2 in tqdm(
            cat_pairs, desc=f"Cat-cat correlations ({len(nominal_columns)})",
            total=len(nominal_columns) * (len(nominal_columns) - 1) // 2
        )
    )
    for (col1, col2), val in zip(cat_pairs, cat_results):
        corr.loc[col1, col2] = corr.loc[col2, col1] = val
        pbar.update()

    return corr.fillna(0)


def sort_column_importance(df: pd.DataFrame):
    columns = []
    if df.shape[-1] < 500:
        corr = associations(
            df if df.shape[0] < 10_000 else df.sample(n=10_000),
            plot=False, max_cpu_cores=20, multiprocessing=True, compute_only=True
        )["corr"].abs()
    else:
        corr = wide_associations(
            df if df.shape[0] < 10_000 else df.sample(n=10_000),
        ).abs()
    nuniques = df.nunique()
    pbar = tqdm(total=df.shape[-1], desc="Sorting column importance")
    for i, c in enumerate(corr.columns):
        if c in columns:
            continue
        this_round = [c]
        for c2 in corr.index[i + 1:]:
            if c2 in columns:
                continue
            if corr.loc[c, c2] >= 0.95:
                this_round.append(c2)
        if len(this_round) > 1:
            this_nuniques = nuniques[this_round].sort_values(ascending=False)
            columns.extend(this_nuniques.index[1:].tolist())
            pbar.update(len(this_round) - 1)

    corr = corr.drop(columns=columns, index=columns)
    eye = np.eye(corr.shape[0], dtype=np.bool_)
    corr[eye] = -1
    while corr.shape[0] > 0:
        max_corr = corr.max()
        max_corr_value = max_corr.max()
        candidates = max_corr[max_corr == max_corr_value].index
        new_col = nuniques[candidates].idxmax()
        columns.append(new_col)
        corr = corr.drop(columns=[new_col], index=[new_col])
        pbar.update()

    # columns.extend(corr.drop(columns=columns, index=columns).max().sort_values(ascending=False).index.tolist())
    return [*reversed(columns)]


def save_to(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=5)


def load_from(path):
    with open(path, "rb") as f:
        return pickle.load(f)


@contextmanager
def log_resource_usage(path, descr):
    process = psutil.Process(os.getpid())
    peak_memory = [0]
    running = [True]

    def sample():
        while running[0]:
            mem = process.memory_info().rss
            if mem > peak_memory[0]:
                peak_memory[0] = mem
            time.sleep(0.5)

    thread = threading.Thread(target=sample)
    thread.start()
    start_time = time.time()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    try:
        logging.info(f"{descr} ...")
        yield
    finally:
        end_time = time.time()
        running[0] = False
        thread.join()
        peak_memory_mb = peak_memory[0] / (1024 ** 2)
        torch.cuda.synchronize()
        peak_gpu = torch.cuda.max_memory_allocated()

        peak_gpu_memory_mb = peak_gpu / (1024 ** 2)
        elapsed = end_time - start_time

        with open(path, 'a') as f:
            f.write(f"{descr},{peak_memory_mb},{peak_gpu_memory_mb},{elapsed}\n")


class CacheBlock(contextlib.AbstractContextManager):
    def __init__(self, description: str, cache_dir: str):
        self.description = description
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, f"{description}.pkl")

    def __enter__(self):
        if os.path.exists(self.cache_file):
            self._should_skip = True
            return None
        else:
            self._should_skip = False
            return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._should_skip or exc_type is not None:
            return False
        frame = inspect.currentframe().f_back
        local_vars = frame.f_locals.copy()

        os.makedirs(self.cache_dir, exist_ok=True)
        with open(self.cache_file, "wb") as f:
            dill.dump({
                "locals": local_vars,
                "random_state": random.getstate(),
                "np_random_state": np.random.get_state(),
                "torch_state": torch.get_rng_state(),
                "torch_cuda_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "torch_deterministic": torch.backends.cudnn.deterministic,
                "torch_benchmark": torch.backends.cudnn.benchmark,
            }, f)
        return False


def resume_from_last(cache_dir):
    if not os.path.exists(cache_dir):
        return {}

    files = sorted(
        (os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.endswith(".pkl")),
        key=os.path.getmtime
    )
    if not files:
        return {}

    last_file = files[-1]
    logging.info(f"Resuming from {last_file}")
    print(f"Resuming from {last_file}")
    with open(last_file, "rb") as f:
        data = dill.load(f)
    frame = inspect.currentframe().f_back
    frame.f_locals.update(data["locals"])
    random.setstate(data["random_state"])
    np.random.set_state(data["np_random_state"])
    torch.set_rng_state(data["torch_state"])
    if torch.cuda.is_available() and data["torch_cuda_state"] is not None:
        torch.cuda.set_rng_state_all(data["torch_cuda_state"])
    torch.backends.cudnn.deterministic = data["torch_deterministic"]
    torch.backends.cudnn.benchmark = data["torch_benchmark"]
    return data["locals"]
