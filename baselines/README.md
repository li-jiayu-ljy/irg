# Baseline Models 

## Baseline Models Summary

| Model           | Implementation                                       |
|-----------------|------------------------------------------------------|
| SDV-IND (`ind`) | SDV SDK modified                                     |
| ClavaDDPM       | [GitHub](https://github.com/weipang142857/ClavaDDPM) |

## Preprocessing and Baseline Experiments

To align the capability of all baseline models without being limited by the trivial and niche data processing
capabilities (e.g., missing data, date time), we preprocess all datasets such that:
1. Datetime values are converted to numeric values by difference to its minimum value in seconds.
2. Except for FK columns, all missing values are filled 
   - Numeric values are filled by a special value smaller than the global minimum and an additional NULL indicator 
     Boolean column is inserted
   - Categorical values are filled by a special category "<special-for-null>" (so this should not be an existing 
     category in the real data)

Nevertheless, we still regard the capability to handle standard categorical and numeric values as requirement of any 
model. Relevant data preprocessing and normalization will be done by the models.
All evaluation will be applied on this processed results too.
Metadata of the processing is saved after execution of `preprocess.py` of a dataset.

Specific processing required for each baseline is shown under the corresponding directory.
