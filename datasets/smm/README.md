# Super Mario Maker Dataset

## Schema

| Table        | PK         | FK                                                                  | TS             |
|--------------|------------|---------------------------------------------------------------------|----------------|
| players      | id         | -                                                                   | -              |
| courses      | id         | -                                                                   | -              |
| course_maker | id         | id -> courses.id, maker -> players.id [w/ NULL]                     | -              |
| plays        | id, player | id -> courses.id, player -> players.id                              | sort by: catch |
| clears       | id, player | (id, player) -> plays.(id, player)                                  |                |
| likes        | id, player | (id, player) -> plays.(id, player)                                  |                |
| records      | id, player | (id, player) -> plays.(id, player)                                  |                |
| course-meta  | -          | id -> courses.id, (id, firstClear) -> clears.(id, player) [w/ NULL] | sort by: catch |

## Download and Prepare

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/leomauro/smmnet/data). Put the downloaded files under
`data/` in this folder.

A problem of the raw dataset is that it does not 100% follow its relational schema, inferred from its semantic meaning,
so we preprocess to clean the data. 
Please run `preprocess.py` and find the preprocessed data under `preprocessed/`. Core changes other than the 
basic changes include:

1. Some columns in this dataset are links or names, which will be dropped for our experiments. 
2. Original `courses` table has the only FK NULL-able. Although theoretically by IRG we can simply handle this by FK 
   matching without additional constraints and treat the entire table with FK encoded as a standalone table, we 
   decompose this table into two to make two tables where the second table contains two FKs, first with a non-NULL-able 
   FK and a NULL-able FK, to avoid additional code implementation.
3. All `clears`, `likes`, and `records` should be `plays`, which is not satisfied. We remove the violations.
4. All non-NULL (id, firstClear) pairs in `coruse-meta` should be found in `clears`, which is not satisfied. 
   We remove the violations.

The raw schema is not applicable to most baselines due to the existence of composite PK by two FKs. Fortunately all 
these tables do not have children, so we do not pass the composite PK information to baselines.
FKs with NULL values will also be removed by feature engineering for some baselines. But this change is specific to
the baseline model, so content in `simplified/` is the same as `preprocessed/`.

The schema for each baseline, along with IRG, is found in `schema/...yaml` or `schema/...json`.
