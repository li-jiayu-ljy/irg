# Football Dataset

## Schema

| Table       | PK               | FK                                                                                                                      | TS              |
|-------------|------------------|-------------------------------------------------------------------------------------------------------------------------|-----------------|
| teams       | teamID           | -                                                                                                                       | -               |
| players     | playerID         | -                                                                                                                       | -               |
| gamestats   | gameID           | -                                                                                                                       | -               |
| games       | gameID           | (gameID, homeTeamID) -> teamstats.(gameID, teamID), (gameID, awayTeamID) -> teamstats.(gameID, teamID)                  | -               |
| appearances | gameID, playerID | gameID -> games.gameID, playerID -> players.playerID                                                                    | -               |
| teamstats   | teamID, gameID   | teamID -> teams.teamID, gameID -> gamestats.gameID                                                                      | sort by: date   |
| shots       | -                | (gameID, shooterID) -> appearances.(gameID, playerID), (gameID, assisterID) -> appearances.(gameID, playerID) [w/ NULL] | sort by: minute |

## Download and Prepare

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/technika148/football-database). Put the downloaded 
files under `data/` in this folder.

A problem of the raw dataset is that it does not 100% follow its relational schema, so we preprocess to clean the data.
Please run `preprocess.py` and find the preprocessed data under `preprocessed/`. Core changes other than the 
basic changes include:

1. Only 5 leagues are present, so the table is removed, and other tables involving league ID are treated as categorical.
2. No textual information would be used in our model, so names, etc. are removed. However, after removal of these 
   columns, some tables become an ID-only table. To avoid corner case bugs, we insert a placeholder unary value in the 
   tables.
3. Some (game + assister) in shots are not "appeared" in the same game, which is transformed into N/A.
4. Cyclic dependency between teamstats and games is decomposed into games -> teamstats -> gamestats.

The raw schema is not applicable to most baselines, so we apply feature engineering to make baselines work.
This result is found in `simplified/`. Core changes include:

1. Uniqueness of composite PKs are not tracked, but composite FKs on a composite PK is kept by inserting auxiliary 
   singular PK to replace the composite PK and use it for FK.
2. Some baselines do not accept NULL FKs, which we remove by feature engineering with a NULL parent record.

The schema for each baseline, along with IRG, is found in `schema/...yaml` or `schema/...json`.