# Brizilian E-Commerce Dataset

## Schema

| Table          | PK                           | FK                                                                                             | TS                               |
|----------------|------------------------------|------------------------------------------------------------------------------------------------|----------------------------------|
| geolocation    | geolocation_zip_code_prefix  | -                                                                                              | -                                |
| products       | product_id                   | -                                                                                              | -                                |
| customers      | customer_id                  | customer_zip_code_prefix -> geolocation.geolocation_zip_code_prefix                            | -                                |
| sellers        | seller_id                    | seller_zip_code_prefix -> geolocation.geolocation_zip_code_prefix                              | -                                |
| orders         | order_id                     | customer_id -> customers.customer_id                                                           | -                                |
| order_items    | order_id, order_item_id      | order_id -> orders.order_id, product_id -> products.product_id, seller_id -> sellers.seller_id | sort by: order_item_id           |
| order_payments | order_id, payment_sequential | order_id -> orders.order_id                                                                    | sort by: payment_sequential      |
| order_reviews  | review_id                    | order_id -> orders.order_id                                                                    | sort by: review_answer_timestamp |

## Download and Prepare

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). Put the downloaded 
files under `data/` in this folder.

A problem of the raw dataset is that it does not 100% follow its relational schema, so we preprocess to clean the data.
Please run `preprocess.py` and find the preprocessed data under `preprocessed/`. Core changes other than the basic 
changes include:

1. Product category names table is ignored because it can be treated as categorical and the information is mainly 
   textual, which is not a focus of this project.
2. There are many different city names in geological information, which is textual and relies on general world knowledge
   to extract its information. Cities can be treated as categorical but the number of categories is way larger than 
   typical categorical columns. For simplicity, we drop cities.
3. Some zip code prefix in child tables of `geolocation` are not provided in `geolocation` table. 
   These rows are removed.
4. Geolocation zip code prefix is not unique, but it is used as a foreign key, which is essentially invalid. We group 
   the same zip code prefix and use the mean `geolocation_lat` and `geolocation_lng`, and use the first 
   `geolocation_state`.
5. The reviews table contains much textual information, which would be dropped.
6. Neither `review_id` nor `order_id` is unique for `order_reviews`, while both are hashed ID values. Therefore, it is 
   hard to infer the initial intention of these two IDs. For simplicity, we maintain only the first of reviews with the
   same `review_id`.

The content of `simplified` is the same as the content of `preprocessed`. Composite primary keys by a foreign key with a 
local ID will be ignored as if the table has no primary key for baseline models.