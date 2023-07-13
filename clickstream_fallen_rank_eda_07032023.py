# Databricks notebook source
# MAGIC %pip install --upgrade effodata_clickstream

# COMMAND ----------

import pandas as pd
from datetime import datetime, date, timedelta
#from piqdata import PiqData
from pyspark.sql import DataFrame, Column, Row
import pyspark.sql.functions as f
import pyspark.sql.types as t
from pyspark.sql.window import Window
from typing import Optional
from effodata_clickstream.effodata_clickstream import CSDM
import matplotlib.pyplot as plt

# COMMAND ----------

csdm = CSDM(spark)

# COMMAND ----------

#get the clickstream product and fact tables for the time period of interest
clicks_prod_interest = csdm.get_clickstream_records(start_date='2023-06-14', end_date='2023-06-16', join_with=['product'])

# COMMAND ----------

clicks_prod_interest.display()

# COMMAND ----------

#pull columns needed for the analysis and limit to 'internal-search' scenarios
search_clicks = clicks_prod_interest.where(f.col('effo_scenario_name')=='internal-search').select('click_date', 'effo_click_id', 'espot_id', 'guid', 'impression_id', 'product_item_index', 'monetized_product', 'relevance_score', 'upc', 'product_impression_id')

# COMMAND ----------

search_clicks.display()

# COMMAND ----------

#how many records are we left with?

search_clicks.count()

# COMMAND ----------

#how many unique effo_click_id do we have?

effo_click_ids = search_clicks.select('effo_click_id').distinct().collect()
len(effo_click_ids)

# COMMAND ----------

# Some "effo_click_id" have null "relevance_score" values for monetized slots, which makes them inelligible for this analysis unless we can recover the score from some other data source. I'd love to know why this happens occassionally. In the meantime, I'm going to identify click IDs where this is an issue and drop those records in their entirety using an anti-join

null_score_search_clicks = search_clicks.where(f.col('relevance_score').isNull())
search_clicks_real_scores = search_clicks.join(null_score_search_clicks, on='effo_click_id', how='left_anti')

# COMMAND ----------

#how many records are left after the anti-join?

search_clicks_real_scores.count()

# COMMAND ----------

#how many distinct effo_click_ids remain for analysis?

search_clicks_real_scores.select('effo_click_id').distinct().count()

# COMMAND ----------

#gut check that the (number of unique effo_click_ids remaining) = (number we started with) - (number we dropped)

null_score_search_clicks.select('effo_click_id').distinct().count()

# COMMAND ----------

#add a column that ranks the products in every slot for every effo_click_id by their search relevancy scores. We will do this using a window over effo_click_id and the rank() function

effo_click_window  = Window.partitionBy("effo_click_id").orderBy("relevance_score")
search_clicks_relevance_rank = search_clicks_real_scores.withColumn("relevance_rank",f.rank().over(effo_click_window)) 

# COMMAND ----------

#have a look at the resulting df

search_clicks_relevance_rank.display()

# COMMAND ----------

#we're now ready to compare where the product appeared on the search results page (product_item_index) with where it would have appeared organically (relevance_rank). We are only concerned with monetized products for the time being, because we want to know how frequently and to what degree advertisers are paying for a monetized spot that is lower than where the product would have appeared organically.

#first, we filter for monetized products and add a new column: rank_diff = relevance_rank - product_item_index. Negative values of rank_diff imply downgrade

search_clicks_relevance_rank = search_clicks_relevance_rank.withColumn('rank_diff', f.col('relevance_rank')-f.col('product_item_index'))
search_clicks_monetized = search_clicks_relevance_rank.where(f.col('monetized_product')==True)

# COMMAND ----------

#lets take a look at the resulting df and make sure we did what we intended to do 

search_clicks_monetized.display()

# COMMAND ----------

#how many products (rows) do we have for analysis?

search_clicks_monetized.count()

# COMMAND ----------

search_clicks_monetized.select('effo_click_id').distinct().count()

# COMMAND ----------

#here is the first result we are interested in: frequency of the different rank_diff to show the scope of the problem

search_clicks_monetized.groupBy('rank_diff').count().display()
