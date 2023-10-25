# Databricks notebook source
# MAGIC %pip install --upgrade snowflake-connector-python

# COMMAND ----------

import pandas as pd
from datetime import datetime, date, timedelta
from pyspark.sql import DataFrame, Column, Row
import pyspark.sql.functions as f
import pyspark.sql.types as t
from pyspark.sql.window import Window
from typing import Optional
#from effodata_clickstream.effodata_clickstream import CSDM
import matplotlib.pyplot as plt

# COMMAND ----------

SNOWFLAKE_SOURCE_NAME = "net.snowflake.spark.snowflake"

# COMMAND ----------

# MAGIC %md
# MAGIC Koddi Log Data

# COMMAND ----------

#opps_df = spark.read.parquet("abfss://media@sa8451camdev.dfs.core.windows.net/certified/streaming/lld/koddi/event_type=opportunities/")
opps_df = spark.read.parquet("abfss://media@sa8451camprd.dfs.core.windows.net/certified/streaming/lld/koddi/event_type=opportunities/")
opps_df.display()

# COMMAND ----------

opps_df.count()

# COMMAND ----------

opps_df.select('ad_request_id').distinct().count()

# COMMAND ----------

opps_df.where(f.col('targeting_placement_id')=='01100').display()

# COMMAND ----------

opps_df.select('target').distinct().collect()

# COMMAND ----------

opps_df.select('targeting_placement_id').distinct().collect()

# COMMAND ----------

opps_df.groupBy(['targeting_placement_id', 'targeting_slot_id']).count().display()

# COMMAND ----------

opps_df = spark.read.parquet("abfss://media@sa8451camprd.dfs.core.windows.net/certified/streaming/lld/koddi/event_type=opportunities/")
opps_df.groupBy('target').count().display()

# COMMAND ----------

opps_df = spark.read.format("delta").load('abfss://media@sa8451camprd.dfs.core.windows.net/certified/streaming/lld_dlt/koddi/tables/opportunities')
opps_df.display()

# COMMAND ----------

opps_df.count()

# COMMAND ----------

opps_df.select('ad_request_id').distinct().count()

# COMMAND ----------

opps_df.groupBy(['targeting_placement_id', 'targeting_slot_id']).count().display()

# COMMAND ----------

opps_df.groupBy('target').count().display()

# COMMAND ----------

opps_df_search = opps_df.where(f.col("target")=='SEARCH')

# COMMAND ----------

search_date='2023-10-17'
opps_df_search_date = opps_df_search.where(f.col('dt')==search_date)

# COMMAND ----------

opps_df_search_date.count()

# COMMAND ----------

opps_df_search_date.select(f.sum(f.col('slots_available'))).show()


# COMMAND ----------

opps_df_search.groupBy(['dt', 'slots_available']).count().display()

# COMMAND ----------

imps_df = spark.read.format("delta").load('abfss://media@sa8451camprd.dfs.core.windows.net/certified/streaming/lld_dlt/koddi/tables/impressions')

# COMMAND ----------

imps_df.display()

# COMMAND ----------

imps_search = imps_df.where(f.col('targeting_slot_id').isin(['1525', '1540']))

# COMMAND ----------

imps_search_date = imps_search.where(f.col('dt')=='2023-10-24')

# COMMAND ----------

imps_search_date.count()

# COMMAND ----------

imps_search_toas_plas = imps_df.where(f.col('targeting_slot_id').isin(['1525', '1540', '11025', '11035']))
imps_search_tp_date = imps_search_toas_plas.where(f.col('dt')=='2023-10-17')
imps_search_tp_date.count()

# COMMAND ----------

imps_search_date = imps_search_date.withColumn('rank_diff', f.col('onsite_organic_rank')-f.col('onsite_paid_rank'))

# COMMAND ----------

imps_search_date.groupBy('rank_diff').count().sort('rank_diff').display()

# COMMAND ----------


