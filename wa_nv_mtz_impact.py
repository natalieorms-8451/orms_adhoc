# Databricks notebook source
# MAGIC %pip install piqdata

# COMMAND ----------

spark.conf.set("spark.sql.adaptive.enabled",'true')
spark.conf.set("spark.sql.adaptive.skewJoin.enabled",'true')
spark.conf.set("spark.databricks.queryWatchdog.outputRatioThreshold", "1500000000")

# COMMAND ----------

import pandas as pd
from datetime import datetime, date, timedelta
from piqdata import PiqData
from pyspark.sql import DataFrame, Column, Row
import pyspark.sql.functions as f
import pyspark.sql.types as t
from pyspark.sql.window import Window
from typing import Optional
from effodata import ACDS, golden_rules
import datetime as dt
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
from dateutil.relativedelta import relativedelta, FR

# COMMAND ----------

piq = PiqData(spark)

# COMMAND ----------

acds = ACDS(use_sample_mart=False)

# COMMAND ----------

#Enter analysis start and end dates here
analysis_start = "2022-10-01"
analysis_end = "2023-09-30"

# COMMAND ----------

hshd_dim = spark.read.parquet("abfss://landingzone@sa8451entlakegrnprd.dfs.core.windows.net/mart/comms/prd/dim/digital_customer_dim/")
store_dim = acds.stores

# COMMAND ----------

store_dim_select = store_dim.select('mgt_div_no', 'sto_no', 'sto_sta_cd') \
  .withColumn('DIV_STORE', f.concat(f.col('mgt_div_no'), f.col('sto_no'))) \
    .filter(f.col('sto_sta_cd').isin(['WA', 'NV']))

# COMMAND ----------

hshd_dim_store = hshd_dim.join(store_dim_select, on='DIV_STORE')

# COMMAND ----------

hshd_dim_store_wanv = hshd_dim_store.filter(f.col('sto_sta_cd').isin(['WA', 'NV']))

# COMMAND ----------

clicks = piq.get_log(start_date = analysis_start,
                    end_date = analysis_end,
                    log_type = 'CLICK') 
clicks = piq.join_with_dimension(clicks, 'household')

# COMMAND ----------

impressions = piq.get_log(start_date = analysis_start,
                    end_date = analysis_end,
                    log_type = 'IMPRESSION') 
impressions = piq.join_with_dimension(impressions, 'household')

# COMMAND ----------

#filter clicks for PLAs (unit_type = 'CPC') and filter impressions for TOAs (unit_type = 'CPM')
clicks_pla = clicks.filter(f.col('unit_type') == 'CPC')
imps_toa = impressions.filter(f.col('unit_type') == 'CPM')

# COMMAND ----------

clicks_pla_wanv = clicks_pla.join(store_dim_select, clicks_pla.div_store == store_dim_select.DIV_STORE)
imps_toa_wanv = imps_toa.join(store_dim_select, imps_toa.div_store == store_dim_select.DIV_STORE)

# COMMAND ----------

clicks_pla_wanv.count()

# COMMAND ----------

imps_toa_wanv.count()

# COMMAND ----------

clicks_pla_wanv = clicks_pla_wanv.withColumn('clearing_price_dollars', f.col('clearing_price')/1000000)
imps_toa_wanv = imps_toa_wanv.withColumn('clearing_price_dollars', f.col('clearing_price')/1000000)

# COMMAND ----------

pla_aggs = clicks_pla_wanv.select('sto_sta_cd', 'clearing_price_dollars').groupby('sto_sta_cd') \
  .agg(f.sum(f.col('clearing_price_dollars')).alias('Total PLA Spend ($)'))

# COMMAND ----------

toa_aggs = imps_toa_wanv.select('sto_sta_cd', 'clearing_price_dollars').groupby('sto_sta_cd') \
  .agg(f.sum(f.col('clearing_price_dollars')).alias('Total TOA Spend ($)'))

# COMMAND ----------

total_spend_aggs = pla_aggs.join(toa_aggs, on='sto_sta_cd') \
  .withColumn('Total Spend ($)', f.col('Total PLA Spend ($)') + f.col('Total TOA Spend ($)'))

# COMMAND ----------

total_spend_aggs.display()

# COMMAND ----------

hshd_dim_store_wanv.select('sto_sta_cd', 'GUID').groupby('sto_sta_cd').count().display()

# COMMAND ----------

hshd_dim_store_wanv.select('sto_sta_cd', 'EHHN').distinct().groupby('sto_sta_cd').count().display()
