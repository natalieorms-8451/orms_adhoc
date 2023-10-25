# Databricks notebook source
#default 3.0.0 is broken FYI
! pip install --upgrade googletrans==4.0.0rc1

# COMMAND ----------

! pip install effodata_clickstream

# COMMAND ----------

import pandas as pd
from datetime import datetime, date, timedelta
from pyspark.sql import DataFrame, Column, Row
import pyspark.sql.functions as f
import pyspark.sql.types as t
from pyspark.sql.window import Window
from typing import Optional
from effodata_clickstream.effodata_clickstream import CSDM
import matplotlib.pyplot as plt
from googletrans import Translator

# COMMAND ----------

csdm = CSDM(spark)

# COMMAND ----------

translator = Translator()

# COMMAND ----------

start_date_analysis = '2023-09-29'
end_date_analysis = '2023-09-29'

# COMMAND ----------

#get the clickstream product and fact tables for the time period of interest
clicks = csdm.get_clickstream_records(start_date=start_date_analysis, end_date=end_date_analysis, join_with=['product','session'])

# COMMAND ----------

#limit clickstream logs to internal-search scenarios
search_clicks = clicks.where(f.col('effo_scenario_name')=='internal-search')

# COMMAND ----------

# A function that leverages Google Translate's detect functionality to output the detected human language (a shorthand string) given an input string. The output also has a .confidence attribute that can be collected, if desired

def translate_query(text_to_trans):
  try:
    detected = translator.detect(text_to_trans)
    language = detected.lang
    #confidence = detected.confidence
  except:
    language = 'error'
  return language

# COMMAND ----------

search_clicks.count()

# COMMAND ----------

#clickstream provides UPC level granularity, so there will be lots of duplicate rows for the three columns of interest
search_clicks_unique = search_clicks.select('effo_click_id', 'text_typed', 'banner').dropDuplicates()

# COMMAND ----------

search_clicks_unique.count()

# COMMAND ----------

search_clicks_pd = search_clicks_unique.toPandas()

# COMMAND ----------

#there are some null query strings. Drop them for now. Figure out why later.
search_clicks_pd = search_clicks_pd.dropna(subset=['text_typed'])

# COMMAND ----------

search_clicks_pd_sample = search_clicks_pd.sample(frac=0.5, replace=True, random_state=1)

# COMMAND ----------

search_clicks_pd['search_language'] = search_clicks_pd['text_typed'].apply(translate_query)

# COMMAND ----------

translated_searches_freq = search_clicks_pd.groupby(['search_language'])['search_language'].count()

# COMMAND ----------

translated_searches_freq

# COMMAND ----------

translated_search_terms_freq = search_clicks_pd.groupby(['search_language','text_typed'])['text_typed'].count()

# COMMAND ----------

translated_search_terms_freq
