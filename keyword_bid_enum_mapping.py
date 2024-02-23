# Databricks notebook source
# MAGIC %pip install effo_embeddings

# COMMAND ----------

import pandas as pd
from datetime import datetime, date, timedelta
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
from effo_embeddings.core import Embedding
import random
import pickle

# COMMAND ----------

# MAGIC %md
# MAGIC ### Set Up

# COMMAND ----------

random.seed(989)
e=Embedding('query2concept2vec_1.0')

# COMMAND ----------

### DATES FOR PULLING KODDI LOGS ###

num_weeks = 8
analysis_window = dt.timedelta(days=num_weeks*7) #this defines the time window used for pulling logs

analysis_end_raw = datetime.now() + relativedelta(weekday=FR(-1)) #here we grab the most recent Friday and use that as the starting point for our date logic
analysis_start_raw = analysis_end_raw - analysis_window

analysis_end = analysis_end_raw.strftime('%Y-%m-%d')
analysis_start = analysis_start_raw.strftime('%Y-%m-%d')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get top search strings from Koddi opportunitiy log. Clean those strings a bit, and do an aggregation to determine the top query strings by frequency (frequency = number of ad requests over the analysis period)

# COMMAND ----------

#Read in the Koddi opportunity log

dlt_opportunities_path = "abfss://eda@sa8451mapinbprd.dfs.core.windows.net/streaming/lld/tables/opportunities"

opportunities_df=spark.read.format("delta").load(dlt_opportunities_path)

opportunities_df = opportunities_df.filter(opportunities_df.dt.between(analysis_start, analysis_end))

# COMMAND ----------

#filter for Search PLA web and app slots (these are the PLA slots that will have non-null "targeting_keyword" vals)
opportunities_df_query = opportunities_df.filter((opportunities_df.targeting_slot_id.isin(['1540', '1525'])))

# COMMAND ----------

#There are lots of search strings that are long alphanumeric strings corresponding to coupon codes. These aren't useful (we assume) from a keyword bidding perspective, so we're going to use some fuzzy regex to filter out strings that are 15+ alphanumeric with no white space. We won't catch everything, but the things we will catch will largely be misspellings (e.g., "spaghettigarlic" vs "spaghetti garlic" ) or these coupon codes (e.g., "buy5save1shopall2142").

opportunities_df_query_mess = opportunities_df_query.filter(f.col("targeting_keyword").rlike("^[a-zA-Z0-9]{15,}$"))

# COMMAND ----------

opportunities_df_query_mess.select('targeting_keyword').distinct().display()

# COMMAND ----------

#Do a left anti-join to get rid of opportunity rows corresponding to the messy coupon query strings

opportunities_df_query_clean = opportunities_df_query.join(opportunities_df_query_mess, on='targeting_keyword', how="left_anti")
opportunities_df_query_clean = opportunities_df_query_clean.withColumn('targeting_keyword_lower', f.lower(f.col('targeting_keyword')))

# COMMAND ----------

#Count the number of times query strings appear in the opportunity log

opportunities_per_query = opportunities_df_query_clean.select('targeting_keyword_lower').groupBy('targeting_keyword_lower').count()
opportunities_per_query = opportunities_per_query.withColumnRenamed('targeting_keyword_lower', 'targeting_keyword')

# COMMAND ----------

#Sort descending, so the most frequent strings are at the top

opportunities_per_query = opportunities_per_query.sort(f.desc("count"))

# COMMAND ----------

opportunities_per_query.dropna().display()

# COMMAND ----------

#Store the query strings and their assoiated counts for later reference

opportunities_per_query.write.parquet("abfss://media@sa8451dbxadhocprd.dfs.core.windows.net/keyword_bidding_analysis/data/targeting_keyword_frequency_cleaned_sorted_{start}_{end}".format(start = analysis_start, end = analysis_end), mode='overwrite')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use the top ranking (by frequency) search strings to get linguistically similar variants from embeddings 

# COMMAND ----------

opportunities_per_query = spark.read.parquet('abfss://media@sa8451dbxadhocprd.dfs.core.windows.net/keyword_bidding_analysis/data/targeting_keyword_frequency_cleaned_sorted_{start}_{end}'.format(start = analysis_start, end = analysis_end)).sort(f.desc("count"))

# COMMAND ----------

#Get a list of all the most popular "clean" keywords (we define "popular" and "clean" to mean the top ~100K search strings that together account for ~90% of PLA search ad request traffic). These are the strings we will input into embeddings to get conceptually similar match strings. They will also become keys in our mapping dict that allows us to go from popular, cleaner keyword strings to messier, less popular similar strings (e.g., "pineapple": ["pineapples", "pinapple", "puneapple"])

targeting_keywords_list = opportunities_per_query.select('targeting_keyword').rdd.flatMap(lambda x: x).collect()[:100000]

# COMMAND ----------

#Initialize a dict with keys (clean keywords) but no vals (i.e., where the embeddings output lists will go)

keyword_dict = dict.fromkeys(targeting_keywords_list)

# COMMAND ----------

#Function that calls the embeddings module. Increasing limit will increase the number (and scope) of similar match strings returned. 5-10 is optimal.

def get_relevant_keywords_df(input_keyword, limit=5):

  s,_=e.get_similar_by_key([input_keyword ],limit)
  output_temp= [str(i) for i in s[0] if not i.isdigit()]

  return output_temp

# COMMAND ----------

#This will take an estimated 7 hours to run. Here we are initializing an empty set to hold all cumulative output from embeddings. For every input (i.e., "clean") keyword string, we check to see if it's in the output set already. If it is, we skip it, as it already belongs to a previous, more popular keyword's "concept space." If it is not present in the output set, we retieve embeddings matches. We then update our keyword dict with those matches, and we add the matches to the output set too.

output_set = set()
for i in targeting_keywords_list:
  if i not in output_set:
    output_for_i = get_relevant_keywords_df(i)
    output_set.update(output_for_i)
    keyword_dict.update({ i: output_for_i })
  else:
    continue

# COMMAND ----------

keyword_dict

# COMMAND ----------

pickle.dump(keyword_dict, open("/dbfs/n510252/keyword_dict_raw_{start}_{end}.p".format(start = analysis_start, end = analysis_end), "wb"))

# COMMAND ----------

#create a filtered version of the keyword dict that drops keys where value is None (i.e., keys that were intentionally skipped or had no output from embeddings)

keyword_dict_filtered = {k: v for k, v in keyword_dict.items() if v is not None and len(v)>0}

# COMMAND ----------

pickle.dump(keyword_dict_filtered, open("/dbfs/n510252/keyword_dict_filtered_{start}_{end}.p".format(start = analysis_start, end = analysis_end), "wb"))

# COMMAND ----------

len(keyword_dict)

# COMMAND ----------

len(keyword_dict_filtered)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Convert the dictionaries to dataframes for wrangling

# COMMAND ----------

#with open('/dbfs/n510252/keyword_dict_raw_{start}_{end}.p'.format(start = analysis_start, end = analysis_end), 'rb') as handle:
#    keyword_dict = pickle.load(handle)

# COMMAND ----------

#with open('/dbfs/n510252/keyword_dict_filtered_{start}_{end}.p'.format(start = analysis_start, end = analysis_end), 'rb') as handle:
#    keyword_dict_filtered = pickle.load(handle)

# COMMAND ----------

#from pyspark.sql.types import StructType, StructField, StringType, ArrayType

# Define the schema for the DataFrame
#schema = StructType([
#    StructField("keyword", StringType(), True),
#    StructField("matches", ArrayType(StringType(), True), True)
#])

#keyword_dict_df = spark.createDataFrame(keyword_dict.items(), schema)
#keyword_dict_filtered_df = spark.createDataFrame(keyword_dict_filtered.items(), schema)

# COMMAND ----------

#keyword_dict_filtered = keyword_dict_filtered_df.where(f.size("matches") > 0)

# COMMAND ----------

#keyword_dict_filtered_explode = keyword_dict_filtered.withColumn('matches_exploded', f.explode(keyword_dict_filtered.matches))
