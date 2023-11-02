# Databricks notebook source
# MAGIC %pip install piqdata

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

# COMMAND ----------

piq = PiqData(spark)

# COMMAND ----------

acds = ACDS(use_sample_mart=False)

# COMMAND ----------

#Enter attribution start and end dates here
attr_start = "2023-10-13"
attr_end = "2023-10-26"
#attr_end = "2023-10-14"

# COMMAND ----------

#This is core attribution date logic - may not need this but putting it here just in case

def get_minimum_exposure_transaction_date(start_date):
  return str((datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=-3)).date())

def get_maximum_exposure_transaction_date(end_date):
  return str((datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=3)).date())

def get_maximum_conversion_date(end_date):
  return str((datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=14)).date())

min_tran_date = get_minimum_exposure_transaction_date(camp_start)
max_tran_date = get_maximum_exposure_transaction_date(camp_end)
max_conv_date = get_maximum_conversion_date(camp_end)

# COMMAND ----------

#Use effodata to get ACDS txns for the attribution window
# `txns` will be a Spark DataFrame housing a transaction pull:
txns = acds.get_transactions(
           start_date=attr_start,              # start date for analysis
           end_date=attr_end,                # end date for analysis
           join_with=["products"],     # attribute tables to join with
           apply_golden_rules=golden_rules(),    # apply curated golden rules
       )

# COMMAND ----------

txns.display()

# COMMAND ----------

#Here are the columns from transaction + product dim that might be required to curate the dataset we will pass Koddi (note that we will transform and/or drop a few later)
  #ehhn - string - enterprise household number
  #con_upc_no - string - scan (GTIN) UPC
  #net_spend_amt - double - what was actually spent on the units purchased
  #cly_dol_am - double - price for all units that includes yellow tag and targeted offers
  #rtl_dol_am - double - retail price for all units (no discounts) 
  #scn_unt_qy - biginteger - number of units purchased
  #trn_dt - integer - transaction date in the form YYYYMMDD
  #trn_tm - string - transaction time in the form HH24:MI
cols = ['ehhn', 'con_upc_no', 'net_spend_amt', 'cly_dol_am', 'rtl_dol_am', 'scn_unt_qy', 'trn_dt', 'trn_tm']

# COMMAND ----------

txns_select = txns.select(cols)

# COMMAND ----------

txns_select.display()

# COMMAND ----------

#Write transaction pull data so I don't have to keep doing this expensive pull
path = "abfss://media@sa8451dbxadhocprd.dfs.core.windows.net/n510252/attribution_backfill/temp_out/10132023_10262023_txn_pull"

# COMMAND ----------

txns_select.write.parquet(path)

# COMMAND ----------

txns_select = spark.read.parquet(path)

# COMMAND ----------

#read in PIQ HH dimension. Don't be misled by the fact that this is in the PIQ data package--it's not actually a PIQ-dependent dimension and it will remain current going forward (for now)
hhs = piq.get_dimension('household').select(['ehhn', 'guid_hash'])

# COMMAND ----------

hhs.display()

# COMMAND ----------

#join on HH dimension to get to a hashed GUID
txns_select_guid_hashed = txns_select.join(hhs, on='ehhn')

# COMMAND ----------

txns_select_guid_hashed.display()

# COMMAND ----------

#lose some of the price columns that we don't need, and of course lose the ehhn
txns_final = txns_select_guid_hashed.drop('ehhn', 'net_spend_amt', 'rtl_dol_am')

# COMMAND ----------

txns_final.display()

# COMMAND ----------

#how many transactions were there in this period?
txns_select.count()

# COMMAND ----------

#how many rows do we get after HH-GUID join (probably some duplication, but should be <10% or so)
txns_final.count()

# COMMAND ----------

#how many unique hashed guids do we have in our final DF?
txns_final.select('guid_hash').distinct().count()

# COMMAND ----------

path = "abfss://media@sa8451dbxadhocprd.dfs.core.windows.net/n510252/attribution_backfill/out/10132023_10262023_txn_final_unfiltered"
txns_final.write.parquet(path)

# COMMAND ----------

txns_final_raw = spark.read.parquet( "abfss://media@sa8451dbxadhocprd.dfs.core.windows.net/n510252/attribution_backfill/out/10132023_10262023_txn_final_unfiltered")

# COMMAND ----------

#koddi provided us with the attribution UPCs that we need backfill for. We'll read those in and use them to filter an otherwise extremely long txn dataset
attr_upcs = spark.read.csv("abfss://media@sa8451dbxadhocprd.dfs.core.windows.net/n510252/attribution_backfill/input/2023-11-01-kroger-attributable-entities.csv", header=True)

# COMMAND ----------

#check to make sure no duplicates snuck in
attr_upcs.count()

# COMMAND ----------

attr_upcs.select('upc').distinct().count()

# COMMAND ----------

#looks like some duplicates snuck in. Drop them.
attr_upcs = attr_upcs.select('upc').distinct()

# COMMAND ----------

txns_final_raw.display()

# COMMAND ----------

attr_upcs.display()

# COMMAND ----------

#Pad the UPCs to 13 digits and make them strings

attr_upcs = attr_upcs.withColumn('attr_upc_padded', f.format_string("%013d", f.col('upc').cast('int')))

# COMMAND ----------

attr_upcs.display()

# COMMAND ----------

#Join attribution UPCs to transactions, so that we're left with only transactions on the attribution UPCs
attr_transactions = txns_final_raw.join(attr_upcs.select('attr_upc_padded'), txns_final_raw.con_upc_no == attr_upcs.attr_upc_padded)

# COMMAND ----------

attr_transactions.display()

# COMMAND ----------

#Do a count after the join to check data drop. It should be substantial, relative to unfiltered transactions 
attr_transactions.count()

# COMMAND ----------

#drop the redundant attr_upc_padded column and write the final file.
attr_transactions = attr_transactions.drop('attr_upc_padded')

# COMMAND ----------

attr_transactions.display()

# COMMAND ----------

path = "abfss://media@sa8451dbxadhocprd.dfs.core.windows.net/n510252/attribution_backfill/out/10132023_10262023_txn_final_filtered"
attr_transactions.write.parquet(path)

# COMMAND ----------

#read in the output
attr_transactions = spark.read.parquet('abfss://media@sa8451dbxadhocprd.dfs.core.windows.net/n510252/attribution_backfill/out/10132023_10262023_txn_final_filtered')

# COMMAND ----------

attr_transactions.count()

# COMMAND ----------

#write to a pipeline delimited csv
#attr_transactions.write.option("sep","|").option("header","true").csv('abfss://media@sa8451dbxadhocprd.dfs.core.windows.net/n510252/attribution_backfill/out/10132023_10262023_txn_final_filtered_delim.csv')
attr_transactions.write.format('csv') \
    .options(delimiter='|') \
    .save('abfss://media@sa8451dbxadhocprd.dfs.core.windows.net/n510252/attribution_backfill/out/10132023_10262023_txn_final_filtered_delim.csv')
