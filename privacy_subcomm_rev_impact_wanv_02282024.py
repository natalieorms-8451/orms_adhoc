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

#Enter analysis start and end dates here
analysis_start = "2022-10-01"
analysis_end = "2023-09-30"

# COMMAND ----------

#This is the sensitive subcom table provided by the business. I manually groomed it a bit, then uploaded into ADLS as a csv
sensitive_subcoms = spark.read.csv("abfss://media@sa8451dbxadhocprd.dfs.core.windows.net/n510252/privacy_upcs/special_sub_com_list_wa.csv", header=True)

# COMMAND ----------

sensitive_subcoms.limit(5).display()

# COMMAND ----------

sensitive_subcom_cds = [str(row.SUBC) for row in sensitive_subcoms.select('SUBC').collect()]

# COMMAND ----------

sensitive_subcom_cds

# COMMAND ----------

# MAGIC %md
# MAGIC ### Store Dim for WA/NV specific filtering

# COMMAND ----------

store_dim = acds.stores

# COMMAND ----------

store_dim_select = store_dim.select('mgt_div_no', 'sto_no', 'sto_sta_cd') \
  .withColumn('DIV_STORE', f.concat(f.col('mgt_div_no'), f.col('sto_no'))) \
    .filter(f.col('sto_sta_cd').isin(['WA', 'NV']))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get ACDS products/txns and filter for the sub-commodities included in analysis

# COMMAND ----------

#Use effodata to get ACDS txns for the analysis window
txns = acds.get_transactions(
           start_date=analysis_start,              # start date for analysis
           end_date=analysis_end,                # end date for analysis
           join_with=["products"],     # attribute tables to join with
           apply_golden_rules=golden_rules(),    # apply curated golden rules
       )

# COMMAND ----------

acds.products.select('con_upc_no', 'pid_fyt_sub_com_cd', 'pid_fyt_sub_com_dsc_tx')

# COMMAND ----------

#Here are the columns from ACDS that will be required for PIQ monetization revenue analysis
  #con_upc_no - string - scan (GTIN) UPC
  #pid_fyt_sub_com_cd - string - PID family tree sub commodity code
  #pid_fyt_sub_com_dsc_tx - string - PID family tree sub commodity description
taxonomy_cols = ['con_upc_no', 'pid_fyt_sub_com_cd', 'pid_fyt_sub_com_dsc_tx']

# COMMAND ----------

only_tax_cols = txns.select(taxonomy_cols).dropDuplicates()
privacy_upcs = only_tax_cols.where(f.col('pid_fyt_sub_com_cd').isin(sensitive_subcom_cds))

# COMMAND ----------

privacy_upcs.display()

# COMMAND ----------

#Write transaction pull data so I don't have to keep doing this expensive pull
privacy_upcs_path = "abfss://media@sa8451dbxadhocprd.dfs.core.windows.net/n510252/privacy_upcs/10012022_09302023_upcs"

# COMMAND ----------

privacy_upcs.write.parquet(privacy_upcs_path)

# COMMAND ----------

privacy_upcs = spark.read.parquet(privacy_upcs_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get PIQ Clicks (PLA rev) & Impressions (TOA rev)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Click log pull and join to privacy UPCs

# COMMAND ----------

#read in clicks with the product dimension, so we have UPCs too
clicks = piq.get_log(start_date = analysis_start,
                    end_date = analysis_end,
                    log_type = 'CLICK') 

# COMMAND ----------

clicks_products = piq.join_with_dimension(clicks, ['content', 'slot', 'household'])

# COMMAND ----------

#inner join on the upcs relevant to the analysis
clicks_select = privacy_upcs.join(clicks_products, (privacy_upcs.con_upc_no == clicks_products.sku))

# COMMAND ----------

#inner join on the WA & NV stores relative to the analysis
clicks_select = clicks_select.join(store_dim_select, clicks_select.div_store == store_dim_select.DIV_STORE)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Impression log pull and join to target values dimension

# COMMAND ----------

#read in impressions with the slot dimension, because TOAs are billed by CPM
impressions = piq.get_log(start_date = analysis_start,
                    end_date = analysis_end,
                    log_type = 'IMPRESSION')

# COMMAND ----------

impressions_products = piq.join_with_dimension(impressions, ['slot', 'household'])

# COMMAND ----------

#I have to pull in the target values dimension separately, per package guidelines, and do the join manually after exploding the matching_target_values array column in the impression log
impressions_products_explode = impressions_products.withColumn('matching_target_values_exp', f.explode(impressions_products.matching_target_values))

# COMMAND ----------

target_vals = piq.get_dimension('target_value')

# COMMAND ----------

impressions_with_target_vals = impressions_products_explode.join(target_vals, impressions_products_explode.matching_target_values_exp == target_vals.id)

# COMMAND ----------

#There shouldn't be much if any data loss after this join. Check.
impressions_products_explode.count()

# COMMAND ----------

impressions_with_target_vals.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Calculating PLA rev on Privacy Subcomms
# MAGIC I'm going to use regex on the slot name, following the convention in training notebooks https://github.com/8451LLC/piq-data/blob/704cc00931d0ff974f98fd71c4b308d2a2624f81/training_notebooks/04_aggregating_key_metrics.ipynb

# COMMAND ----------

pla_slots = (piq.get_dimension('slot')
        .withColumnRenamed('id', 'slot_id')
        .withColumnRenamed('name', 'slot_name')
        .filter(f.col('slot_name').like('%PLA%')))

pla_slot_list = [str(row.slot_id) for row in pla_slots.select('slot_id').collect()]

# COMMAND ----------

#the clearing_price field isn't in dollars, so we need to divide by 1,000,000
clicks_select_pla = clicks_select.filter(clicks_select.slot_id.isin(pla_slot_list)) \
  .withColumn('clearing_price_dollars', f.col('clearing_price')/1000000)

# COMMAND ----------

# MAGIC %md
# MAGIC Aggregate revenue at the sub-commodity level

# COMMAND ----------

clicks_select_pla_aggs = clicks_select_pla.groupBy('sto_sta_cd', 'pid_fyt_sub_com_cd', 'pid_fyt_sub_com_dsc_tx') \
.agg(f.round(f.sum(f.col('clearing_price_dollars')), 2).alias('total_dollars_paid_clicks')) \
  .orderBy('total_dollars_paid_clicks', ascending = False)

# COMMAND ----------

clicks_select_pla_aggs.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ###Calculating TOA rev on Privacy Subcomms

# COMMAND ----------

# MAGIC %md
# MAGIC #### Retrieve CFIC taxonomies and prepare them to be used as a filter list for our impressions table, since TOAs are not delivered/targeted at the UPC level

# COMMAND ----------

cfic = spark.read.parquet('abfss://landingzone@sa8451entlakegrnprd.dfs.core.windows.net/mart/personalization/prd/pim_product_dim/cfic_master')
cfic_cols = ['CFIC_UPC', 'TAXONOMY', 'FAMILY_TREE_SUB_CLASS_CD', 'FAMILY_TREE_SUB_CLASS_NME']

# COMMAND ----------

privacy_upcs_with_cfic = privacy_upcs.join(cfic.select(cfic_cols), privacy_upcs.con_upc_no == cfic.CFIC_UPC)

# COMMAND ----------

privacy_upcs_with_cfic.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Time to do some regex on the CFIC "TAXONOMY" field, to extract the sub-commodity string in the format we need for joining with our impressions DF

# COMMAND ----------

#######
#Disclaimer: As unavoidable as it is, all of this regex-based cleaning is extremely brittle and prone to user error--be careful, and perform QA on the cleaned output every time you revisit this code.
######

# Explode rows with multiple CFIC taxonomies in one. Such rows are characterized by TAXONOMY values that contain a comma with no white spaces on either side.
# For example: 22 Health & Wellness 014 Children's Medicine 00002 Children's Cold, Cough & Flu,22 Health & Wellness 011 Cold, Cough & Flu 00002 Children's Medicine

privacy_upcs_with_cfic = privacy_upcs_with_cfic.withColumn("TAXONOMY_EXP", f.explode(f.split("TAXONOMY", "(?<!\\s),(?!\\s)")))

#Now, I'm going to grab the subcommodity name substring from TAXONOMY_EXP and give it its own column. The expression I'm using looks for everything between any five-digit number (i.e., the subcomm code) and the end of the line. Also, I will make the captured group (i.e., the subcomm name) uppercase, to agree with the "label" field we will be filtering in our impressions df

privacy_upcs_with_cfic = privacy_upcs_with_cfic.withColumn("SUBCOMM_DSC", f.upper(f.regexp_extract(f.col("TAXONOMY_EXP"), "(?<=(\d{5,})).*", 0)))

#A leading whitespace will sneak into the subommodity name if we don't remove it here.
privacy_upcs_with_cfic = privacy_upcs_with_cfic.withColumn("SUBCOMM_DSC", f.regexp_extract(f.col("SUBCOMM_DSC"), "(?<=\s).*", 0))

# COMMAND ----------

privacy_upcs_with_cfic.display()

# COMMAND ----------

privacy_cfic_subcomms = [str(row.SUBCOMM_DSC) for row in privacy_upcs_with_cfic.select('SUBCOMM_DSC').collect()]
privacy_cfic_subcomms

# COMMAND ----------

# MAGIC %md
# MAGIC #### Aggregate 

# COMMAND ----------

# MAGIC %md
# MAGIC Use the same regex approach we used for PLA to filter the impressions df for TOA slot names 

# COMMAND ----------

toa_slots = (piq.get_dimension('slot')
        .withColumnRenamed('id', 'slot_id')
        .withColumnRenamed('name', 'slot_name')
        .filter(f.col('slot_name').like('%TOA%')))

toa_slot_list = [str(row.slot_id) for row in toa_slots.select('slot_id').collect()]

# COMMAND ----------

#the clearing_price field isn't in dollars, so we need to divide by 1,000,000
impressions_with_target_vals = impressions_with_target_vals.filter(impressions_with_target_vals.slot_id.isin(toa_slot_list)) \
  .withColumn('clearing_price_dollars', f.col('clearing_price')/1000000)

# COMMAND ----------

impressions_with_target_vals.display()

# COMMAND ----------

impressions_with_target_vals.count()

# COMMAND ----------

impressions_for_sensitive_filtering = impressions_with_target_vals.filter(impressions_with_target_vals.label.isin(privacy_cfic_subcomms))

# COMMAND ----------

privacy_subcomms_with_cfic = privacy_upcs_with_cfic.select('SUBCOMM_DSC', 'TAXONOMY_EXP', 'pid_fyt_sub_com_cd', 'pid_fyt_sub_com_dsc_tx').distinct()

# COMMAND ----------

privacy_subcomms_with_cfic.count()

# COMMAND ----------

impressions_filtered_tax = impressions_for_sensitive_filtering.join(privacy_subcomms_with_cfic, impressions_with_target_vals.label == privacy_subcomms_with_cfic.SUBCOMM_DSC)

# COMMAND ----------

#inner join on the WA & NV stores relative to the analysis
impressions_for_sensitive_filtering = impressions_for_sensitive_filtering.join(store_dim_select, impressions_for_sensitive_filtering.div_store == store_dim_select.DIV_STORE)

# COMMAND ----------

impressions_filtered_tax.count()

# COMMAND ----------

impressions_for_sensitive_filtering.count()

# COMMAND ----------

imps_toa_aggs = impressions_for_sensitive_filtering.groupBy('sto_sta_cd', 'label') \
.agg(f.round(f.sum(f.col('clearing_price_dollars')), 2).alias('total_dollars_paid_impressions')) \
  .orderBy('total_dollars_paid_impressions', ascending = False)

# COMMAND ----------

imps_toa_aggs.display()

# COMMAND ----------

imps_toa_aggs.join(privacy_subcomms_with_cfic, imps_toa_aggs.label == privacy_subcomms_with_cfic.SUBCOMM_DSC).display()

# COMMAND ----------

# MAGIC %md #In-Store - Catalina use case

# COMMAND ----------

from pyspark.sql.types import StructField, StructType, StringType, DoubleType

customschema = StructType([
StructField("SUBCOMMOD", StringType(), True),
StructField("PRINTS", DoubleType(), True)
])

catalina = spark.read.csv('/FileStore/shared_uploads/r957507@8451.com/in_store_adhoc/catalina_prints_subcommodity.csv', schema=customschema, header=True)

# COMMAND ----------

catalina.sort("SUBCOMMOD").display()

# COMMAND ----------

list_of_sensitive_subcoms = ["ACNE"]

# COMMAND ----------

sensitive_subcoms.select("SUBCOM DESCRIPTION").sort("SUBCOM DESCRIPTION").display()

# COMMAND ----------

# Confirm we are looking at the right column
acds.products.select('pid_fyt_sub_com_cd', 'pid_fyt_sub_com_dsc_tx').distinct().join(sensitive_subcoms.withColumnRenamed("SUBCOM DESCRIPTION", 'pid_fyt_sub_com_dsc_tx'), 'pid_fyt_sub_com_dsc_tx', 'inner').count()

# COMMAND ----------

acds.products.display()

# COMMAND ----------

acds.products.select('fyt_pmy_dpt_cd','fyt_pmy_dpt_cct_dsc_tx').distinct().display()

# COMMAND ----------

kroger_subcomms = acds.products.select(f.col('pid_fyt_sub_com_dsc_tx').alias('SUBCOMMOD')).distinct().withColumn("KROGER", f.lit(1))

# COMMAND ----------

catalina_subcomms = catalina.withColumn("CATALINA", f.lit(1))

# COMMAND ----------

# These are subcomms that are not in Kroger data and may be sensitive
unclear_catalina_subcomms = ["JOHNSON & JOHNSON","ADULT SINUS","LIPCARE MEDICATED"]

# COMMAND ----------

impact = (
  catalina_subcomms
  .join(kroger_subcomms, 'SUBCOMMOD', 'left')
  .join(sensitive_subcoms.select(f.col("SUBCOM DESCRIPTION").alias("SUBCOMMOD"),f.lit("1").alias("SENSITIVE")), "SUBCOMMOD", "left")
  .withColumn("SENSITIVE", f.when(
    (f.col("KROGER")==1) & (f.col("CATALINA")==1) & (f.col("SENSITIVE").isNull()), "0"
  ).when(
    (f.col("KROGER").isNull()) & (f.col("CATALINA")==1) & (~f.col("SUBCOMMOD").isin(unclear_catalina_subcomms)), "0"
  ).when(
    f.col("SUBCOMMOD").isin(unclear_catalina_subcomms), "UNKNOWN"
  ).otherwise(f.col("SENSITIVE"))
              )
  .withColumn("MIN_IMPACT", f.when(
    (f.col("SENSITIVE") == 0) | (f.col("SENSITIVE")=="UNKNOWN"), 0
  ).when(
    (f.col("SENSITIVE") == 1), f.col("PRINTS")
  ).otherwise(None)
  )
  .withColumn("MAX_IMPACT", f.when(
    (f.col("SENSITIVE") == 1) | (f.col("SENSITIVE")=="UNKNOWN"), f.col("PRINTS")
  ).when(
    (f.col("SENSITIVE") == 0), 0
  ).otherwise(None)
              )
)
(
  impact
  .drop("CATALINA","KROGER")
  .sort(f.col("SENSITIVE").desc(),f.col("PRINTS").desc())
  .display())

# COMMAND ----------

(impact.agg(
  f.sum("PRINTS").alias("TOTAL PRINTS"),
  f.sum("MIN_IMPACT").alias("MIN IMPACT"),
  f.sum("MAX_IMPACT").alias("MAX IMPACT"),
  )
 .withColumn("MIN IMPACT PERCENT", f.col("MIN IMPACT")/f.col("TOTAL PRINTS"))
 .withColumn("MAX IMPACT PERCENT", f.col("MAX IMPACT")/f.col("TOTAL PRINTS"))
 .display())

# COMMAND ----------

(impact.agg(
  f.sum("PRINTS").alias("TOTAL PRINTS"),
  f.sum("MIN_IMPACT").alias("MIN IMPACT"),
  f.sum("MAX_IMPACT").alias("MAX IMPACT"),
  )
 .withColumn("MIN IMPACT PERCENT", f.col("MIN IMPACT")/f.col("TOTAL PRINTS"))
 .withColumn("MAX IMPACT PERCENT", f.col("MAX IMPACT")/f.col("TOTAL PRINTS"))
 .display())

# COMMAND ----------

(
  kroger_subcomms
  .join(catalina_subcomms, 'SUBCOMMOD', 'outer')
  .join(sensitive_subcoms.select(f.col("SUBCOM DESCRIPTION").alias("SUBCOMMOD"),f.lit(1).alias("SENSITIVE")), "SUBCOMMOD", "left")
  .withColumn("SENSITIVE", f.when((f.col("KROGER")==1) & (f.col("SENSITIVE").isNull()), 0).otherwise(f.col("SENSITIVE")))
  .filter("SENSITIVE == 1 or SENSITIVE is null or PRINTS > 0")
  .withColumn("IMPACT", f.when(
    (f.col("KROGER")==1) & (f.col("CATALINA") ==1) & (f.col("SENSITIVE") == 0), 0
    ).when(
      (f.col("KROGER")==1) & (f.col("CATALINA") ==1) & (f.col("SENSITIVE") == 1), f.col("PRINTS")
    ).when(
      (f.col("KROGER").isNull()) & (f.col("CATALINA") ==1), "UNKNOWN - missing Kroger"
    ).when(
      (f.col("KROGER")==1) & (f.col("CATALINA").isNull()), "UNKNOWN - missing Catalina"
    ).otherwise(None)
              )
  #.fillna(0)
  .display()
)
