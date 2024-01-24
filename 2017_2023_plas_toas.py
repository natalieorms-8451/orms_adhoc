# Databricks notebook source
# MAGIC %md
# MAGIC This notebook was the primary tool I used to arrive at 2017-2023 all-time estimates for onsite campaigns (TOAs and PLAs) and impressions. A few things to note here:
# MAGIC   - Due to the 10/23 migration to KAP, we have a bit of a Frankenstein data situation. You will see me grabbing some counts from PIQ/logs dimensions and some from Koddi sources, then piecing them together. With impressions, we don't have to worry about any double counting due to platform migration. But to count campaigns accurately, I reached out to David Fetters for a tally of campaigns that were migrated to KAP by his team, so I didn't double count them. These values are noted and applied in the code where applicable
# MAGIC   - A similar all-time count of campaigns and impressions was generated through 10-10-2022. While I do not have the supporting code/data for this report, I did have the excel. I used 10-11-2022 as the start date for my impression pull in this analysis, and added the numbers I obtained on top.
# MAGIC   - Deduping the Koddi impression log, which is unfortunately necessary as of when this analysis was done, is an expensive task. Due to the half-day turnaround required for this ask, I acknowledged the issue and reported the impression count with dupes to the business. I latter reported an addendum referencing the deduped count, once the .distinct() task completed. Fortunately, the impact is minimal and the duplicate rows constitute <1% of 2023 Koddi impression data.
# MAGIC   - I use 12-31-2023 as the cutoff date for this analysis

# COMMAND ----------

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

piq.get_dimension('campaign').orderBy('start_date', ascending=False).dropna().filter(f.col('budget') > 1.00).display()

# COMMAND ----------

piq_camps_2023 = 11200 #I just sorted by start_date then downloaded the csv to get this number in like 10 seconds
total_camps_2023 = piq_camps_2023 + original_kodi_camps
total_camps_2023

# COMMAND ----------

#The numbers of total and migrated KAP campaigns are provided by David Fetters and reflect counts as of 12-31-2023.
total_koddi_camps = 11620
migrated_koddi_camps = 7744 #these are the dupes that would also appear in PIQ, and we don't want to double count them
original_kodi_camps = total_koddi_camps - migrated_koddi_camps
piq_camps = piq.get_dimension('campaign').dropna().filter(f.col('budget') > 1.00).filter(f.col("status") != "DRAFT").count()

# COMMAND ----------

original_kodi_camps

# COMMAND ----------

total_camps = piq_camps + original_kodi_camps
total_camps

# COMMAND ----------

# MAGIC %md
# MAGIC ^That number looks reasonable, and reflects serious growth since the last all-time report was generated 10-10-2022, where the number of campaigns with non-zero spend was 39,366

# COMMAND ----------

# MAGIC %md
# MAGIC Can we do anything with the spend archive to vet ^this number? Maybe...

# COMMAND ----------

spend_archive = spark.read.parquet("abfss://media@sa8451dbxadhocprd.dfs.core.windows.net/piq_report_archive/performance_parquet")

# COMMAND ----------

spend_archive.display()

# COMMAND ----------

spend_archive.select('DATE').distinct().display()

# COMMAND ----------

spend_archive = spend_archive.withColumn('IMPRESSIONS INT', f.col('IMPRESSIONS').cast(t.IntegerType()))

# COMMAND ----------

spend_archive.select(f.sum('IMPRESSIONS INT')).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ^That impression count is way too low for all 2023, I don't trust it, so I'll be going to the logs instead.

# COMMAND ----------

spend_archive_agg = spend_archive.select('CAMPAIGN ID', 'CAMPAIGN NAME', 'UNITS SOLD').groupBy('CAMPAIGN ID', 'CAMPAIGN NAME').agg(f.sum('UNITS SOLD').alias("TOTAL_UNITS"))

# COMMAND ----------

spend_archive_agg.display()

# COMMAND ----------

spend_archive_agg.count()

# COMMAND ----------

camps_with_spend = spend_archive_agg.where(f.col('TOTAL_UNITS')>0).count() 
camps_with_spend

# COMMAND ----------

original_kodi_camps + camps_with_spend + 39366 #this number is all-time through 10-10-2022

# COMMAND ----------

# MAGIC %md
# MAGIC ^That count of campaigns is a little higher than the one I'm getting from the logs, probably because I'm double counting campaigns that were running around the time the 2022 all-time report ended (I only have the excel deliverable with high-level stats from that report, not the raw data or code, so I can't dedupe). 
# MAGIC
# MAGIC However, this number is a good upper bound to support the more fine-grained all-time pull from the PIQ campaign dimension (with late 2023 deduped Koddi campaigns added on top, of course).

# COMMAND ----------

#Enter impression log analysis start and end dates here
piq_analysis_start = "2022-10-11" #this is where the 2022 all-time report ended, so I'm only going to count impressions since then, and add to the total
piq_analysis_end = "2023-12-31"
koddi_analysis_start = '2023-10-13'
koddi_analysis_end = '2023-12-31'

# COMMAND ----------

impressions = piq.get_log(start_date = piq_analysis_start,
                    end_date = piq_analysis_end,
                    log_type = 'IMPRESSION')

# COMMAND ----------

impressions_2023 = piq.get_log(start_date = "2023-01-01",
                    end_date = piq_analysis_end,
                    log_type = 'IMPRESSION')

# COMMAND ----------

piq_imps_total_2023 = impressions_2023.count()

# COMMAND ----------

piq_imps_total_2023

# COMMAND ----------

piq_imps_total = impressions.count()

# COMMAND ----------

koddi_imps_df = spark.read.format("delta").load('abfss://eda@sa8451mapinbprd.dfs.core.windows.net/streaming/lld/tables/impressions')
koddi_imps_df = koddi_imps_df.select([f.col(c) for c in koddi_imps_df.columns if not c.startswith('__')]).distinct()

# COMMAND ----------

koddi_imps_df.display()

# COMMAND ----------

koddi_imps_2023 = koddi_imps_df.where(f.col('year')=='2023')

# COMMAND ----------

koddi_imps_total = koddi_imps_2023.count()

# COMMAND ----------

koddi_imps_total

# COMMAND ----------

#without koddi log deduping - too computationally expensive
piq_imps_total + koddi_imps_total + 144384953626 #total as of 10-10-2022

# COMMAND ----------

#without koddi log deduping - too computationally expensive
piq_imps_total_2023 + koddi_imps_total

# COMMAND ----------

#with koddi log deduping
piq_imps_total + koddi_dedupe_2023_imps + 144384953626

# COMMAND ----------

#with koddi log deduping
piq_imps_total_2023 + koddi_dedupe_2023_imps

# COMMAND ----------

koddi_dedupe_2023_imps = 15532246190 #session crashed after generating this number, so hard-coding this expensive value

# COMMAND ----------

(242421921874 - 240218827545)/242421921874*100
