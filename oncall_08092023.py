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

# COMMAND ----------

piq = PiqData(spark)

# COMMAND ----------

slot_dim = (piq.get_dimension('slot')
            .withColumnRenamed('id', 'slot_id')
            .withColumnRenamed('name', 'slot_name'))

# COMMAND ----------

campaigns = piq.get_dimension('campaign').where(f.col('id').isin(['212456', '212457', '212458', '205879', '205881', '205882'])).withColumnRenamed('id', 'campaign_id')

# COMMAND ----------

campaigns.display()

# COMMAND ----------

slot_dim.display()

# COMMAND ----------



# COMMAND ----------

def get_impression_data(piq, camp_start_date_min, camp_end_date_max, camp_join, slot):
    """
    get PIQ conversion data for specified start and end dates

    piq: PiqData object
    camp_start_date_min: date parameter for minimum campaign start date
    camp_end_date_max: date parameter for maximum campaign end date
    camp_join: PIQ campaign dimension joined to spark intake form
    slot: PIQ slot dimension

    return: PIQ impression dataframe (spark df)
    """
    imp = (piq.get_log(start_date=camp_start_date_min,
                        end_date=camp_end_date_max,
                        log_type='IMPRESSION')
            .join(f.broadcast(camp_join), ['campaign_id']))
            # filter impressions to between specified start date and specified end date
           # .filter(f.col('transaction_date').between(f.col('camp_start'), f.col('camp_end')))
            #.join(f.broadcast(slot), ['slot_id'])
            #.withColumn('camp_type', f.when(f.col('slot_name').like('%TOA%'), f.lit('TOA')).otherwise(f.lit('PLA'))))

    return imp

# COMMAND ----------

camp_start_unilever = "2023-04-01"
camp_end_unilever = "2023-06-30"

# COMMAND ----------

imp = piq.get_log(start_date=camp_start_unilever,
                        end_date=camp_end_unilever,
                        log_type='IMPRESSION')

# COMMAND ----------

imp.display()

# COMMAND ----------

imp_camps = imp.where(f.col('campaign_id').isin(['212456', '212457', '212458', '205879', '205881', '205882']))

# COMMAND ----------

imp_camps.display()
