# Databricks notebook source
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

pd.set_option('display.max_columns', 500)

# COMMAND ----------

opp = piq.get_log(start_date = '2023-08-21', 
                  end_date = '2023-08-21',
                  log_type = 'OPPORTUNITY') 

# COMMAND ----------

opp.display()

# COMMAND ----------

num_requests = opp.count()

# COMMAND ----------

num_requests

# COMMAND ----------

num_opps = opp.agg(f.sum('count_fill')).show()
