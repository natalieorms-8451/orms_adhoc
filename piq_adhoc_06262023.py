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

imp = piq.get_log(start_date = '2023-06-15', 
                  end_date = '2023-06-15',
                  log_type = 'IMPRESSION') 

# COMMAND ----------

# Impression ending in "94B" was downgraded from pos 1 to pos 16. Impression ending in "8FE" was downgraded from pos 2 to 12
repos_impressions = imp.filter(f.col('impression_id').isin(['48C07EA7-A1F1-4FC1-9387-133BC146C8FE','7E3AA309-881A-4C6C-BED2-6C443599394B']))

# COMMAND ----------

repos_impressions.display()

# COMMAND ----------

repos_request = imp.filter(f.col('request_id') == '0833182C-4AEF-493C-8F97-C9C0D8CE2E36')

# COMMAND ----------

#Weird that bid and clearing prices are identical and max for these two impressiosn. Let's look at request_id to see all the participants (impressions) in the request
repos_request.display()

# COMMAND ----------

# Bid and clearing prices all identical and max for the above request_id-- every eligible PLA was from the same Pepsi campaign ID, and whatever logic PIQ has for advertisers bidding against themselves maxes the ad spend for every participant in the "second price" auction. 

# Impression_id ending in "94B" (downgraded from pos 1 to pos 16 on the website) was ranked in pos 4 by PIQ --> this is consistent with observation
# Impression_id ending in "8FE" (downgraded from pos 2 to pos 12 on the website) was ranked in pos 3 by PIQ --> this is consistent with observation
