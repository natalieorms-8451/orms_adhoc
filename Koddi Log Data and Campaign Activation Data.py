# Databricks notebook source
# MAGIC %pip install --upgrade snowflake-connector-python

# COMMAND ----------

SNOWFLAKE_SOURCE_NAME = "net.snowflake.spark.snowflake"

# COMMAND ----------

# MAGIC %md
# MAGIC Koddi Log Data

# COMMAND ----------

opps_df = spark.read.parquet("abfss://media@sa8451camdev.dfs.core.windows.net/certified/streaming/lld/koddi/event_type=opportunities/")
opps_df.display()

# COMMAND ----------

clicks_df = spark.read.parquet("abfss://media@sa8451camdev.dfs.core.windows.net/certified/streaming/lld/koddi/event_type=clicks/")
clicks_df.display()

# COMMAND ----------

impressions_df = spark.read.parquet("abfss://media@sa8451camdev.dfs.core.windows.net/certified/streaming/lld/koddi/event_type=impressions/")
impressions_df.display()

# COMMAND ----------

conversions_df = spark.read.parquet("abfss://media@sa8451camdev.dfs.core.windows.net/certified/streaming/lld/koddi/event_type=attributed_conversions/")
conversions_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Campaign and Activation Data in Snowflake

# COMMAND ----------

sfOptions = {
"sfURL" : "eighty451.east-us-2.azure.snowflakecomputing.com",
"sfAccount" : "eighty451.east-us-2.azure",
"sfUser" : "SVC_MM_DS_RO",
"sfPassword" : "#daTaMediA@45$caMp",
"sfDatabase" : "MM_CAMPAIGN_ACTIVATION_DEV",
"sfSchema" : "V1",
"sfWarehouse" : "MM_CAMPAIGN_ACTIVATION_NPRD_XS_WH",
"sfRole" : "MM_CAMPAIGN_ACTIVATION_DEV_RO"
}
sf_activations_df = spark.read.format(SNOWFLAKE_SOURCE_NAME).options(**sfOptions).option("dbtable", "activation_v").load()

# COMMAND ----------

sfIntakeOptions = {
"sfURL" : "eighty451.east-us-2.azure.snowflakecomputing.com",
"sfAccount" : "eighty451.east-us-2.azure",
"sfUser" : "SVC_MM_DS_RO",
"sfPassword" : "#daTaMediA@45$caMp",
"sfDatabase" : "MM_CAMPAIGN_INTAKE_DEV",
"sfSchema" : "V1",
"sfWarehouse" : "MM_CAMPAIGN_INTAKE_NPRD_XS_WH",
"sfRole" : "MM_CAMPAIGN_INTAKE_DEV_RO"
}
sf_campaign_intake_df = spark.read.format(SNOWFLAKE_SOURCE_NAME).options(**sfIntakeOptions).option("dbtable", "campaign_v").load()

# COMMAND ----------

sf_activations_df.display()

# COMMAND ----------

sf_campaign_intake_df.display()
