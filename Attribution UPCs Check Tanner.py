# Databricks notebook source
import pyspark.sql.functions as f

# COMMAND ----------

cfic = spark.read.parquet('abfss://landingzone@sa8451entlakegrnprd.dfs.core.windows.net/mart/personalization/prd/pim_product_dim/cfic_master')

# COMMAND ----------

product_list = (cfic
 .filter((f.col('CFIC_BRAND') == 'energizer®') |
         (f.col('CFIC_BRAND') == "l'oreal paris") |
         (f.col('CFIC_BRAND') == "hershey's") |
         (f.col('CFIC_BRAND') == 'almond joy') |
         (f.col('CFIC_BRAND') == 'brownberry'))
 .select('CFIC_UPC','CFIC_ITEM_DESCRIPTION','CFIC_BRAND')
 .withColumnRenamed('CFIC_UPC','upc')
 .withColumnRenamed('CFIC_ITEM_DESCRIPTION', 'item_desc')
 .withColumnRenamed('CFIC_BRAND','marketing_brand')
 .withColumn('upc', f.concat(f.lit("'"), f.col('upc')))
 )

# COMMAND ----------

product_list.display()

# COMMAND ----------

output_path = 'abfss://media@sa8451dbxadhocprd.dfs.core.windows.net/Users/t228353/attribution_check/product_list.csv'

# COMMAND ----------

product_list.coalesce(1).write.csv(output_path, header=True, mode='overwrite')

# COMMAND ----------


