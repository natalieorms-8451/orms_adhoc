# Databricks notebook source
# MAGIC %pip install --upgrade effodata_clickstream

# COMMAND ----------

import pandas as pd
from datetime import datetime, date, timedelta
from piqdata import PiqData
from pyspark.sql import DataFrame, Column, Row
import pyspark.sql.functions as f
import pyspark.sql.types as t
from pyspark.sql.window import Window
from typing import Optional
from effodata_clickstream.effodata_clickstream import CSDM
import matplotlib.pyplot as plt

# COMMAND ----------

csdm = CSDM(spark)

# COMMAND ----------

start_date_analysis = '2023-10-17'
end_date_analysis = '2023-10-17'

# COMMAND ----------

#get the clickstream product and fact tables for the time period of interest
clicks_prod_interest = csdm.get_clickstream_records(start_date=start_date_analysis, end_date=end_date_analysis, join_with=['product','session'])

# COMMAND ----------

clicks_prod_interest.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## A quick dive into view-product effo scenarios. We can refine view-product scenarios by using the component field. component_name = 'internal search' will limit us to product clicks within the internal search page context.

# COMMAND ----------

#limit scenarios to internal-search and view-product
search_clicks = clicks_prod_interest.where(f.col('effo_scenario_name')=='internal-search')
view_prod_clicks = clicks_prod_interest.where(f.col('effo_scenario_name')=='view-product')

# COMMAND ----------

#view-product scenarios can be further refined by the area of the site where the view product action originated. Here
#we want to look at product views from internal search results
view_prod_clicks_internal_search = view_prod_clicks.where(f.col('component_name')=='internal search')

# COMMAND ----------

#how many view product clicks does that leave us with
view_prod_clicks_internal_search.count()

# COMMAND ----------

#do a join with the internal search scenario df to check for data loss. I would not expect there to be much, if any
joined_scenarios = search_clicks.join(view_prod_clicks_internal_search, on=['effo_session', 'upc'])

# COMMAND ----------

joined_scenarios.count()

# COMMAND ----------

#In the previous step, the rowcount after the join exceeded what I expected. This is likely due to multiple visits to a UPC's product page within a single internal search
joined_scenarios.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Main Quest: Analyze deboosting of PLAs over the defined time period, and leverage view-product scenarios to ascribe CTR on deboosted products

# COMMAND ----------

#pull columns needed for the analysis and limit to 'internal-search' scenarios
search_clicks = clicks_prod_interest.where(f.col('effo_scenario_name')=='internal-search').select('effo_session', 'click_date', 'effo_click_id', 'guid', 'impression_id', 'product_item_index', 'monetized_product', 'relevance_score', 'upc', 'product_impression_id')

# COMMAND ----------

search_clicks.display()

# COMMAND ----------

#how many records are we left with?

search_clicks.count()

# COMMAND ----------

#how many unique effo_click_id do we have?

effo_click_ids = search_clicks.select('effo_click_id').distinct().collect()
len(effo_click_ids)

# COMMAND ----------

#how many PLAs do we have?
total_plas = search_clicks.where(f.col('monetized_product')==True).count()

# COMMAND ----------

total_plas

# COMMAND ----------

total_plas/search_clicks.count()

# COMMAND ----------

# Some "effo_click_id" have null "relevance_score" values for monetized slots, which makes them inelligible for this analysis. Here, I'm going to identify click IDs where this is an issue and drop those records in their entirety using an anti-join

null_score_search_clicks = search_clicks.where(f.col('relevance_score').isNull())
search_clicks_real_scores = search_clicks.join(null_score_search_clicks, on='effo_click_id', how='left_anti')

# COMMAND ----------

search_clicks.count()

# COMMAND ----------

#how many records are left after the anti-join?

search_clicks_real_scores.count()

# COMMAND ----------

null_score_search_clicks.count()

# COMMAND ----------

#how many distinct effo_click_ids remain for analysis?

search_clicks_real_scores.select('effo_click_id').distinct().count()

# COMMAND ----------

#gut check that the (number of unique effo_click_ids remaining) = (number we started with) - (number we dropped)

null_score_search_clicks.select('effo_click_id').distinct().count()

# COMMAND ----------

search_clicks_real_scores.orderBy('effo_click_id').display()

# COMMAND ----------

#all columns are strings, so we must recast the relevant ones as numbers or else ranking will be wonky
search_clicks_real_scores = search_clicks_real_scores.withColumn("relevance_score",search_clicks_real_scores.relevance_score.cast('double')) \
                                                    .withColumn("product_item_index",search_clicks_real_scores.product_item_index.cast('int'))


# COMMAND ----------

search_clicks_real_scores.dtypes

# COMMAND ----------

#add a column that ranks the products in every slot for every effo_click_id by their search relevancy scores. We will do this using a window over effo_click_id and the row_number() function. We'll also create a new rank column for product_item_index, because sometimes it is discontinous and/or starts at values other than 1

##NOTES FROM CODE REVIEW##

#examine dense_rank() -- this is worse than rank for this use case, I think, because all products that tie for slot 1 cant share slot 1. row_num() is actually better than rank() or dense_rank()
#look at effo_click_ids where relevance_score() are all the same? how many? should we drop (or dense_rank()) -- switch to row_num() as ranking function (because we can't have ties) and simultaneously orderBy product_item_index as a secondary sort. This should minimize rank_diff as much as possible
effo_click_window_index  = Window.partitionBy("effo_click_id").orderBy(f.col("product_item_index").asc())
search_clicks_index_rank = search_clicks_real_scores.withColumn("product_item_index_rank",f.row_number().over(effo_click_window_index)) 
effo_click_window  = Window.partitionBy("effo_click_id").orderBy(f.col("relevance_score").desc(), f.col("product_item_index_rank").asc())
search_clicks_relevance_rank = search_clicks_index_rank.withColumn("relevance_rank",f.row_number().over(effo_click_window)) 

# COMMAND ----------

#have a look at the resulting df

search_clicks_relevance_rank.display()

# COMMAND ----------

#we're now ready to compare where the product appeared on the search results page (product_item_index) with where it would have appeared organically (relevance_rank). We are only concerned with monetized products for the time being, because we want to know how frequently and to what degree advertisers are paying for a monetized spot that is lower than where the product would have appeared organically.

#first, we filter for monetized products and add a new column: rank_diff = relevance_rank - product_item_index. Negative values of rank_diff imply downgrade

search_clicks_relevance_rank = search_clicks_relevance_rank.withColumn('rank_diff', f.col('relevance_rank')-f.col('product_item_index_rank'))
search_clicks_monetized = search_clicks_relevance_rank.where(f.col('monetized_product')==True)

# COMMAND ----------

#lets take a look at the resulting df and make sure we did what we intended to do 

search_clicks_monetized.display()

# COMMAND ----------

#how many products (rows) do we have for analysis?

search_clicks_monetized.count()

# COMMAND ----------

search_clicks_monetized.select('effo_click_id').distinct().count()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Now lets pull some stats for analysis

# COMMAND ----------

#here is the first result we are interested in: frequency of the different rank_diff to show the scope of the problem

search_clicks_monetized.groupBy('rank_diff').count().display()

# COMMAND ----------

#how many products were deboosted (rank_diff < 0)
neg_rank_diff = search_clicks_monetized.where(f.col("rank_diff")<0)
neg_rank_diff_count = neg_rank_diff.count()

# COMMAND ----------

neg_rank_diff_count

# COMMAND ----------

#what fraction of all PLAs (including ones we had to filter out because they had empty search relevancy scores) are deboosted?
frac_all_pla_deboosted = neg_rank_diff_count/total_plas

# COMMAND ----------

frac_all_pla_deboosted

# COMMAND ----------

#what fraction of all search results are deboosted PLAs?
frac_neg_rank_diff_total = neg_rank_diff_count/search_clicks.count()

# COMMAND ----------

frac_neg_rank_diff_total

# COMMAND ----------

#join with the view_product internal-search component table to see what fraction of deboosted PLAs were actually clicked on (this means an advertiser was billed for the deboosted spot at CPC)
neg_rank_clicks = neg_rank_diff.join(view_prod_clicks_internal_search, on=['effo_session', 'upc'])

# COMMAND ----------

neg_rank_clicks.count()

# COMMAND ----------

frac_neg_rank_click = neg_rank_clicks.count()/neg_rank_diff_count

# COMMAND ----------

# TAKEAWAY: only 1% of deboosted PLAs are actually clicked
frac_neg_rank_click

# COMMAND ----------

#finally, get clicks on deboosted PLAs as a percentage of all PLAs?
frac_neg_rank_click_total_pla = neg_rank_clicks.count()/search_clicks_monetized.count()

# COMMAND ----------

frac_neg_rank_click_total_pla

# COMMAND ----------

#Out of curiosity, what is the click % of boosted PLAs?
pos_rank_diff = search_clicks_monetized.where(f.col("rank_diff")>=0)
pos_rank_clicks = pos_rank_diff.join(view_prod_clicks_internal_search, on=['effo_session', 'upc'])

# COMMAND ----------

frac_pos_rank_clicks = pos_rank_clicks.count()/pos_rank_diff.count()

# COMMAND ----------

#TAKEAWAY fraction of boosted PLAs that are clicked is more than 2x that of deboosted
frac_pos_rank_clicks

# COMMAND ----------

frac_pos_rank_click_total_pla = pos_rank_clicks.count()/search_clicks_monetized.count()
frac_pos_rank_click_total_pla
