# Databricks notebook source
# MAGIC %pip install rbo

# COMMAND ----------

import rbo

# COMMAND ----------

import pandas as pd
import numpy as np
# spark packages
from pyspark.sql.functions import *
import pyspark.sql.functions as f
from pyspark.sql.functions import countDistinct
from pyspark.sql.window import Window
from pyspark.sql import DataFrame
from pyspark.sql.types import *
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

import matplotlib.pyplot as plt

from pyspark.sql.functions import date_format, month, year
from pyspark.sql.functions import broadcast
import datetime
import re


import scipy.stats as stats

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Final datasets that this notebook ends up creating
# MAGIC
# MAGIC Base location - 'abfss://media@sa8451dbxadhocprd.dfs.core.windows.net/publisher_quality_score_project'
# MAGIC 1. Final DF for  PLER evaluation -  
# MAGIC     f'{base_location}/model_evaluation/final_df_for_pler_eval'  
# MAGIC
# MAGIC 2. DF post PLER evaluation -  
# MAGIC     f'{base_location}/model_evaluation/pler'  
# MAGIC
# MAGIC 3. Final DF for RBO evaluation -  
# MAGIC     f'{base_location}/model_evaluation/final_df_for_rbo_eval'  
# MAGIC
# MAGIC 4. DF post RBO Evaluation -  
# MAGIC     f'{base_location}/model_evaluation/rbo'
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### CFIC data

# COMMAND ----------

# UDF to replace the pipe character
charReplace = f.udf(lambda x: x.replace('|','',1))
# abfss://landingzone@sa8451entlakegrnprd.dfs.core.windows.net/mart/personalization/prd/pim_product_dim/cfic_master
# abfss://data@sa8451entlakegrnprd.dfs.core.windows.net/source/core/prd/cfic_pqt/krg_dh_prnsl_cfic_20221208_223241/*
#  -- latest file
cfic = spark.read.parquet('abfss://landingzone@sa8451entlakegrnprd.dfs.core.windows.net/mart/personalization/prd/pim_product_dim/cfic_master')
cfic_cols = ['CFIC_UPC', 'CFIC_ITEM_DESCRIPTION', 'TAXONOMY', 'CATEGORY','CFIC_BRAND']
cfic_data = cfic.select(cfic_cols).withColumnRenamed('CFIC_UPC','sku')\
                .filter(col('TAXONOMY').isNotNull())\
                .withColumn('TAXONOMY_2', regexp_replace('TAXONOMY', '\s*(\d+)\s*', '|'))\
                .withColumn("TAXONOMY_2", charReplace("TAXONOMY_2"))\
                .withColumn('TAXONOMY_2',f.upper(f.col("TAXONOMY_2")))


# COMMAND ----------

cfic_data_cols = ['sku', 'CFIC_ITEM_DESCRIPTION', 'TAXONOMY', 'CATEGORY','TAXONOMY_2','CFIC_BRAND']
cfic_data = cfic_data\
                .withColumn('taxonomy1', regexp_replace('TAXONOMY', '^[0-9]{2}\s+', 'DEPT: '))\
                .withColumn('taxonomy2', regexp_replace('taxonomy1', ',[0-9]{2}\s+', ' ||DEPT: '))\
                .withColumn('taxonomy3', regexp_replace('taxonomy2', '\s+[0-9]{3}\s+', ' |COMMODITY: '))\
                .withColumn('taxonomy4', regexp_replace('taxonomy3', '\s+[0-9]{5}\s+', ' |SUB_COMMODITY: '))\
                .select('taxonomy4',*cfic_data_cols)
cfic_data_pd = cfic_data.toPandas()

cfic_data_pd['taxonomy5'] = cfic_data_pd['taxonomy4'].apply(lambda x: x.split("||") )  

print(cfic_data_pd.shape)


# COMMAND ----------

def extract_dept(input_string):
    start = input_string.find("DEPT:") + len("DEPT:")
    end = input_string.find("|", start)
    return input_string[start:end].strip()

def extract_commodity(input_string):
    start = input_string.find("COMMODITY:") + len("COMMODITY:")
    end = input_string.find("|", start)
    return input_string[start:end].strip()

def extract_sub_commodity(input_string):
    start = input_string.find("SUB_COMMODITY:") + len("SUB_COMMODITY:")
    return input_string[start:].strip()


cfic_data_pd = cfic_data_pd.explode('taxonomy5', ignore_index=True)
cfic_data_pd['dept'] = cfic_data_pd['taxonomy5'].apply(lambda x: extract_dept(x))
cfic_data_pd['commodity'] = cfic_data_pd['taxonomy5'].apply(lambda x: extract_commodity(x))
cfic_data_pd['sub_commodity'] = cfic_data_pd['taxonomy5'].apply(lambda x: extract_sub_commodity(x))

print(cfic_data_pd.shape)
#cfic_data_pd.display()

# COMMAND ----------

cfic_data_exploded_cols= ['sku', 'CFIC_ITEM_DESCRIPTION', 'TAXONOMY', 'CATEGORY','TAXONOMY_2','taxonomy5','CFIC_BRAND','dept','commodity','sub_commodity' ]
cfic_data_exploded = spark.createDataFrame( cfic_data_pd ).select(*cfic_data_exploded_cols)
print(cfic_data_exploded.count())
#cfic_data_exploded.limit(1000).display()

# COMMAND ----------

#### Restricting Each SKU to just one sub commodity and ignoring any other additional mapping
window_spec = Window.partitionBy(cfic_data_exploded['sku']).orderBy(cfic_data_exploded['sku'].desc())

# Add a row number column within each partition
cfic_data_exploded = cfic_data_exploded.withColumn('row_num', row_number().over(window_spec))
cfic_1_to_1_map = cfic_data_exploded.filter(f.col('row_num')==1)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Functions

# COMMAND ----------

# MAGIC %md
# MAGIC ##### PLER

# COMMAND ----------

        
def pler_function (reqId_df, rank_soft_cutoff=4):


  '''
  rank_soft_cutoff :n soft cutoff to anylyse top n ranked items by PIQ
  reqId_df : Pandas dataframe that has the following columns 
             request_ids - all request ids that have at least 1 click
             champ_sku_list : ranked list of skus by champion model in ascending order of rank 
             champ_rank : corresponding ranks of the each skus in the champ_sku_list columns for a given request id by the CHAMPION MODEL
             corresp_chal_rank : corresponding ranks of the each skus in the champ_sku_list columns for a given request id by the CHALLENGER MODEL
             champ_click_ind : corresponding binary indicator (1,0) for skus in the champ_sku_list column that get clicked on... 1- click,  0 - no click

             chal_sku_list : ranked list of skus by CHALLENGER model in ascending order of rank 
             chal_rank : corresponding ranks of the each skus in the champ_sku_list columns for a given request id by the CHAMPION MODEL
             chal_click_ind : corresponding ranks of the each skus in the champ_sku_list columns for a given request id by the CHALLENGER MODEL

  returns a dataframe with error logs for each request Id             
  '''     
        

  request_list = reqId_df.request_id.tolist()

  correct_cnt =0
  error = 0
  
  error_type_dict = {}
  master_error= pd.DataFrame()


  for ind,request in enumerate(request_list):
    
    #print(f' row : {ind} , req id : {request}' )
    champ_list = reqId_df.loc[reqId_df['request_id'].isin([request]),'champ_sku_list'].values[0][:rank_soft_cutoff].tolist()
    chal_list  = reqId_df.loc[reqId_df['request_id'].isin([request]),'chal_sku_list'].values[0][:rank_soft_cutoff].tolist()

    ### Getting 1,0 click indicator for champ and chal
    champ_click_list =  reqId_df.loc[reqId_df['request_id'].isin([request]),'champ_click_ind'].values[0][:rank_soft_cutoff].tolist()
    chal_click_list  =  reqId_df.loc[reqId_df['request_id'].isin([request]),'chal_click_ind'].values[0][:rank_soft_cutoff].tolist()
    

    #identify the rank of clicked items in CHAMP MODEL ..
    champ_clicked_items =  [ champ_list[index] for (index, item) in enumerate(champ_click_list) if item == 1]
    #print(f'clicked items {ind}: {champ_clicked_items}')  

    ################ Ground work for Insertion Error ########################
    if set(champ_list)!= set(chal_list):

      ## Checking if new inserted items are ranked above THE ITEM
      inserted_item_list = list(set(chal_list) - set(champ_list))
      ranks_of_inserted_items = [chal_list.index(item)+1 for item in inserted_item_list ]
      
      ##### NON CLICKED ITEMS RANK in Challenger 
      chal_non_clicked_items = [ chal_list[index] for (index, item) in enumerate(chal_click_list) if item == 0] 
      ranks_of_non_clicked_items = [chal_list.index(item)+1 for item in chal_non_clicked_items ]  

      #### Inserted item clicked if any
      chal_clicked_items =  [ chal_list[index] for (index, item) in enumerate(chal_click_list) if item == 1]

      ins_clicked_items = list(set(chal_clicked_items) & set(inserted_item_list))

    ######################## ##################################################

    for item in champ_clicked_items:

      #Champ Rank  
      champ_pos_ = champ_list.index(item)+1

      if item in chal_list:
        
        chal_pos_ = (chal_list.index(item)+1 )
        
        if chal_pos_ == champ_pos_ :
          

          error_type_dict= { 'request_id':request,
                            'result_type'  : 'Correct-same rank',
                            'item': item}

          correct_cnt +=1
        
        elif chal_pos_ < champ_pos_ :
        

          error_type_dict= { 'request_id':request,
                            'result_type'  : 'Correct-better rank',
                            'item': item}

          correct_cnt +=1
          
        #### SUBSTITUTION ERROR ####
        elif set(champ_list)== set(chal_list):

          error_type_dict= { 'request_id':request,
                            'result_type'  : 'Substitution Error',
                            'item': item}

          error +=1

        #### INSERTION ERROR ####
        elif set(champ_list)!= set(chal_list):

          ### Case when all items above THE ITEM are clicked, then there is no penalty  
          if all( non_click_item_rank > chal_pos_ for non_click_item_rank in ranks_of_non_clicked_items):  
            correct_cnt +=1

            error_type_dict= { 'request_id':request,
                              'result_type'  : 'Correct Substitution',
                              'item': item}

          ## Checking if new inserted items are ranked above THE ITEM 
          elif any(ins_item_rank < chal_pos_ for ins_item_rank in ranks_of_inserted_items ) & \
              any(non_click_item_rank < chal_pos_ for non_click_item_rank in ranks_of_non_clicked_items):
                  
            error_type_dict= { 'request_id' : request,
                              'result_type' : 'Insertion Error Type 1',
                              'item': item }
          
            error += 1
          else:
            error_type_dict= { 'request_id' : request,
                              'result_type' : 'Insertion Error Type 2',
                              'item': item }
          
            error += 1


      else:

        #### DELETION ERROR ####
        error_type_dict= { 'request_id' : request,
                            'result_type' : 'Deletion Error',
                            'item': item }
        error += 1
      master_error = master_error.append(error_type_dict, ignore_index=True)


    #### Adding Clicked Inserted items to Correct ####
    if (set(champ_list)!= set(chal_list)):
      if len(ins_clicked_items)>0:


        for ins_click_item in ins_clicked_items:
          #print(f'{ins_click_item} : {request}')
          correct_cnt +=1

          error_type_dict= { 'request_id':request,
                            'result_type'  : 'Correct-Inserted item',
                            'item': ins_click_item}

          master_error = master_error.append(error_type_dict, ignore_index=True)

          #print(f' correct: {correct_cnt}, wrong : {error}' )
  
  return master_error


# COMMAND ----------

# MAGIC %md
# MAGIC ##### RBO

# COMMAND ----------


def rbo_calculator(reqId_pdf):
  '''
  reqId_pdf: pandas dataframe with following columns
             request_id : id of the reequest
             champ_rank : ranks of the each skus given by the CHAMPION MODEL for a given request id
             chal_rank : corresponding ranks of the each skus by the CHALLENGER MODEL for a given request id 
  
  '''

  request_list = reqId_pdf.request_id.tolist()

  rbo_dict = {}
  master_rbo_df= pd.DataFrame()

  for request in request_list:
    
    
      rank1 = reqId_pdf.loc[reqId_pdf['request_id'].isin([request]),'champ_rank'].values[0].tolist()
      rank2 = reqId_pdf.loc[reqId_pdf['request_id'].isin([request]),'chal_rank'].values[0].tolist()
      rbo_sim = rbo.RankingSimilarity(rank1, rank2).rbo(k=4,p=0.78)
      rbo_dict = {
        'request_id' : request,
        'rbo':rbo_sim
      }

      master_rbo_df = master_rbo_df.append(rbo_dict, ignore_index=True)

  return master_rbo_df


# COMMAND ----------

# MAGIC %md 
# MAGIC #### Kendall Tau

# COMMAND ----------

def kendall_tau_calculator(reqId_pdf):
  '''
  reqId_pdf: pandas dataframe with following columns
             request_id : id of the reequest
             champ_rank : ranks of the each skus given by the CHAMPION MODEL for a given request id
             chal_rank : corresponding ranks of the each skus by the CHALLENGER MODEL for a given request id 
  
  '''

  request_list = reqId_pdf.request_id.tolist()

  kendt_dict = {}
  master_tau_df= pd.DataFrame()

  for request in request_list:
    
    
      rank1 = reqId_pdf.loc[reqId_pdf['request_id'].isin([request]),'champ_rank'].values[0].tolist()
      rank2 = reqId_pdf.loc[reqId_pdf['request_id'].isin([request]),'chal_rank'].values[0].tolist()
      tau,p_value = stats.kendalltau(rank1, rank2)
      kendt_dict = {
        'request_id' : request,
        'tau':tau,
        'pvalue':p_value
      }

      master_tau_df = master_tau_df.append(kendt_dict, ignore_index=True)

  return master_tau_df

# COMMAND ----------

# MAGIC %md
# MAGIC ### Input Datasets

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Champion dataset

# COMMAND ----------

piq_data = spark.read.parquet('abfss://media@sa8451dbxadhocprd.dfs.core.windows.net/publisher_quality_score_project/data/query_request_sample_train_05_07')

print(piq_data.count())


#request id with clicks 
req_Id_w_clicks = piq_data.filter(f.col('click_ind')>0)\
                          .select('request_id')\
                          .distinct()
                        

# COMMAND ----------

### Grouping by all Requests type - Clicked and Non Clicked
reqId_champ_df = piq_data\
                  .orderBy(f.col('request_id'), f.col('score_quality_rowNum'))\
                  .groupBy(col("request_id"),col('query'))\
                  .agg(collect_list(col("sku")).alias("champ_sku_list"),
                       collect_list(col("score_quality_rowNum")).alias("champ_rank"),
                       collect_list(col("click_ind")).alias("champ_click_ind"))\
                  
print(reqId_champ_df.count())

### Grouping only by CLICKED Request type
clicked_reqId_champ_df = reqId_champ_df.join(req_Id_w_clicks, on=['request_id'], how='inner')

print(clicked_reqId_champ_df.count())
#reqId_champ_df.display()                

# COMMAND ----------

# MAGIC %md
# MAGIC #### Challenger Dataset

# COMMAND ----------

piq_data_2 = piq_data\
  .select('sku','request_id','query','click_ind','score_quality_rowNum')\
  .withColumnRenamed('score_quality_rowNum','champ_rank')\
  .toPandas()

print(piq_data_2.shape)

# COMMAND ----------


### Reading Full Retrieval Data ###
full_chal_df = spark.read.parquet('abfss://media@sa8451dbxadhocprd.dfs.core.windows.net/publisher_quality_score_project/data/full_retrival_data')\
  .filter(f.col('retrival_results').isNotNull())

full_chal_df = full_chal_df.withColumn("combined_arrays", arrays_zip(col("retrival_results._1"), col("retrival_results._2")))\
                        .withColumn("sorted_elements", sort_array("combined_arrays", asc=False))\
                        .withColumn("sorted_champ_sku", col("sorted_elements._2"))\
                        .withColumn("sorted_sem_scr", col("sorted_elements._1"))

print(full_chal_df.count())
#full_chal_df.limit(10).display()

# COMMAND ----------

### Convert to Pandas
df_pdf = full_chal_df[['query','request_id','sku_list','sorted_champ_sku','sorted_sem_scr']].toPandas()

### Explode... Boom! ###
df_pdf_exploded = df_pdf.explode(['sorted_champ_sku','sorted_sem_scr'])[['request_id','sorted_champ_sku','sorted_sem_scr']]
print(df_pdf.shape)
print(df_pdf_exploded.shape)
#df_pdf_exploded.display()

# COMMAND ----------


#### Assigning Challenger Rank
df_pdf_exploded['chal_rank']= df_pdf_exploded.groupby(['request_id'])['sorted_sem_scr'].rank(method='first', ascending = False)
df_pdf_exploded.sort_values(by= ['request_id','chal_rank'], inplace = True)
#df_pdf_exploded.display()


# COMMAND ----------

# MAGIC %md 
# MAGIC ##### QA - matching input and output SKUs

# COMMAND ----------

#QA
#### restricting to requests that had 100% retrieval 

chal_op_cnt_pdf = df_pdf_exploded.groupby(['request_id']).agg({
    'sorted_champ_sku': lambda x:  len(list(x)),
    'chal_rank': lambda x: list(x)
})\
.rename(columns={'sorted_champ_sku': 'output_sku_length'}).reset_index()


input_df = piq_data_2\
              .sort_values(by=['request_id','champ_rank'])\
              .groupby(['request_id']).agg({
                  'sku': lambda x:  len(list(x)),
                  'champ_rank': lambda x: list(x)
              }).rename(columns={'sku': 'input_sku_length'}).reset_index()

ip_op_merged_df = input_df.merge(chal_op_cnt_pdf, 
                                 left_on=  ['request_id', 'input_sku_length'],
                                 right_on= ['request_id', 'output_sku_length'], 
                                 how='inner' )


print( f'sum  : {ip_op_merged_df.output_sku_length.sum()},{ip_op_merged_df.input_sku_length.sum()}')
print( f'mean  : {ip_op_merged_df.output_sku_length.mean()},{ip_op_merged_df.input_sku_length.mean()}')

ip_op_merged_df.isnull().sum()



# COMMAND ----------

print(df_pdf_exploded.shape)

# COMMAND ----------

chal_pdf = df_pdf_exploded.merge(ip_op_merged_df[['request_id']], on=['request_id'], how= 'inner')\
                          .merge(piq_data_2, left_on=['request_id','sorted_champ_sku'],
                                             right_on= ['request_id','sku'],
                                             how= 'inner')
print(chal_pdf.shape)
print(piq_data_2.shape)
#chal_pdf.display()

# COMMAND ----------

chal_Pydf = spark.createDataFrame(chal_pdf)
cfic_cols = ['sku', 'CFIC_ITEM_DESCRIPTION','CFIC_BRAND','sub_commodity']
chal_final_PyDf = chal_Pydf\
                  .join(cfic_1_to_1_map.select(*cfic_cols), on=['sku'], how='left')\
                  .orderBy(f.col('request_id'), f.col('chal_rank'))\
                  .groupBy(col("request_id"))\
                  .agg(collect_list(col("sku")).alias("chal_sku_list"),
                       collect_list(col("chal_rank")).alias("chal_rank"),
                       collect_list(col("sorted_sem_scr")).alias("sorted_sem_scr"),
                       collect_list(col("click_ind")).alias("chal_click_ind"),
                       collect_list(col("CFIC_ITEM_DESCRIPTION")).alias("chal_item_desc"),
                       collect_list(col("CFIC_BRAND")).alias("chal_brand"),
                       collect_list(col("sub_commodity")).alias("chal_sub_comm")
                       
                       )

print(chal_final_PyDf.count())
#chal_final_PyDf.limit(10).display()

# COMMAND ----------

# chal_final_pdf = chal_pdf.groupby(['request_id','query']).agg({
#     'sku': lambda x: list(x),
#     'chal_rank': lambda x: list(x),
#     'sorted_sem_scr': lambda x: list(x),
#     'click_ind': lambda x: list(x),
#     'sku' : lambda x: len(list(x))
# }).rename(columns={
#     'sku': 'chal_sku_list',
#     'click_ind': 'chal_click_ind'
# }).reset_index()

# chal_final_pdf.display()

# COMMAND ----------

# chal_final_PyDf = spark.createDataFrame(chal_final_pdf)
print(chal_final_pdf.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Obtaining the corresponding challenger rank column 

# COMMAND ----------

# chal_Pydf\
#         .join(cfic_1_to_1_map.select(*cfic_cols), on=['sku'], how='left')\
#         .orderBy(f.col('request_id'), f.col('champ_rank'))\
#         .filter(f.col('request_id')=='F41860E3-3D8C-48B5-85EF-E0BDFC8D6F8E').display()


# COMMAND ----------

corresp_chal_rank_PyDf = chal_Pydf\
                            .join(cfic_1_to_1_map.select(*cfic_cols), on=['sku'], how='left')\
                            .orderBy(f.col('request_id'), f.col('champ_rank'))\
                            .groupBy(col("request_id"))\
                            .agg(
                                collect_list(col("champ_rank")).alias("champ_rank"),
                                collect_list(col("chal_rank")).alias("corresp_chal_rank"),
                                collect_list(col("CFIC_ITEM_DESCRIPTION")).alias("champ_item_desc"),
                                collect_list(col("CFIC_BRAND")).alias("champ_brand"),
                                collect_list(col("sub_commodity")).alias("champ_sub_comm")
                                 )
print(corresp_chal_rank_PyDf.dtypes)
#corresp_chal_rank_PyDf.display()                            

# COMMAND ----------

# corresp_chal_rank_pdf = chal_pdf\
#   .sort_values(by= ['request_id','champ_rank'])\
#   .groupby(['request_id','query'])\
#   .agg({'chal_rank': lambda x: list(x)})\
#   .rename(columns={'chal_rank':'corresp_chal_rank'})\
#   .reset_index()
 
# corresp_chal_rank_PyDf = spark.createDataFrame(corresp_chal_rank_pdf)
# print(corresp_chal_rank_pdf.shape)
# print(corresp_chal_rank_PyDf.dtypes)
# corresp_chal_rank_PyDf.display()

# COMMAND ----------

corresp_chal_rank_pdf.isna().sum()

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Joining all to get the final dataset

# COMMAND ----------


#### FINAL DF for PLER ####
final_pler_eval_Pydf = clicked_reqId_champ_df.join(chal_final_PyDf,on=['request_id'])\
                              .join(corresp_chal_rank_PyDf[['request_id','corresp_chal_rank']], on=['request_id'])

final_pler_eval_df = final_pler_eval_Pydf.toPandas()
print(f'final PLER df shape : {final_pler_eval_df.shape}')

#final_pler_eval_Pydf.write.mode('overwrite').parquet('abfss://media@sa8451dbxadhocprd.dfs.core.windows.net/publisher_quality_score_project/model_evaluation/final_df_for_pler_eval')

#final_pler_eval_df.display()


# COMMAND ----------

#### FINAL DF for RBO ####
final_rbo_eval_Pydf = reqId_champ_df.join(chal_final_PyDf,on=['request_id'])\
                              .join(corresp_chal_rank_PyDf[['request_id','corresp_chal_rank']], on=['request_id'])\
                              
final_rbo_eval_df = final_rbo_eval_Pydf.toPandas()

#final_rbo_eval_Pydf.write.mode('overwrite').parquet('abfss://media@sa8451dbxadhocprd.dfs.core.windows.net/publisher_quality_score_project/model_evaluation/final_df_for_rbo_eval')

print(f'final RBO df shape : {final_rbo_eval_df.shape}')
#final_rbo_eval_Pydf.limit(100).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Applying PLER

# COMMAND ----------


error_df = pler_function(final_pler_eval_df)
#error_df.display()

# COMMAND ----------

error_df.groupby('result_type', as_index=False).size()

# COMMAND ----------

pler_eval_df = spark.createDataFrame(error_df)
#pler_eval_df.write.mode('overwrite').parquet('abfss://media@sa8451dbxadhocprd.dfs.core.windows.net/publisher_quality_score_project/model_evaluation/pler')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Applying RBO

# COMMAND ----------

rbo_reqid_df = final_rbo_eval_df[['request_id','champ_rank', 'corresp_chal_rank']]
rbo_reqid_df.columns = ['request_id','champ_rank', 'chal_rank']
rbo_reqid_df.shape

# COMMAND ----------

rbo_eval_df = rbo_calculator(rbo_reqid_df)
#rbo_eval_df.display()

# COMMAND ----------

rbo_eval_df = spark.createDataFrame(rbo_eval_df)
rbo_eval_df.write.mode('overwrite').parquet('abfss://media@sa8451dbxadhocprd.dfs.core.windows.net/publisher_quality_score_project/model_evaluation/rbo')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Applying Kendall Tau

# COMMAND ----------

kt_pydf = spark.read.parquet('abfss://media@sa8451dbxadhocprd.dfs.core.windows.net/publisher_quality_score_project/model_evaluation/final_df_for_rbo_eval').select('request_id','champ_rank', 'corresp_chal_rank')

kt_df = kt_pydf.toPandas()
kt_df.columns = ['request_id','champ_rank', 'chal_rank']

kt_eval_df = kendall_tau_calculator(kt_df)
#kt_eval_df.head(3)
kt_eval_pydf = spark.createDataFrame(kt_eval_df)
kt_eval_pydf.write.mode('overwrite').parquet('abfss://media@sa8451dbxadhocprd.dfs.core.windows.net/publisher_quality_score_project/model_evaluation/kendall_tau')


# COMMAND ----------

kt_eval_df = spark.read.parquet('abfss://media@sa8451dbxadhocprd.dfs.core.windows.net/publisher_quality_score_project/model_evaluation/kendall_tau').toPandas().dropna()

plt.hist(kt_eval_df['tau'], bins=30, alpha=0.7, edgecolor='black')
#rbo_df.toPandas().plot.hist(by='rbo' ,bins=12, alpha=0.5)

# Calculate the median
median_value = np.median(kt_eval_df['tau'])

# Draw a line at the median
plt.axvline(median_value, color='blue', linestyle='dashed', linewidth=2, label=f'Median: {median_value:.2f}')

# Add labels and title
plt.xlabel('tau')
#plt.ylabel('Probability Density')
#plt.title('RBO Histogram with Median Line')

# Add a legend
plt.legend()


# COMMAND ----------

kt_eval_df.isnull().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reading in PLER and RBO datasets

# COMMAND ----------

rbo_df = spark.read.parquet('abfss://media@sa8451dbxadhocprd.dfs.core.windows.net/publisher_quality_score_project/model_evaluation/rbo')
#rbo_df.display()

# COMMAND ----------


rbo_pandas_df = rbo_df.toPandas()
plt.hist(rbo_pandas_df['rbo'], bins=30, alpha=0.7, edgecolor='black')
#rbo_df.toPandas().plot.hist(by='rbo' ,bins=12, alpha=0.5)

# Calculate the median
median_value = np.median(rbo_pandas_df['rbo'])

# Draw a line at the median
plt.axvline(median_value, color='blue', linestyle='dashed', linewidth=2, label=f'Median: {median_value:.2f}')

# Add labels and title
plt.xlabel('RBO')
#plt.ylabel('Probability Density')
#plt.title('RBO Histogram with Median Line')

# Add a legend
plt.legend()


# COMMAND ----------

rbo_0_val_requests = ['0882243D-11CD-46B0-9E1B-383F9A5DD926','090A017B-0738-43CA-9EA1-84851E42F337','0CD444F4-D0FF-4207-927D-2ABEA32ADA1B','0E530F9F-CDE9-4BC0-9556-CE378C15FB63']
rbo_reqid_df.loc[rbo_reqid_df['request_id'].isin(rbo_0_val_requests),:]

# COMMAND ----------

# MAGIC %md
# MAGIC #### Additional PLER Insights

# COMMAND ----------

cfic_cols =['sku', 'CFIC_ITEM_DESCRIPTION', 'TAXONOMY', 'CFIC_BRAND','dept', 'commodity', 'sub_commodity']
cfic = cfic_1_to_1_map.select(*cfic_cols)


# COMMAND ----------

pler_df = spark.read.parquet('abfss://media@sa8451dbxadhocprd.dfs.core.windows.net/publisher_quality_score_project/model_evaluation/pler')\
                    .withColumnRenamed('item', 'pler_item')

pler_final_df = clicked_reqId_champ_df.join(chal_final_PyDf,on=['request_id'])\
                              .join(corresp_chal_rank_PyDf, on=['request_id'])\
                              .join(pler_df, on=['request_id'], how='inner')

print(f' {pler_df.count()} : {pler_final_df.count()}')
print(pler_final_df.columns)
#pler_final_df.limit(100).display()

# COMMAND ----------

pler_final_df.write.mode('overwrite').parquet('abfss://media@sa8451dbxadhocprd.dfs.core.windows.net/publisher_quality_score_project/model_evaluation/pler')

# COMMAND ----------

# MAGIC %md 
# MAGIC ###### %age breakout by Error

# COMMAND ----------

pler_final_df = spark.read.parquet('abfss://media@sa8451dbxadhocprd.dfs.core.windows.net/publisher_quality_score_project/model_evaluation/pler')
print(pler_final_df.count())

pler_final_df = pler_final_df\
                      .join(cfic_1_to_1_map.select('sku','CFIC_BRAND','sub_commodity'), pler_final_df.pler_item==cfic_1_to_1_map.sku, how='left')\
                      .join(clicked_reqId_champ_df.select('request_id','query'), on=['request_id'],how='inner')\
                      .drop(f.col('sku'))\
                      .withColumnRenamed('chal_item_desc','chal_item_desc_list' )\
                      .withColumnRenamed('chal_brand','chal_brand_list' )\
                      .withColumnRenamed('chal_sub_comm','chal_sub_comm_list' )\
                      .withColumnRenamed('champ_item_desc','champ_item_desc_list' )\
                      .withColumnRenamed('champ_brand','champ_brand_list' )\
                      .withColumnRenamed('champ_sub_comm','champ_sub_comm_list' )\
                      .withColumnRenamed('CFIC_BRAND','pler_item_brand' )\
                      .withColumnRenamed('sub_commodity','pler_item_sub_comm' )

# print(pler_final_df.count())
# pler_final_df.limit(10).display()

pler_final_df.write.mode('overwrite').parquet('abfss://media@sa8451dbxadhocprd.dfs.core.windows.net/publisher_quality_score_project/model_evaluation/pler_v2')

# COMMAND ----------

 


# Calculate total count of distinct request_id
total_count = pler_final_df.select(f.countDistinct('request_id')).first()[0]

pler_final_df.groupBy('result_type').agg(f.countDistinct('request_id').alias('Dcnt_request_id'))\
                                   .withColumn('Percentage', (f.col('Dcnt_request_id') / total_count) * 100)\
                                   .display()


# COMMAND ----------

windowSpec = Window.partitionBy(pler_final_df['pler_item_sub_comm']).orderBy()

pler_final_df.groupBy('pler_item_sub_comm','result_type').agg(f.countDistinct('request_id').alias('Dcnt_request_id'))\
                                                      .withColumn('sc_total', sum(f.col("Dcnt_request_id")).over(windowSpec))\
                                                     .withColumn('Percentage', f.round( (f.col('Dcnt_request_id')/f.col('sc_total'))*100, 1))\
                                                      .orderBy(f.col('sc_total').desc())\
                                                     .display()

# COMMAND ----------

windowSpec = Window.partitionBy(pler_final_df['pler_item_brand']).orderBy()

pler_final_df.groupBy('pler_item_brand','result_type').agg(f.countDistinct('request_id').alias('Dcnt_request_id'))\
                                                         .withColumn('brand_total', sum(f.col("Dcnt_request_id")).over(windowSpec))\
                                                         .withColumn('Percentage', f.round( (f.col('Dcnt_request_id')/f.col('brand_total'))*100, 1))\
                                                         .orderBy(f.col('brand_total').desc())\
                                                         .display()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Sampling for Qualitative eval

# COMMAND ----------

len(pler_final_Pdf.columns)

# COMMAND ----------

cols = ['request_id','query', 'result_type', 'pler_item', 'pler_item_brand',
       'pler_item_sub_comm',
       'champ_sku_list', 'champ_rank', 'corresp_chal_rank', 'champ_click_ind',
       'chal_sku_list', 'chal_rank', 'sorted_sem_scr', 'chal_click_ind',
       'chal_item_desc_list', 'chal_brand_list', 'chal_sub_comm_list',
       'champ_item_desc_list', 'champ_brand_list',
       'champ_sub_comm_list' ]
len(cols)       

# COMMAND ----------

pler_final_Pdf = pler_final_df.toPandas()
sampled_pler_df = pler_final_Pdf.sample(800, random_state=8451)
#sampled_pler_df[cols].display()

# COMMAND ----------

sampled_pler_Pydf = spark.createDataFrame(sampled_pler_df[cols])
sampled_pler_Pydf.write.mode('overwrite').parquet('abfss://media@sa8451dbxadhocprd.dfs.core.windows.net/publisher_quality_score_project/model_evaluation/sampled_df_for_qual_pler_eval')

# COMMAND ----------

# MAGIC %md
# MAGIC ### QA
# MAGIC ##### Full Retrieval data

# COMMAND ----------

### Reading Full Retrieval Data ###
full_chal_df = spark.read.parquet('abfss://media@sa8451dbxadhocprd.dfs.core.windows.net/publisher_quality_score_project/data/full_retrival_data')\
  .filter(f.col('retrival_results').isNotNull())

full_chal_df = full_chal_df.withColumn("combined_arrays", arrays_zip(col("retrival_results._1"), col("retrival_results._2")))\
                        .withColumn("sorted_elements", sort_array("combined_arrays", asc=False))\
                        .withColumn("sorted_champ_sku", col("sorted_elements._2"))\
                        .withColumn("sorted_sem_scr", col("sorted_elements._1"))

### Convert to Pandas
df_pdf = full_chal_df[['query','request_id','sku_list','sorted_champ_sku','sorted_sem_scr']].toPandas()

### Explode... Boom! ###
df_pdf_exploded = df_pdf.explode(['sorted_champ_sku','sorted_sem_scr'])[['request_id','query','sorted_champ_sku','sorted_sem_scr']]
print(df_pdf.shape)
print(df_pdf_exploded.shape)

#### Assigning Challenger Rank
df_pdf_exploded['chal_rank']= df_pdf_exploded.groupby(['request_id'])['sorted_sem_scr'].rank(method='first', ascending = False)
df_pdf_exploded.sort_values(by= ['request_id','chal_rank'], inplace = True)
#df_pdf_exploded.display()


# COMMAND ----------

chal_final_pdf = df_pdf_exploded.groupby(['request_id','query']).agg({
    'sorted_champ_sku': lambda x:  len(list(x)),
    'chal_rank': lambda x: list(x)
})\
.rename(columns={'sorted_champ_sku': 'output_sku_length'}).reset_index()

#chal_final_pdf.display()

# COMMAND ----------

piq_data_2 = piq_data\
  .select('sku','request_id','query','click_ind','score_quality_rowNum')\
  .withColumnRenamed('score_quality_rowNum','champ_rank')\
  .toPandas()

input_df = piq_data_2\
              .sort_values(by=['request_id','champ_rank'])\
              .groupby(['request_id']).agg({
                  'sku': lambda x:  len(list(x)),
                  'champ_rank': lambda x: list(x)
              }).rename(columns={'sku': 'input_sku_length'}).reset_index()

ip_op_merged_df = input_df.merge(chal_final_pdf, 
                                 left_on=  ['request_id', 'input_sku_length'],
                                 right_on= ['request_id', 'output_sku_length'], 
                                 how='inner' )

print( f'sum  : {ip_op_merged_df.output_sku_length.sum()},{ip_op_merged_df.input_sku_length.sum()}')
print( f'mean  : {ip_op_merged_df.output_sku_length.mean()},{ip_op_merged_df.input_sku_length.mean()}')

ip_op_merged_df.isnull().sum()

# COMMAND ----------


