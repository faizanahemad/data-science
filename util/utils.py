
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mplt
import matplotlib.pyplot as plt

def get_all_column_names(df):
    return df.columns.values.tolist()


def count_distinct_values_column(df, colname):
    return pd.DataFrame(df[colname].value_counts(dropna=False)).rename(columns={0: "Count"})


def count_null_per_column(df):
    """Missing value count per column grouped by column name"""
    return pd.DataFrame(df.isnull().sum()).rename(columns={0:"# of Nulls"})

def unique_values_per_column(df):
    unique_counts = {}
    for idx in df.columns.values:
        #cnt=len(df[idx].unique())
        cnt = df[idx].nunique()
        unique_counts[idx]=cnt
    unique_ctr = pd.DataFrame([unique_counts]).T
    unique_ctr_2 = unique_ctr.rename(columns={0: '# Unique Values'})
    return unique_ctr_2


def particular_values_per_column(df,values):
    counts = {}
    for idx in df.columns.values:
        cnt=np.sum(df[idx].isin(values).values)
        counts[idx]=cnt
    ctr = pd.DataFrame([counts]).T
    ctr_2 = ctr.rename(columns={0: '# Values as %s'%values})
    return ctr_2

def get_column_datatypes(df):
    dtype = {}
    for idx in df.columns.values:
        dt = df[idx].dtype
        dtype[idx]=dt
    ctr = pd.DataFrame([dtype]).T
    ctr_2 = ctr.rename(columns={0: 'datatype'})
    return ctr_2

def column_summaries(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum()/len(df)
    particular_ctr = particular_values_per_column(df,[0])
    unique_ctr = unique_values_per_column(df)
    statistical_summary = df.describe().T
    datatypes = get_column_datatypes(df)
    skewed = pd.DataFrame(df.skew()).rename(columns={0: 'skew'})
    mis_val_table = pd.concat([mis_val, mis_val_percent, unique_ctr, particular_ctr,datatypes,skewed,statistical_summary], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% missing of Total Values'})
    return mis_val_table_ren_columns

def filter_dataframe(df,filter_columns):
    df_filtered = df
    for feature in filter_columns:
        values = filter_columns[feature]
        if(len(values)==1):
            df_filtered = df_filtered[df_filtered[feature]==values[0]]
        elif(len(values)==2):
            df_filtered = df_filtered[(df_filtered[feature]>=values[0]) & (df_filtered[feature]<=values[1])]
    return df_filtered

def filter_dataframe_percentile(df, filter_columns):
    df_filtered = df
    for feature in filter_columns:
        quantiles = filter_columns[feature]
        values = df[feature].quantile(quantiles).values
        if(len(values)==1):
            # if only one value present assume upper quantile
            df_filtered = df_filtered[df_filtered[feature]<=values[0]]
        elif(len(values)==2):
            df_filtered = df_filtered[(df_filtered[feature]>=values[0]) & (df_filtered[feature]<=values[1])]
    return df_filtered