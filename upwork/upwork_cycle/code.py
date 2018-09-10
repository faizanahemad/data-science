import numpy as np # linear algebra
import pandas as pd # pandas for dataframe based data processing and CSV file I/O
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from IPython.core.interactiveshell import InteractiveShell
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.dates as mdates
%matplotlib inline
import seaborn as sns
import math
import gc
import ipaddress
from urllib.parse import urlparse
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from data_science_utils import dataframe as df_utils
from data_science_utils import models as model_utils
from data_science_utils import plots as plot_utils
from data_science_utils.dataframe import column as column_utils

from IPython.display import display, HTML


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report


sns.set_style('whitegrid')
%matplotlib inline
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

np.set_printoptions(threshold=np.nan)




plt.rcParams["figure.figsize"] = (24,4)

pd.set_option('display.max_seq_items', None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')
def string_column_processor(string):
    if string is None:
        return string
    string = string.strip().lower()
    return ' '.join(string.split())
df = pd.read_csv("Inventorys_Sold_BBB.csv")
df = df[:3163]
df.columns = ['id','brand','model','year','type','sold_marketplace','date','price']
df_utils.drop_columns_safely(df,["id"],inplace=True)

known_values = {}
encoders = {}
for column in ['brand','model','type','sold_marketplace',]:
    df[column] = df[column].astype(str)
    df[column] = df[column].apply(string_column_processor)
    known_values[column] = set(df[column].unique())
    le = LabelEncoder()
    le.fit(list(df[column].unique())+["unknown"])
    df[column] = le.transform(df[column])
    df[column] = df[column].astype(int)
    encoders[column] = le

features = ["brand","model","year","sold_marketplace"]
target = "price"
xgr=XGBRegressor(n_estimators=100, learning_rate=0.6, gamma=2,max_depth=12,n_jobs=48,missing=np.nan)
xgr.fit(df[features],df[target])


cols = ["brand","model","year","price","sold_marketplace"]
df_test1 = pd.read_csv("s1.csv")
df_test2 = pd.read_csv("s2.csv")
df_test3 = pd.read_csv("s3.csv")
df_test4 = pd.read_csv("s4.csv")
df_test1.columns = cols
df_test2.columns = cols
df_test3.columns = cols
df_test4.columns = cols

df_test1["year"] = pd.to_numeric(df_test1["year"], errors='coerce')
df_test2["year"] = pd.to_numeric(df_test2["year"], errors='coerce')
df_test3["year"] = pd.to_numeric(df_test3["year"], errors='coerce')
df_test4["year"] = pd.to_numeric(df_test4["year"], errors='coerce')
def label_encode_test(df):
    for column in ['brand','model','sold_marketplace']:
        df[column] = df[column].astype(str)
        df[column] = df[column].apply(string_column_processor)
        df.loc[~df[column].isin(known_values[column]),column] = "unknown"
        le = encoders[column]
        df[column] = le.transform(df[column])
        df[column] = df[column].astype(int)
    return df
label_encode_test(df_test1);
label_encode_test(df_test2);
label_encode_test(df_test3);
label_encode_test(df_test4);
def mean_absolute_percentage_error(y_true, y_pred):
    diff = np.abs((y_true - y_pred) / np.clip(np.abs(y_true),1,1e8))
    return 100. * np.mean(diff)
def predict_and_check_error(df):
    preds = np.round(xgr.predict(df[features]))
    y_true = df[target]
    y_pred = preds
    rmse = model_utils.rmse(y_true,y_pred)
    diff = 100.* np.abs((y_true - y_pred) / np.clip(np.abs(y_true),1,1e8))
    count_ten_percent = np.sum(diff<10)
    percent_cols = 100*count_ten_percent/len(y_true)
    mape = mean_absolute_percentage_error(df[target],preds)
    return {"preds":preds,"rmse":rmse,"mape":mape,"ten_percent":count_ten_percent,"ten_in_ten":percent_cols}
r1 = predict_and_check_error(df_test1)
r2 = predict_and_check_error(df_test2)
r3 = predict_and_check_error(df_test3)
r4 = predict_and_check_error(df_test4)
pd.DataFrame({
            "rmse":list(map(lambda r:r["rmse"],[r1,r2,r3,r4])),
             "mape":list(map(lambda r:r["mape"],[r1,r2,r3,r4])),
             "ten":list(map(lambda r:r["ten_percent"],[r1,r2,r3,r4])),
"ten_in_ten":list(map(lambda r:r["ten_in_ten"],[r1,r2,r3,r4]))})

pd.DataFrame({"preds":r1["preds"]}).to_csv("r1.csv",index=False)
pd.DataFrame({"preds":r2["preds"]}).to_csv("r2.csv",index=False)
pd.DataFrame({"preds":r3["preds"]}).to_csv("r3.csv",index=False)
pd.DataFrame({"preds":r4["preds"]}).to_csv("r4.csv",index=False)

