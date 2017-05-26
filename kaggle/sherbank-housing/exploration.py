import sys
from pathlib import Path
d = Path().resolve().parent.parent
sys.path.insert(0, str(d))
from tabulate import tabulate

import util.utils as utils
import util.plot_utils as plot_utils


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mplt
import matplotlib.pyplot as plt

df = pd.read_csv("../data/sherbank-housing/train.csv")

# print(tabulate(df.head(),headers="keys"))

# df.plot.scatter(x='full_sq', y='price_doc',xlim=[10,2000],title="Price vs Full Area",logx=True)
# plt.show();

plot_utils.plot_numeric_features_filtered("build_count_after_1995","price_doc",df,{'full_sq':[100,200]},strategy="prefix")
