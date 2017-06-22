# Otto, tune number of threads
from pandas import read_csv
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import time
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot
print(matplotlib.get_backend())
# load data
data = read_csv('../data/otto-group-product-classification/train.csv')
print("===data-read===")
dataset = data.values
# split data into X and y
X = dataset[:,0:94]
y = dataset[:,94]
# encode string class values as integers
label_encoded_y = LabelEncoder().fit_transform(y)
# evaluate the effect of the number of threads
results = []
num_threads = [8]
for n in num_threads:
  start = time.time()
  model = XGBClassifier(n_jobs=n)
  model.fit(X, label_encoded_y)
  elapsed = time.time() - start
  print(n, elapsed)
  print("---")
  results.append(elapsed)
# plot results
print("===plotting===")
pyplot.interactive(False)
pyplot.figure()
pyplot.plot(num_threads, results)
pyplot.ylabel('Speed (seconds)')
pyplot.xlabel('Number of Threads')
pyplot.title('XGBoost Training Speed vs Number of Threads')
pyplot.show(block=True)