import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler

#Input the data, labels need to be column headers. remove missing values
print('reading the data')
columns = ['user','activity','timestamp', 'Feature_A', 'Feature_B', 'Label']
df = pd.read_csv('data/dataset.txt', header = None, names = columns)
df = df.dropna()

#divide the data into sequences
print('preprocessing...')
N_TIME_STEPS = 20
N_FEATURES = 2
step = N_TIME_STEPS
segments = []
labels = []
X = df[['Feature_A','Feature_B']]
Y = df['Label']

scaler = StandardScaler()
X = scaler.fit_transform(X.values)
target_scaler = StandardScaler()
Y = target_scaler.fit_transform(Y.values.reshape(-1, 1))
for i in range(0, len(df) - N_TIME_STEPS, step):
    xs = X[i: i + N_TIME_STEPS]
    label = Y[i:i + N_TIME_STEPS]
    segments.append(xs)
    labels.append(label)

#reshape sequences
reshaped_segments = np.asarray(segments, dtype= np.float32)
reshaped_labels = np.asarray(labels).reshape(-1, N_TIME_STEPS)

print('splitting/reshaping data')
#split into test and train
RANDOM_SEED = 42
X_train, X_test, y_train, y_test = train_test_split(
        reshaped_segments, reshaped_labels, test_size=0.2, random_state=RANDOM_SEED)

#reshape the y sets to make them 2D
y_train = y_train.reshape((y_train.shape[0], y_train.shape[1]))
y_test = y_test.reshape((y_test.shape[0], y_test.shape[1]))

from keras import optimizers
adam = optimizers.Adam(lr=0.01, clipnorm=4., beta_1=0.9, beta_2=0.99, epsilon=1e-06, decay=0.0)

print('building model')
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(N_TIME_STEPS, N_FEATURES))) 
model.add(LSTM(20))
model.compile(loss='mean_squared_error', optimizer=adam)

print('model building...')
print(model.summary())
history = model.fit(X_train, y_train, epochs=70, batch_size=2048, validation_data=(X_test, y_test), verbose=2, shuffle=False)
evaluation = model.evaluate(X_train, y_train, batch_size=None, verbose=2, sample_weight=None)

print('make prediction on test dataset')
prediction = model.predict(X_test, batch_size=None, verbose=2)
prediction = target_scaler.inverse_transform(prediction.reshape(-1, 1)).reshape(-1,step)

# Printing the mean squared error
y_test = target_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1,step)
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test.reshape(-1, 1),prediction.reshape(-1, 1)))
