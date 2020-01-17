################################## PACKAGES ##############################################################################
from utils import * #Not in zipline
import time
import numpy as np
from mxnet import nd, autograd, gluon #Not in zipline
from mxnet.gluon import nn, rnn  #Not in zipline
import mxnet as mx #Not in zipline
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
context = mx.cpu(); model_ctx=mx.cpu()
mx.random.seed(1719)


################################## LOADING DATA ########################################################################

import pandas as pd
import quandl
import datetime
quandl.ApiConfig.api_key = "tPcvbv--Kqz3Y61s6X1P"

import yfinance as yf

# Get the data for the stock Apple by specifying the stock ticker, start date, and end date
data = yf.download('AAPL','2014-01-01','2019-11-17')


dataset_ex_df = data
#dataset_ex_df = pd.read_csv('C:/Users/Dinesh/Downloads/Alpha.csv', header=0, parse_dates=[0], date_parser=parser)

#dataset_ex_df.head(20).index

print(data)

dataset_ex_df['Close'] = dataset_ex_df['Adj Close']
#dataset_ex_df['Date'] = dataset_ex_df['date']

print('There are {} number of days in the dataset.'.format(dataset_ex_df.shape[0]))

num_training_days = int(dataset_ex_df.shape[0]*.7)
print('Number of training days: {}. Number of test days: {}.'.format(num_training_days, \
                                                                    dataset_ex_df.shape[0]-num_training_days))


def get_technical_indicators(dataset):
    # Create 7 and 21 days Moving Average
    dataset['ma7'] = dataset['Close'].rolling(window=7).mean()
    dataset['ma21'] = dataset['Close'].rolling(window=21).mean()

    # Create MACD
    dataset['26ema'] = dataset['Close'].ewm(span=26).mean()
    dataset['12ema'] = dataset['Close'].ewm(span=12).mean()
    dataset['MACD'] = (dataset['12ema']-dataset['26ema'])
#ewm(span=2).mean()
    # Create Bollinger Bands
    dataset['20sd'] = dataset['Close'].rolling(20).std()
    dataset['upper_band'] = dataset['ma21'] + (dataset['20sd']*2)
    dataset['lower_band'] = dataset['ma21'] - (dataset['20sd']*2)

    # Create Exponential moving average
    dataset['ema'] = dataset['Close'].ewm(com=0.5).mean()

    # Create Momentum
    dataset['momentum'] = dataset['Close']-1

    return dataset

dataset_TI_df = get_technical_indicators(dataset_ex_df)

dataset_TI_df.head()

import math

for i in dataset_TI_df['momentum']:
    dataset_TI_df['log_momentum'] = math.log(i)


#This is Googles NLP library. We can use it to do sentiment analysis on the news
#import bert

#Frontier Transformation
data_FT = dataset_ex_df[['Close']]

close_fft = np.fft.fft(np.asarray(data_FT['Close'].tolist()))
fft_df = pd.DataFrame({'fft':close_fft})
fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))



from collections import deque
from statsmodels.tsa.arima_model import ARIMA
from pandas import DataFrame
from pandas import datetime

data_FT = data_FT.iloc[1:]
series = data_FT['Close']
model = ARIMA(series, order=(5, 1, 0))
model_fit = model.fit(disp=0)
print(model_fit.summary())


from pandas import read_csv
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)



'''
# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					mse = evaluate_arima_model(dataset, order)
					if mse < best_score:
						best_score, best_cfg = mse, order
					print('ARIMA%s MSE=%.3f' % (order,mse))
				except:
					continue
	print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
	'''''

error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

dataset_total_df = dataset_TI_df
print('Total dataset has {} samples, and {} features.'.format(dataset_total_df.shape[0], \
                                                              dataset_total_df.shape[1]))


Newer = dataset_TI_df[['Close', 'ma7', 'ma21', '26ema', '12ema', 'MACD', '20sd', 'upper_band', 'lower_band', 'ema', 'momentum', 'log_momentum']].copy()

def get_feature_importance_data(data_income):
    data = data_income.copy()
    y = data['Close']
    X = data.iloc[:, 1:]

    train_samples = int(X.shape[0] * 0.65)

    X_train = X.iloc[:train_samples]
    X_test = X.iloc[train_samples:]

    y_train = y.iloc[:train_samples]
    y_test = y.iloc[train_samples:]

    return (X_train, y_train), (X_test, y_test)


(X_train_FI, y_train_FI), (X_test_FI, y_test_FI) = get_feature_importance_data(Newer)



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = X_train_FI.drop(X_train_FI.index[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]])
y_train = y_train_FI.drop(y_train_FI.index[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]])


X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test_FI)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


classifier = Sequential()


classifier.add(Dense(units = 128, kernel_initializer = 'random_uniform', activation = 'relu', input_dim = X_test_FI.shape[1]))

classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'mean_absolute_percentage_error', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)






########################################################################################################################

#importing required libraries
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

#creating dataframe
data = dataset_ex_df
new_data = pd.DataFrame(index=range(0,len(data)),columns=['Date', 'Close'])
for i in range(0,len(data)):
    new_data['Date'][i] = data.index[i]
    new_data['Close'][i] = data['Close'][i]

#setting index
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

#creating train and test sets
dataset = new_data.values

train = dataset[0:987,:]
valid = dataset[987:,:]

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=150, batch_size=10, verbose=2)

#predicting 246 values, using past 60 from the train data
inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))
rms

#for plotting
train = new_data[:987]
valid = new_data[987:]
valid['Predictions'] = closing_price
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])

valid["Asset"] = "GS"

#Create a new dataframe called predictions
#Forecast = Forecast.drop('Close', 1)
Forecast = pd.DataFrame()
#For alll future predictions we have
Forecast = Forecast.append(valid)

stock = "GS"
valid["Asset"] = "{}".format(stock)




#export_csv = valid.to_csv ('C:/Users/Dinesh/Downloads/Goldman_Forecast_New.csv', index = True, header=True) #Don't forget to add '.csv' at the end of the path

