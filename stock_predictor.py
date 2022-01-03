#Description: This program uses an artificial reccurrent neural network called Long Short Term Memory (LSTM)
#             to predict the closing stock price of a corporation (Apple Inc.) using the past 60 day stock price.

#Import the libraries
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import datetime as dt

plt.style.use('fivethirtyeight')
#constants
crypto_currency = 'ETH'
against_currency = 'CAD'
_start = dt.datetime(2016, 1, 1)
_end = dt.datetime.now()

#get the stock quote
df = web.DataReader(f'{crypto_currency}-{against_currency}', data_source='yahoo', start=_start, end=_end)
#Show the data
print(df)

#get the number of rows and columns in the data set
print(df.shape)

#Visualize the closing price history
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price CAD ($)', fontsize=18)
#plt.show()

#Create a new dataframe with only the 'Close column
data = df.filter(['Close'])
#Convert the dataframe to a numpy array
dataset = data.values
#Get the number of rows to train the model on
training_data_len = math.ceil(len(dataset) * .8)

print(training_data_len)

#Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

print(scaled_data)

#Create the training data set
#Create the scaled training data set
train_data = scaled_data[0:training_data_len , :]
#Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])#training data from 1st to 60th values
    y_train.append(train_data[i, 0])#predict value, 61st value
    if i <=60:
        print(x_train)
        print(y_train)
        print()

#Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshape the data
#an LSTM sample requests data be in the form (number samples, number time steps, number features)

print(f'x_train shape: {x_train.shape}')
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
print(f'new x_train shape: {x_train.shape}')#notice 3 d shape

#Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

#Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

#Train the model
#epoch is number of iterations that a forwards+backwards pass counts as
model.fit(x_train, y_train, batch_size=1, epochs=1)

#Create the testing data set
#create a new array containing scaled values from index (training_data_len) to last index
test_data = scaled_data[training_data_len - 60: , :]
#Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])#the 60 prev values before y_test(e.g. 0-59 for 60

#Convert the data to a numpy array
x_test = np.array(x_test)

#Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))#number of rows, (columns = number of time steps, 1 feature

#Get the model's predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)#unscale the values to $ for y scale

#Get the root mean squared error (RMSE) : MODELS how accurately the model predicts the response
rmse = np.sqrt( np.mean(predictions - y_test)**2)
print(rmse)
#amount of $ off prediction is off on average

#Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Ethereum Market Predictor')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price CAD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

#Now try to predict future values more than 1 day in advance.
# Previously 60 prev days of data was used to predict the next day

#Get the quote
apple_quote = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2019-12-17')
#Create a new dataframe
new_df = apple_quote.filter(['Close'])
#Get the last 60 day closing price values and convert the dataframe to an array
last_60_days=new_df[-60:].values
#Scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)
#Create an empty list
x_test = []
#Append the past 60 days
x_test.append(last_60_days_scaled)
#Convert the x_test data set to a numpy array
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
#Get the predicted scaled price
pred_price = model.predict(x_test)
#undo the scaling
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)

#compare to the actual price:
apple_quote2 = web.DataReader('AAPL', data_source='yahoo', start='2019-12-18', end='2019-12-18')
print(apple_quote2['Close'])