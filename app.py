import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st

st.title('Stock trend prediction')
data = pd.read_csv("C:/Users/rohit/Downloads/Google_train_data.csv")
data.head()

#describing data
st.subheader('Description:')
st.write(data.describe())

#visualizations
st.subheader('Opening Price vs Time Graph')
fig= plt.figure(figsize=(12,6))
plt.plot(data.Open,'b')
st.pyplot(fig)

st.subheader('Opening Price vs Time Graph with 100MA')
ma100= data.Open.rolling(100).mean()
fig= plt.figure(figsize=(12,6))
plt.plot(ma100, 'g')
plt.plot(data.Open, 'b')
st.pyplot(fig)

st.subheader('Opening Price vs Time Graph with 100MA and 200MA')
ma100= data.Open.rolling(100).mean()
ma200= data.Open.rolling(200).mean()
fig= plt.figure(figsize=(12,6))
plt.plot(ma200, 'r')
plt.plot(ma100, 'g')
plt.plot(data.Open, 'b')
st.pyplot(fig)


training_set = data.iloc[:,1:2].values

print(training_set)
print(training_set.shape)

from sklearn.preprocessing import MinMaxScaler
data["Close"]=pd.to_numeric(data.Close,errors='coerce') #to change 'close' datatype
data= data.dropna() #to drop any missing values
trainData= data.iloc[:,4:5].values
sc = MinMaxScaler(feature_range=(0,1)) #rescale values between 0 and 1 for better performance
trainData = sc.fit_transform(trainData) #transform the data

X_train = []
y_train = []
for i in range(60,1149): #input for lstm is 3d so it will have timestep, sequence length and batch size
    #timestep= 60, seqlen=1149
    X_train.append(trainData[i-60:i,0])
    y_train.append(trainData[i,0])
X_train = np.array(X_train)
y_train = np.array(y_train)

#load the model
model=load_model('model.h5')

#testing data
testData = pd.read_csv("C:/Users/rohit/Downloads/Google_test_data.csv")
testData["Close"]=pd.to_numeric(testData.Close,errors='coerce') #to change 'close' Datatype
testData= testData.dropna() #to drop any missing values
testData= testData.iloc[:,4:5] #to use only 'close' column to train the testData

y_test = testData.iloc[60:,0:].values

#input array for the model

inputClosing = testData.iloc[:,0:].values
inputClosing_scaled = sc.transform(inputClosing)

X_test = []
length = len(testData)
timestep = 60
for i in range(timestep, length):
    X_test.append(inputClosing_scaled[i-timestep:i,0])
X_test = np.array(X_test) #converting x_test data into numpy array
x_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))


y_pred = model.predict(X_test)
predicted_price = sc.inverse_transform(y_pred)


#final graph
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, color= 'red', label= 'Actual Stock Price')
plt.plot(predicted_price, color= 'green', label= 'Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
st.pyplot(fig2)