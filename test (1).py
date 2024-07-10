import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import pandas_datareader as data
import yfinance as yf
from keras.models import load_model
import streamlit as st
from datetime import date
from stocknews import StockNews
#import plotly.express as px

start = st.sidebar.date_input('Start Date')
#end = '2023-01-01'
#end = date.today()
end = st.sidebar.date_input('End Date')

#@st.cache
st.title('Stock Trend Prediction')
user_input = st.text_input('Enter Stock Ticker' , 'AAPL')
#df = data.DataReader('AAPL', 'yahoo' , start , end)
df = yf.download(user_input, start=start, end=end)

pricing_data , prediction , news = st.tabs(['Pricing','Prediction','News'])

with pricing_data:
    st.write('Price')


    #DESCRIBING DATA
    st.subheader('Data from 2010 - 2023')
    st.write(df.describe())

    #data visualization
    st.subheader('Closing Price vs Time')
    fig = plt.figure(figsize=(12,6))
    plt.plot(df.Close)
    st.pyplot(fig)

    st.subheader('Closing Price vs Time with 100MA')
    ma100 = df.Close.rolling(100).mean()
    ma100_series = pd.Series(ma100)  # Convert ma100 to a Series explicitly
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100_series)
    plt.plot(df.Close)
    st.pyplot(fig)


    st.subheader('Closing Price vs Time with 100MA & 200MA')
    ma100 = df.Close.rolling(100).mean()
    ma100_series = pd.Series(ma100)  # Convert ma100 to a Series explicitly
    ma200 = df.Close.rolling(200).mean()
    ma200_series = pd.Series(ma200)
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100_series , 'r')
    plt.plot(ma200_series , 'b')
    plt.plot(df.Close , 'g')
    st.pyplot(fig)

    #Splitting data for training and testing

    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range = (0,1))

    data_training_array = scaler.fit_transform(data_training)

    #load my model
    model = load_model('keras_model.h5')

    #testing part
    past_100_days = data_training.tail(100)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
    input_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append((input_data[i-100: i]))
        y_test.append(input_data[i, 0])

    x_test = np.array(x_test)
    y_test = np.array(y_test)
    y_predicted = model.predict(x_test)

    #for factor by which data has been scaled down
    scaler = scaler.scale_
    scale_factor = 1/scaler[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor


    #final graph


with prediction:
    st.write('Prediction')
    st.subheader('Predictions vs Original')
    fig2 = plt.figure(figsize=(12,6))
    plt.plot(y_test , 'b' , label= 'Original price')
    plt.plot(y_predicted , 'r' , label= 'Predicted price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)

with news:
    st.write('News')
    st.subheader(f'News of {user_input}')
    sn = StockNews(user_input , save_news=False)
    df_news = sn.read_rss()
    for i in range(10):
        st.subheader(f'News {i+1}')
        st.write(df_news['published'][i])
        st.write(df_news['title'][i])
        st.write(df_news['summary'][i])
        title_sentiment = df_news['sentiment_tite'][i]
        st.write(f'Title sentiment {title_sentiment}')
        news_sentiment = df_news['sentiment_summary'][i]
        st.write(f'News sentiment {news_sentiment}')

