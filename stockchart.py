import pandas as pd
import numpy as np
import streamlit as st
from yahoofinancials import YahooFinancials
from datetime import datetime, timedelta
import matplotlib.pyplot as pt
import RSI
import file
import yfinance as yf

def basic_data(ticker):
 
    #yahoo_financials = YahooFinancials(ticker)
    
    yahoo_financials= yf.Ticker(ticker)

    print(yahoo_financials.info)

    
    ps = yahoo_financials.info['priceToSalesTrailing12Months']
    if ps is None:
        ps = np.nan
    pe = yahoo_financials.info['forwardPE']
    if pe is None:
        pe = np.nan
    mktcap = yahoo_financials.info['marketCap']
    
    try:
        divd = yahoo_financials.info['dividendYield'] * 100
        if divd is None:
            divd = np.nan
    except:
        divd = np.nan
        
    high = yahoo_financials.info['fiftyTwoWeekHigh']
    low = yahoo_financials.info['fiftyTwoWeekLow']
    
    try:
        beta = yahoo_financials.info['beta']
        if beta is None:
            beta = np.nan
    except:
        beta = np.nan

    pb = yahoo_financials.info['priceToBook']
    if pb is None:
        pb  = np.nan

    short = yahoo_financials.info['shortPercentOfFloat'] * 100
    if short is None:
        short  = np.nan
    

    df = {'P/S': [ps],
          'P/E': [pe],
          'P/B': [pb],
          'Beta': [beta],
         'Mktcap(M)': [mktcap/1000000],
          'Dividend yield %': [divd],
          'Yearly High': [high],
          'Yearly Low': [low],
          'Shares Short %': [short]

    }
    index = ['Data']
    df = pd.DataFrame(data=df,index=index)
    st.write("General Market Data")
    st.table(df.style.format("{:.2f}"))



def display_stock(period_view, data, rsi_period, stock_ticker, mv_fast, mv_slow):

    if(period_view==True):
        period_options = ['1y', '3mo', '6mo','ytd','2y', '5y', '10y', 'max']
        period = st.sidebar.selectbox("period", period_options)

        if (period == '3mo' or '6mo' or '1y' or 'ytd' or '2y' or '5y' or '10y' or 'max'):
            interval_options = ['1d', '5d', '1wk', '1mo', '3mo']
            interval = st.sidebar.selectbox("interval", interval_options)
            plot_price_volume(period,interval,data,rsi_period, stock_ticker,mv_fast, mv_slow)

    elif (period_view==False):
        start = st.sidebar.text_input("start", datetime.strftime(datetime.today()-timedelta(365),"%Y-%m-%d"))
        end = st.sidebar.text_input("end", datetime.strftime(datetime.today(),"%Y-%m-%d"))
        display_price_volume(data,start,end,rsi_period,stock_ticker, mv_fast, mv_slow)


def display_price_volume(data,start,end,rsi_period,stock_ticker, mv_fast, mv_slow):
    interval_options = ['1d', '5d', '1wk', '1mo']
    interval = st.sidebar.selectbox("interval", interval_options)
    if(end == ''):
        plot_price_volume_3(start,interval,data,rsi_period,stock_ticker, mv_fast, mv_slow)
    else:
        plot_price_volume_2(start,end,interval,data,rsi_period,stock_ticker, mv_fast, mv_slow)



def plot_price_volume(period, interval, data, rsi_period, stock_ticker, mv_fast, mv_slow):

    stock_data = data.history(period = period, interval = interval)
    st.subheader('Close Price')
    st.line_chart(stock_data.Close)

    basic_data(stock_ticker)

    st.subheader('Volume')
    st.line_chart(stock_data.Volume)


    stock_data = RSI.RSI_function(stock_data, rsi_period)
    st.subheader("RSI Data")
    st.line_chart(stock_data.RSI)

    fig = pt.figure(figsize=(8, 5))
    fast = stock_data.Close.rolling(window = int(mv_fast)).mean()
    slow = stock_data.Close.rolling(window = int (mv_slow)).mean()

    pt.plot(stock_data.Close, label='Close Price')
    pt.plot(fast,label = 'mvag ' + mv_fast + ' days')
    pt.plot(slow, label ='mvag ' + mv_slow + ' days')
    pt.legend()
    st.subheader("Moving Averages")
    st.pyplot(fig)

    stock_data = stock_data.reset_index()
    stock_data.Date = convert_datetime(stock_data)
    stock_data = stock_data.iloc[::-1]
    file.download_interface(stock_data,stock_ticker)
    st.write(stock_data)




def plot_price_volume_2(start, end, interval, data, rsi_period, stock_ticker, mv_fast, mv_slow):
    

    stock_data = data.history(start = start, end = end, interval = interval)
    
    st.subheader('Close Price')
    st.line_chart(stock_data.Close)
    basic_data(stock_ticker)


    st.subheader('Volume')
    st.line_chart(stock_data.Volume)

    stock_data = RSI.RSI_function(stock_data, rsi_period)
    st.subheader("RSI Data")
    st.line_chart(stock_data.RSI)


    fig = pt.figure(figsize=(8, 5))
    fast = stock_data.Close.rolling(window = int(mv_fast)).mean()
    slow = stock_data.Close.rolling(window = int (mv_slow)).mean()
    pt.plot(stock_data.Close, label='Close Price')
    pt.plot(fast,label = 'mvag ' + mv_fast + ' days')
    pt.plot(slow, label ='mvag ' + mv_slow + ' days')
    pt.legend()
    st.subheader("Moving Averages")
    st.pyplot(fig)


    stock_data = stock_data.reset_index()
    stock_data.Date = convert_datetime(stock_data)
    st.subheader("Stock Data")
    stock_data = stock_data.iloc[::-1]
    st.write(stock_data)
    file.download_interface(stock_data,stock_ticker)



def plot_price_volume_3(start,interval,data, rsi_period, stock_ticker, mv_fast, mv_slow):
    stock_data = data.history(start = start, interval = interval)

    st.subheader('Close Price')
    st.line_chart(stock_data.Close)

    st.subheader("Basic Data")
    basic_data(stock_ticker)

    st.subheader('Volume')
    st.line_chart(stock_data.Volume)

    stock_data = RSI.RSI_function(stock_data, rsi_period)
    st.subheader("RSI Data")
    st.line_chart(stock_data.RSI)


    fig = pt.figure(figsize=(8, 5))
    fast = stock_data.Close.rolling(window = int(mv_fast)).mean()
    slow = stock_data.Close.rolling(window = int (mv_slow)).mean()
    pt.plot(stock_data.Close, label='Close Price')
    pt.plot(fast,label = 'mvag ' + mv_fast + ' days')
    pt.plot(slow, label ='mvag ' + mv_slow + ' days')
    pt.legend()
    st.pyplot(fig)


    stock_data = stock_data.reset_index()
    stock_data.Date = convert_datetime(stock_data)
    stock_data = stock_data.iloc[::-1]
    st.subheader("Stock Data")
    st.write(stock_data)
    file.download_interface(stock_data,stock_ticker)



def convert_datetime(stock_data):
    dates = []
    for date in stock_data.Date:
        date_obj = date.to_pydatetime()
        dt = date_obj.strftime("%Y-%m-%d")
        dates.append(dt)
    return dates
