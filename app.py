import streamlit as st
import yfinance as yf
from PIL import Image
import requests
from io import BytesIO
import financial
import stockchart
import backtest


if __name__ == '__main__':

    st.title('Trading DashBoard')
    url = 'https://g.foolcdn.com/editorial/images/602904/why-tesla-stock-is-up-today.jpg'
    response = requests.get(url)
    img = Image.open(BytesIO(response. content))


    #Stock Overview Section
    st.image(img, use_column_width=True)
    st.sidebar.header('Enter your Inputs')
    stock_ticker = st.sidebar.text_input("Stock Ticker", "AAPL")
    period_view = st.sidebar.checkbox('period')
    mv_slow = st.sidebar.text_input("Moving Average (Slow)",5)
    mv_fast = st.sidebar.text_input("Moving Average (Fast)",25)
    rsi_period = st.sidebar.text_input("RSI (Period)",14)
    stock_data = yf.Ticker(stock_ticker)

    stockchart.display_stock(period_view, stock_data, rsi_period, stock_ticker.upper(), mv_fast, mv_slow)
    financial.financeStatement_setup(stock_ticker.upper())



    #BackTest Section

    backtest_init = st.sidebar.checkbox('BackTest')

    if backtest_init == True:
        backtest_options = ['SMA','RSI']
        execution = ["Current Day Close Price", "Next Day Open Price"]
        sell_execution = ["Sell All at Next Open", "Customise Sell"]

        st.sidebar.header('Backtest Strategy')
        st.title("BackTest")
        ticker = st.sidebar.text_input("Ticker","AAPL")
        stock_bt = yf.Ticker(ticker)
        stock_close = stock_bt.history(period = "5d", interval = "1d")
        current_price = st.sidebar.text_input( "Current Price" , "%.2f" %stock_close.Close[-1])

        backtest_options = st.sidebar.selectbox("Strategies", backtest_options)
        execution_type = st.sidebar.selectbox("Buy Execution Type", execution)
        sell_execution_type = st.sidebar.selectbox("Sell Execution Type", sell_execution)
        backtest.backtest(backtest_options, stock_bt, ticker.upper(),execution_type,sell_execution_type)


