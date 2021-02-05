import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import backtrader as bt
import backtrader.analyzers as btanalysis
from datetime import datetime,timedelta
import matplotlib.pyplot as pt
import base64
from PIL import Image
import requests
from io import BytesIO





# function to your download dataframes to a csv file
def download_link(object_to_download, download_filename, download_link_text):

    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    b64 = base64.b64encode(object_to_download.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'


def download_interface(stock_data,stock_ticker):
    tmp_download_link = download_link(stock_data, stock_ticker + '.csv', 'Click here to download the Stock data')
    st.markdown(tmp_download_link, unsafe_allow_html=True)


def convert_datetime(stock_data):
    dates = []
    for date in stock_data.Date:
        dt = date.strftime("%Y-%m-%d")
        dates.append(dt)
    return dates


def plot_price_volume(period,interval,data):
    stock_data = data.history(period = period, interval = interval)
    st.write('Close Price')
    st.line_chart(stock_data.Close)
    st.markdown('Volume')
    st.line_chart(stock_data.Volume)

    st.write("Stock Data")
    stock_data = stock_data.reset_index()
    stock_data.Date = convert_datetime(stock_data)
    st.write(stock_data)

    download_interface(stock_data,stock_ticker)


def plot_price_volume_2(start, end, interval, data):
    stock_data = data.history(start = start, end = end, interval = interval)
    st.write('Close Price')
    st.line_chart(stock_data.Close)
    st.markdown('Volume')
    st.line_chart(stock_data.Volume)

    st.write("Stock Data")
    stock_data = stock_data.reset_index()
    stock_data.Date = convert_datetime(stock_data)
    st.write(stock_data)

    fig = pt.figure(figsize=(8, 5))
    fast = stock_data.Close.rolling(window = int(mv_fast)).mean()
    slow = stock_data.Close.rolling(window = int (mv_slow)).mean()
    pt.plot(stock_data.Close, label='Close Price')
    pt.plot(fast,label = 'mvag ' + mv_fast + ' days')
    pt.plot(slow, label ='mvag ' + mv_slow + ' days')
    pt.legend()
    st.pyplot(fig)
    download_interface(stock_data,stock_ticker)

def plot_price_volume_3(start,interval,data):
    stock_data = data.history(start = start, interval = interval)
    st.write('Close Price')
    st.line_chart(stock_data.Close)
    st.markdown('Volume')
    st.line_chart(stock_data.Volume)

    st.write("Stock Data")
    stock_data = stock_data.reset_index()
    stock_data.Date = convert_datetime(stock_data)
    st.write(stock_data)

    fig = pt.figure(figsize=(8, 5))
    fast = stock_data.Close.rolling(window = int(mv_fast)).mean()
    slow = stock_data.Close.rolling(window = int (mv_slow)).mean()
    pt.plot(stock_data.Close, label='Close Price')
    pt.plot(fast,label = 'mvag ' + mv_fast + ' days')
    pt.plot(slow, label ='mvag ' + mv_slow + ' days')
    pt.legend()
    st.pyplot(fig)
    download_interface(stock_data,stock_ticker)


def display_date(data,start,end):
    interval_options = ['1d', '5d', '1wk', '1mo', '3mo']
    interval = st.sidebar.selectbox("interval", interval_options)
    if(end == ''):
        plot_price_volume_3(start,interval,data)
    else:
        plot_price_volume_2(start,end,interval,data)


def display_period(period_view,data):

    if(period_view==True):
        period_options = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y','ytd', 'max','']
        period = st.sidebar.selectbox("period", period_options)

        if (period == '1d' or period == '5d'):
            interval_options = ['1d','1m','5m', '30m', '1h']
            interval = st.sidebar.selectbox("interval", interval_options)
            plot_price_volume(period,interval,data)

        elif period == '1mo':
            interval_options = ['1d','5d','1wk','30m','1h']
            interval = st.sidebar.selectbox("interval", interval_options)
            plot_price_volume(period,interval,data)


        elif (period == '3mo' or '6mo' or '1y' or 'ytd' or '2y' or '5y' or '10y' or 'max'):
            interval_options = ['1d', '5d', '1wk', '1mo', '3mo']
            interval = st.sidebar.selectbox("interval", interval_options)
            plot_price_volume(period,interval,data)

    elif (period_view==False):
        start = st.sidebar.text_input("start", datetime.strftime(datetime.today()-timedelta(365),"%Y-%m-%d"))
        end = st.sidebar.text_input("end", datetime.strftime(datetime.today(),"%Y-%m-%d"))
        display_date(data,start,end)



def backtest(options,stock_bt,ticker):
    cerebro = bt.Cerebro()
    backtest_start = st.sidebar.text_input("Start Period",datetime.strftime(datetime.today()-timedelta(365),"%Y-%m-%d"))
    backtest_end  = st.sidebar.text_input("End Period",datetime.strftime(datetime.today(),"%Y-%m-%d"))
    buy_n_hold  = st.sidebar.checkbox("Buy and Hold")
    trade_size = st.sidebar.text_input("Lot Size for 1 transaction",5000)
    intial_amount = st.sidebar.text_input("Initial Cash",100000)

    if (options=='SMA'):
        SMA_strategy(backtest_start, backtest_end,stock_bt,cerebro, intial_amount,trade_size,ticker,buy_n_hold)

    if(options == 'Simple RSI'):
        RSI_strategy(backtest_start, backtest_end,stock_bt,cerebro, intial_amount,trade_size,ticker,buy_n_hold)





############# Strategies  ######################
def SMA_strategy(start, end, stock_bt,cerebro,initial_amt,size, ticker, buy_n_hold):
    cross_over_fast =  st.sidebar.text_input("Average Cross Over Days",5)
    cross_over_slow = st.sidebar.text_input("Average Days",10)

    class Cross_MA(bt.Strategy):
        def __init__(self):
            slow = bt.ind.SMA(period = int(cross_over_slow))
            fast = bt.ind.SMA(period = int(cross_over_fast))
            self.crossver=bt.ind.CrossOver(fast,slow)

        def next(self):

            if self.crossver>0:
                self.buy()

            if self.position:
                if self.crossver <0:
                    if buy_n_hold == False:
                        self.close()
                    else:
                        self.sell(size=0)

    start = start
    end = end
    start_date = datetime.strptime(start,'%Y-%m-%d')
    end_date = datetime.strptime(end,'%Y-%m-%d')
    data = bt.feeds.YahooFinanceData(dataname=ticker,fromdate = start_date, todate = end_date+timedelta(1))
    back = cerebro_run(cerebro,data,Cross_MA,initial_amt, size)
    SMA_Visualisation(back,start,end,stock_bt,cerebro)


def SMA_Visualisation(back,start,end,stock_bt,cerebro):
    cashDelta_list = []
    date_list = []
    stock_qty = []
    date_column = np.arange(np.datetime64(start), np.datetime64(end))
    for key, value in back[0].analyzers.transaction.get_analysis().items():
        str_date = str(key).split()
        date_list.append(str_date[0])
        cashDelta_list.append(value[0][4])
        stock_qty.append(value[0][0])

    cashDel = []
    index_i = 0
    stock_delta=[]

    for date in date_column:
        if str(date) in date_list:
            stock_delta.append(stock_qty[index_i])
            index_i +=1
        else:
            stock_delta.append(0)

    index_w = 0
    for date in date_column:
        if str(date) in date_list:
            cashDel.append(cashDelta_list[index_w])
            index_w+=1
        else:
            cashDel.append(0)

    Qty = [0+stock_delta[0]]
    stock_delta.pop(0)
    index_q=0
    for delta in stock_delta:
        Qty.append(Qty[index_q] + delta)
        index_q+=1

    pnl_list = [100000+cashDel[0]]

    w = cashDel.pop(0)
    index_x = 0
    for cash in cashDel:
        pnl_list.append(pnl_list[index_x] + cash)
        index_x+=1
    cashDel.insert(0,w)

    data = {'date':date_column ,'cash_del': cashDel,'Stock_Quantity':Qty,'trade_pnl':pnl_list}
    df =pd.DataFrame(data=data)
    stock = stock_bt
    stock_data = stock.history(start = start, end =end)
    stock_data  = stock_data .reset_index()
    data={'date':stock_data ['Date'],'close': stock_data ['Close']}
    final_data =pd.DataFrame(data=data)
    final_df = df.merge(final_data,"inner",'date')

    final_df['final_pnl'] = final_df['Stock_Quantity'] * final_df['close'] + final_df['trade_pnl']
    final_df = final_df.set_index('date')
    fig = pt.figure(figsize=(10, 3))
    st.write("PNL of BackTest")
    pt.plot(final_df.final_pnl)
    st.pyplot(fig)

    pnl = cerebro.broker.getvalue()
    pnl_val = st.sidebar.text_input("PNL of Strategy",pnl)
    sharpe = back[0].analyzers.sharpe.get_analysis()
    sharpe_val = st.sidebar.text_input("Sharpe of Strategy","%.2f" %list(sharpe.values())[0])
    returns = back[0].analyzers.returns.get_analysis()

    transaction = back[0].analyzers.transaction.get_analysis()


    date_column = []
    for date in list(transaction.keys()):
        date = datetime.strftime(date,"%Y-%m-%d")
        date_column.append(date)

    Price = []
    Qty = []
    Cash_change = []
    buy = 0
    sell = 0
    net = 0
    qty = 0
    for transaction in list(transaction.values()):
        Price.append(transaction[0][1])
        Qty.append(transaction[0][0])
        net += transaction[0][4]
        qty += transaction[0][0]
        Cash_change.append(transaction[0][4])
        if int(transaction[0][0]) > 0:
            buy+=1
        else:
            sell +=1;


    transaction_data = {'date':date_column ,'Price': Price,'Quantity':Qty,'trade_pnl': Cash_change}
    df_transaction = pd.DataFrame(data=transaction_data)
    st.write('Trade Records')
    st.write(df_transaction)

    returns = list(returns.values())[0]
    returns_pct = returns*100
    returns_val = st.sidebar.text_input("Returns of Strategy %","%.2f" %returns_pct)
    st.write("Buy : " + str(buy))
    st.write("Sell : " + str(sell))
    stock_close = stock_bt.history(period = "5d", interval = "1d")
    pnl = qty*stock_close.Close[-1] + net
    pnl = round(pnl, 2)
    st.write('Final Pnl : ' + str(pnl))




def RSI_strategy(start, end, stock_bt,cerebro,initial_amt, size, ticker,buy_n_hold):

    RSI_Entry = st.sidebar.text_input("RSI Entry Number",30)
    RSI_Exit = st.sidebar.text_input("RSI Exit Number",70)
    RSI_Period = st.sidebar.text_input("RSI Period", 14)

    a = []


    class RSIStrategy(bt.Strategy):

        def __init__(self):
            self.rsi = bt.indicators.RSI_SMA(self.data.close, period= int(RSI_Period))

        def next(self):

            if self.rsi < int(RSI_Entry):
                self.buy()
            else:
                if self.position:
                    if self.rsi > int(RSI_Exit):
                        if buy_n_hold == False:
                            self.close()
                    else:
                        self.close(size=0)
    start = start
    end = end
    start_date = datetime.strptime(start,'%Y-%m-%d')
    end_date = datetime.strptime(end,'%Y-%m-%d')
    data = bt.feeds.YahooFinanceData(dataname=ticker, fromdate = start_date, todate = end_date)
    back = cerebro_run(cerebro, data, RSIStrategy, initial_amt, size)
    RSI_Visualisation(back,start,end,stock_bt,cerebro)




def RSI_Visualisation(back,start,end,stock_bt,cerebro):
    cashDelta_list = []
    date_list = []
    stock_qty = []
    date_column = np.arange(np.datetime64(start), np.datetime64(end))
    for key, value in back[0].analyzers.transaction.get_analysis().items():
        str_date = str(key).split()
        date_list.append(str_date[0])
        cashDelta_list.append(value[0][4])
        stock_qty.append(value[0][0])

    cashDel = []
    index_i = 0
    stock_delta=[]

    for date in date_column:
        if str(date) in date_list:
            stock_delta.append(stock_qty[index_i])
            index_i +=1
        else:
            stock_delta.append(0)

    index_w = 0
    for date in date_column:
        if str(date) in date_list:
            cashDel.append(cashDelta_list[index_w])
            index_w+=1
        else:
            cashDel.append(0)

    Qty = [0+stock_delta[0]]
    stock_delta.pop(0)
    index_q=0
    for delta in stock_delta:
        Qty.append(Qty[index_q] + delta)
        index_q+=1

    pnl_list = [100000+cashDel[0]]

    w = cashDel.pop(0)
    index_x = 0
    for cash in cashDel:
        pnl_list.append(pnl_list[index_x] + cash)
        index_x+=1
    cashDel.insert(0,w)

    data = {'date':date_column ,'cash_del': cashDel,'Stock_Quantity':Qty,'trade_pnl':pnl_list}
    df =pd.DataFrame(data=data)
    stock = stock_bt
    stock_data = stock.history(start = start, end =end)
    stock_data  = stock_data .reset_index()
    data={'date':stock_data ['Date'],'close': stock_data ['Close']}
    final_data =pd.DataFrame(data=data)
    final_df = df.merge(final_data,"inner",'date')

    final_df['final_pnl'] = final_df['Stock_Quantity'] * final_df['close'] + final_df['trade_pnl']
    final_df = final_df.set_index('date')
    fig = pt.figure(figsize=(10, 3))
    st.write("PNL of BackTest")
    pt.plot(final_df.final_pnl)
    st.pyplot(fig)

    pnl = cerebro.broker.getvalue()
    pnl_val = st.sidebar.text_input("PNL of Strategy",pnl)
    sharpe = back[0].analyzers.sharpe.get_analysis()
    sharpe_val = st.sidebar.text_input("Sharpe of Strategy","%.2f" %list(sharpe.values())[0])
    returns = back[0].analyzers.returns.get_analysis()

    transaction = back[0].analyzers.transaction.get_analysis()


    date_column = []
    for date in list(transaction.keys()):
        date = datetime.strftime(date,"%Y-%m-%d")
        date_column.append(date)

    Price = []
    Qty = []
    Cash_change = []
    buy = 0
    sell = 0
    net = 0
    qty = 0
    for transaction in list(transaction.values()):
        Price.append(transaction[0][1])
        Qty.append(transaction[0][0])
        net += transaction[0][4]
        qty += transaction[0][0]
        Cash_change.append(transaction[0][4])
        if int(transaction[0][0]) > 0:
            buy+=1
        else:
            sell +=1;


    transaction_data = {'date':date_column ,'Price': Price,'Quantity':Qty,'trade_pnl': Cash_change}
    df_transaction = pd.DataFrame(data=transaction_data)
    st.write('Trade Records')
    st.write(df_transaction)

    returns = list(returns.values())[0]
    returns_pct = returns*100
    returns_val = st.sidebar.text_input("Returns of Strategy %","%.2f" %returns_pct)
    st.write("Buy : " + str(buy))
    st.write("Sell : " + str(sell))
    stock_close = stock_bt.history(period = "5d", interval = "1d")
    pnl = qty*stock_close.Close[-1] + net
    pnl = round(pnl, 2)
    st.write('Final Pnl : ' + str(pnl))

















def cerebro_run(cerebro,data,strategy,initial_amount,trade_size):
    cerebro.adddata(data)
    cerebro.addstrategy(strategy)
    cerebro.broker.setcash(int(initial_amount))
    cerebro.addsizer(bt.sizers.SizerFix,stake=int(trade_size))
    cerebro.addanalyzer(btanalysis.SharpeRatio, _name = "sharpe")
    cerebro.addanalyzer(btanalysis.Returns, _name = "returns")
    cerebro.addanalyzer(btanalysis.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(btanalysis.Transactions, _name="transaction")
    back = cerebro.run()
    return back


if __name__ == '__main__':

    st.title('Stock app')
    url = 'https://g.foolcdn.com/editorial/images/602904/why-tesla-stock-is-up-today.jpg'
    response = requests.get(url)
    img = Image.open(BytesIO(response. content))
    st.image(img, use_column_width=True)
    st.sidebar.header('Enter your Inputs')
    stock_ticker = st.sidebar.text_input("Stock Ticker", "AWX.SI")
    stock = yf.Ticker(stock_ticker)
    period_view = st.sidebar.checkbox('period')
    mv_slow = st.sidebar.text_input("Slow",5)
    mv_fast = st.sidebar.text_input("Fast",25)
    display_period(period_view,stock)

    st.sidebar.header('Backtest Strategy')
    ticker = st.sidebar.text_input("Ticker","AWX.SI")
    stock_bt = yf.Ticker(ticker)
    stock_close = stock_bt.history(period = "5d", interval = "1d")
    current_price = st.sidebar.text_input( "Current Price" ,"%.2f" %stock_close.Close[-1])
    backtest_options = ['SMA','Simple RSI']
    backtest_options= st.sidebar.selectbox(" ", backtest_options)
    backtest(backtest_options, stock_bt, ticker)
