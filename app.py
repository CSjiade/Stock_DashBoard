import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
from yahoofinancials import YahooFinancials
import backtrader as bt
import backtrader.analyzers as btanalysis
from datetime import datetime, timedelta
import matplotlib.pyplot as pt
import base64
from PIL import Image
import requests
from io import BytesIO



def basic_data(ticker):

    yahoo_financials = YahooFinancials(ticker)
    ps = yahoo_financials.get_price_to_sales()
    if ps is None:
        ps = np.nan
    pe = yahoo_financials.get_pe_ratio()
    if pe is None:
        pe = np.nan
    mktcap = yahoo_financials.get_market_cap()
    divd = yahoo_financials.get_dividend_yield()
    if divd is None:
        divd = np.nan
    high = yahoo_financials.get_yearly_high()
    low = yahoo_financials.get_yearly_low()
    beta = yahoo_financials.get_beta()
    if beta is None:
        beta = np.nan
    df = {'P/S': [ps], 'P/E': [pe], 'Beta': [beta],
         'Mktcap(M)': [mktcap/1000000], 'Dividend yield %': [divd],
          'Yearly High': [high],
          'Yearly Low': [low]

    }
    index = ['Data']
    df = pd.DataFrame(data=df,index=index)
    st.write("General Market Data")
    st.table(df.style.format("{:.2f}"))



def RSI_function(df,period):
    period = int(period)
    df['Up Move'] = np.nan
    df['Down Move'] = np.nan
    df['Average Up'] = np.nan
    df['Average Down'] = np.nan
    df['RS'] = np.nan
    df['RSI'] = np.nan

    for x in range(1, len(df)):

        df['Up Move'][x] = 0
        df['Down Move'][x] = 0

        if df['Close'][x] > df['Close'][x-1]:
            df['Up Move'][x] = df['Close'][x] - df['Close'][x-1]

        if df['Close'][x] < df['Close'][x-1]:
            df['Down Move'][x] = abs(df['Close'][x] - df['Close'][x-1])

    df['Average Up'][period] = df['Up Move'][1:period].mean()
    df['Average Down'][period] = df['Down Move'][1:period].mean()
    df['RS'][period] = df['Average Up'][period] / df['Average Down'][period]
    df['RSI'][period] = 100 - (100/(1+df['RS'][period]))

## Calculate rest of Average Up, Average Down, RS, RSI
    for x in range(period+1, len(df)):
        df['Average Up'][x] = (df['Average Up'][x-1]*(period-1)+df['Up Move'][x])/period
        df['Average Down'][x] = (df['Average Down'][x-1]*(period-1)+df['Down Move'][x])/period
        df['RS'][x] = df['Average Up'][x] / df['Average Down'][x]
        df['RSI'][x] = 100 - (100/(1+df['RS'][x]))

    df = df.drop(columns=['Up Move', 'Down Move','Average Up','Average Down','RS'])
    return df

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
        date_obj = date.to_pydatetime()
        dt = date_obj.strftime("%Y-%m-%d")
        dates.append(dt)
    return dates


def plot_price_volume(period,interval,data,rsi_period):
    stock_data = data.history(period = period, interval = interval)

    st.subheader('Close Price')
    st.line_chart(stock_data.Close)
    basic_data(stock_ticker)

    st.subheader('Volume')
    st.line_chart(stock_data.Volume)


    stock_data = RSI_function(stock_data,rsi_period)
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
    download_interface(stock_data,stock_ticker)
    st.write(stock_data)




def plot_price_volume_2(start, end, interval, data,rsi_period):
    stock_data = data.history(start = start, end = end, interval = interval)

    st.subheader('Close Price')
    st.line_chart(stock_data.Close)
    basic_data(stock_ticker)


    st.subheader('Volume')
    st.line_chart(stock_data.Volume)

    stock_data = RSI_function(stock_data,rsi_period)
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
    st.write(stock_data)
    download_interface(stock_data,stock_ticker)



def plot_price_volume_3(start,interval,data,rsi_period):
    stock_data = data.history(start = start, interval = interval)

    st.subheader('Close Price')
    st.line_chart(stock_data.Close)

    st.subheader("Basic Data")
    basic_data(stock_ticker)

    st.subheader('Volume')
    st.line_chart(stock_data.Volume)

    stock_data = RSI_function(stock_data,rsi_period)
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
    st.subheader("Stock Data")
    st.write(stock_data)
    download_interface(stock_data,stock_ticker)



def display_date(data,start,end,rsi_period):
    interval_options = ['1d', '5d', '1wk', '1mo']
    interval = st.sidebar.selectbox("interval", interval_options)
    if(end == ''):
        plot_price_volume_3(start,interval,data,rsi_period)
    else:
        plot_price_volume_2(start,end,interval,data,rsi_period)





def display_period(period_view,data,rsi_period):

    if(period_view==True):
        period_options = ['1y', '3mo', '6mo','ytd','2y', '5y', '10y', 'max']
        period = st.sidebar.selectbox("period", period_options)

        if (period == '3mo' or '6mo' or '1y' or 'ytd' or '2y' or '5y' or '10y' or 'max'):
            interval_options = ['1d', '5d', '1wk', '1mo', '3mo']
            interval = st.sidebar.selectbox("interval", interval_options)
            plot_price_volume(period,interval,data,rsi_period)

    elif (period_view==False):
        start = st.sidebar.text_input("start", datetime.strftime(datetime.today()-timedelta(365),"%Y-%m-%d"))
        end = st.sidebar.text_input("end", datetime.strftime(datetime.today(),"%Y-%m-%d"))
        display_date(data,start,end,rsi_period)




def backtest(options,stock_bt,ticker,execution_type,sell_execution_type):
    cerebro = bt.Cerebro()
    sell_percentage = 0
    if sell_execution_type == "Customise Sell":
        sell_options = ["Current Day Close Price","Next Day Open Price"]
        sell_execution_type = st.sidebar.selectbox("Sell Execution Type", sell_options)
        sell_percentage = st.sidebar.text_input("Sell Percentage %",50)
        sell_percentage = float(sell_percentage)/100

    RSI_Period = st.sidebar.text_input("RSI Period", 14)
    backtest_start = st.sidebar.text_input("Start Period",datetime.strftime(datetime.today()-timedelta(365),"%Y-%m-%d"))
    backtest_end  = st.sidebar.text_input("End Period",datetime.strftime(datetime.today(),"%Y-%m-%d"))
    buy_n_hold  = st.sidebar.checkbox("Buy and Hold")
    trade_size = st.sidebar.text_input("Lot Size for 1 transaction", 100)
    initial_amount = st.sidebar.text_input("Initial Cash",100000)

    if (options=='SMA'):
        SMA_strategy(backtest_start, backtest_end,stock_bt,cerebro, initial_amount,trade_size,ticker,buy_n_hold,RSI_Period,execution_type,sell_execution_type,sell_percentage)

    if(options == 'Simple RSI'):
        RSI_strategy(backtest_start, backtest_end,stock_bt,cerebro, initial_amount,trade_size,ticker,buy_n_hold,RSI_Period,execution_type,sell_execution_type,sell_percentage)


#execution = ["Day Close", "Next Day Open"]



############# Strategies  ######################
def SMA_strategy(start, end, stock_bt,cerebro,initial_amt,trade_size, ticker, buy_n_hold,RSI_Period,execution_type,sell_execution_type, sell_percentage,):
    cross_over_fast =  st.sidebar.text_input("Average Cross Over Days",5)
    cross_over_slow = st.sidebar.text_input("Average Days",10)

    class Cross_MA(bt.Strategy):

        def __init__(self):
            slow = bt.ind.SMA(period = int(cross_over_slow))
            fast = bt.ind.SMA(period = int(cross_over_fast))
            self.crossver = bt.ind.CrossOver(fast,slow)
            self.curr =0

        def next(self):
            if self.crossver>0:
                if execution_type == "Current Day Close Price":
                    self.buy(exectype=bt.Order.Close)
                    self.curr += int(trade_size)
                else:
                    self.buy(exectype=bt.Order.Market)
                    self.curr += int(trade_size)

            if self.position:
                if self.crossver <0:
                    if buy_n_hold == False:
                        if sell_execution_type == "Sell All at Next Open":
                            self.close()
                        if sell_execution_type == "Current Day Close Price":
                            if self.curr <= 10:
                                sell_size = self.curr
                            else:
                                sell_size = self.curr * float(sell_percentage)
                            self.sell(exectype = bt.Order.Close, size=int(sell_size))
                            self.curr += -int(sell_size)

                        if sell_execution_type == "Next Day Open Price":
                            if self.curr <= 10:
                                sell_size = self.curr
                            else:
                                sell_size = self.curr * float(sell_percentage)
                            self.sell(exectype = bt.Order.Market, size=int(sell_size))
                            self.curr += -int(sell_size)
                    else:
                        self.sell(size=0)

    start = start
    end = end
    start_date = datetime.strptime(start,'%Y-%m-%d')
    end_date = datetime.strptime(end,'%Y-%m-%d')
    data = bt.feeds.YahooFinanceData(dataname=ticker,fromdate = start_date, todate = end_date+timedelta(1))
    back = cerebro_run(cerebro,data,Cross_MA,initial_amt, trade_size)
    SMA_Visualisation(back,start,end,stock_bt,cerebro,initial_amt,RSI_Period)


def SMA_Visualisation(back,start,end,stock_bt,cerebro,initial_amt,RSI_Period):
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

    pnl_list = [int(initial_amt)+cashDel[0]]

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
    fig = pt.figure(figsize=(8, 5))
    st.subheader("PNL of BackTest")
    pt.plot(final_df.final_pnl)
    st.pyplot(fig)

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


    transaction_data = {'Date':date_column ,'Price': Price,'Quantity':Qty,'trade_pnl': Cash_change}
    df_transaction = pd.DataFrame(data=transaction_data)



    RSI_df = RSI_function(stock_data,RSI_Period)

    RSI_df = pd.DataFrame(RSI_df)

    date_column = []
    RSI_column = []
    for date in RSI_df['Date']:
        date = datetime.strftime(date,"%Y-%m-%d")
        date_column.append(date)

    for rsi in RSI_df['RSI']:
        if str(rsi) != "nan":
            RSI_column.append(int(rsi))
        else:
             RSI_column.append(rsi)

    RSI_df['Date'] = date_column
    RSI_df['RSI'] = RSI_column


    df_transaction = pd.DataFrame(data=transaction_data)

    df_transaction = df_transaction.merge(RSI_df, left_on='Date', right_on='Date')


    df_transaction = df_transaction[['Date','Price','Quantity','trade_pnl','Open','High','Low','Close','Volume','RSI']]

    st.subheader('Trade Records')
    st.write(df_transaction)

    st.write("Buy : " + str(buy))
    st.write("Sell : " + str(sell))
    stock_close = stock_bt.history(period = "5d", interval = "1d")
    pnl = qty*stock_close.Close[-1] + net
    pnl = round(pnl, 2)
    st.write('Final Pnl : ' + str(pnl))
    pnl = cerebro.broker.getvalue()
    pnl_val = st.sidebar.text_input("PNL of Strategy",round(pnl,2))
    returns_pct = ((pnl/int(initial_amt))-1) * 100
    returns_val = st.sidebar.text_input("Returns of Strategy %","%.2f" %returns_pct)





def RSI_strategy(start, end, stock_bt,cerebro,initial_amt, trade_size, ticker,buy_n_hold,RSI_Period,execution_type,sell_execution_type,sell_percentage):

    RSI_Entry = st.sidebar.text_input("RSI Entry Number",30)
    RSI_Exit = st.sidebar.text_input("RSI Exit Number",70)


    class RSIStrategy(bt.Strategy):

        def __init__(self):
            self.rsi = bt.indicators.RSI_SMA(self.data.close, period= int(RSI_Period))
            self.curr =0

        def next(self):

            if self.rsi < int(RSI_Entry):
                if execution_type == "Current Day Close Price":
                    self.buy(exectype=bt.Order.Close)
                    self.curr += int(trade_size)
                else:
                    self.buy(exectype=bt.Order.Market)
                    self.curr += int(trade_size)
            else:
                if self.position:
                    if self.rsi > int(RSI_Exit):
                        if buy_n_hold == False:
                            if sell_execution_type == "Sell All at Next Open":
                                 self.close()
                            if sell_execution_type == "Current Day Close Price":
                                if self.curr <= 10:
                                    sell_size = self.curr
                                else:
                                    sell_size = self.curr * float(sell_percentage)
                                self.sell(exectype = bt.Order.Close, size=int(sell_size))
                                self.curr += -int(sell_size)
                            if sell_execution_type == "Next Day Open Price":

                                if self.curr <= 10:
                                    sell_size = self.curr
                                else:
                                    sell_size = self.curr * float(sell_percentage)
                                self.sell(exectype = bt.Order.Market, size=int(sell_size))
                                self.curr += -int(sell_size)
                        else:
                            self.sell(size=0)
    start = start
    end = end
    start_date = datetime.strptime(start,'%Y-%m-%d')
    end_date = datetime.strptime(end,'%Y-%m-%d')
    data = bt.feeds.YahooFinanceData(dataname=ticker, fromdate = start_date, todate = end_date)
    back = cerebro_run(cerebro, data, RSIStrategy, initial_amt, trade_size)
    RSI_Visualisation(back,start,end,stock_bt,cerebro,initial_amt,RSI_Period)




def RSI_Visualisation(back,start,end,stock_bt,cerebro,initial_amt,RSI_Period):
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

    pnl_list = [int(initial_amt)+cashDel[0]]

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
    stock_data = stock_data.reset_index()



    data={'date':stock_data ['Date'],'close': stock_data ['Close']}

    final_data =pd.DataFrame(data=data)
    final_df = df.merge(final_data,"inner",'date')

    final_df['final_pnl'] = final_df['Stock_Quantity'] * final_df['close'] + final_df['trade_pnl']
    final_df = final_df.set_index('date')
    fig = pt.figure(figsize=(8, 5))
    st.write("PNL of BackTest")
    pt.plot(final_df.final_pnl)
    st.pyplot(fig)


    sharpe = back[0].analyzers.sharpe.get_analysis()
    sharpe_val = st.sidebar.text_input("Sharpe of Strategy","%.2f" %list(sharpe.values())[0])


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


    transaction_data = {'Date':date_column ,'Price': Price,'Quantity':Qty,'trade_pnl': Cash_change}


    RSI_df = RSI_function(stock_data,RSI_Period)

    RSI_df_Graph = stock_data.set_index('Date')

    RSI_df = pd.DataFrame(RSI_df)


    date_column = []
    RSI_column = []
    for date in RSI_df['Date']:
        date = datetime.strftime(date,"%Y-%m-%d")
        date_column.append(date)

    for rsi in RSI_df['RSI']:
        if str(rsi) != "nan":
            RSI_column.append(int(rsi))
        else:
             RSI_column.append(rsi)

    RSI_df['Date'] = date_column
    RSI_df['RSI'] = RSI_column
    RSI_df['RSI'] = RSI_df['RSI'].shift(1)

    df_transaction = pd.DataFrame(data=transaction_data)

    df_transaction = df_transaction.merge(RSI_df, left_on='Date', right_on='Date')

    st.subheader('Trade Records')
    df_transaction = df_transaction[['Date','Price','Quantity','trade_pnl','RSI','Open','High','Low','Close','Volume']]
    st.write(df_transaction)

    st.write("Buy : " + str(buy))
    st.write("Sell : " + str(sell))
    stock_close = stock_bt.history(period = "5d", interval = "1d")
    pnl = qty*stock_close.Close[-1] + net
    pnl = round(pnl, 2)

    #pnl = cerebro.broker.getvalue()

    pnl_val = st.sidebar.text_input("PNL of Strategy", str(int(pnl)+ int(initial_amt)))
    returns_pct = ((int(pnl_val)/int(initial_amt))-1) * 100
    returns_val = st.sidebar.text_input("Returns of Strategy %","%.2f" %returns_pct)
    st.write('Final Pnl : ' + str(pnl))


    st.subheader("RSI Graph")
    st.line_chart(RSI_df_Graph.RSI)







def FinanceStatement_Setup(stock_ticker):
    yhfin = YahooFinancials(stock_ticker)
    annual_fin = yhfin.get_financial_stmts('annual', 'income')
    date = []
    revenue = []
    cogs = []
    gross_profit = []
    gross_profit_margin = []
    ebit = []
    ebit_margin = []
    int_exp = []
    income = []
    income_margin = []
    int_coverage = []
    million = 1000000
    for x in annual_fin['incomeStatementHistory'][stock_ticker]:
        data = list(x.items())
        date.append(data[0][0])

        rev_data = data[0][1]['totalRevenue']
        if rev_data is None or rev_data ==0:
            revenue.append(np.nan)
        else:
            revenue.append(rev_data/million)

        cogs_data = data[0][1]['costOfRevenue']
        if cogs_data is None or cogs_data ==0:
            cogs.append(np.nan)
        else:
            cogs.append(cogs_data/million)

        gross_profit_data = data[0][1]['grossProfit']

        if gross_profit_data is None or gross_profit_data ==0:
            gross_profit.append(np.nan)
            gross_profit_margin.append(np.nan)
        else:
            gross_profit.append(gross_profit_data/million)
            gp_margin_data = 100*data[0][1]['grossProfit']/data[0][1]['totalRevenue']
            if gp_margin_data < 0:
                gp_margin_data = np.nan
            gross_profit_margin.append(gp_margin_data)

        ebit_data = data[0][1]['ebit']
        if ebit_data is None or ebit_data ==0:
            ebit.append(np.nan)
            ebit_margin.append(np.nan)
        else:
            ebit.append(ebit_data/million)
            ebit_margin_data = 100*data[0][1]['ebit']/data[0][1]['totalRevenue']
            if ebit_margin_data < 0:
                ebit_margin_data = np.nan
            ebit_margin.append(ebit_margin_data)

        int_exp_data = data[0][1]['interestExpense']
        if int_exp_data is None or int_exp_data==0:
            int_exp_data = np.nan
            int_exp.append(int_exp_data)
            int_coverage.append(int_exp_data)
        else:
            int_exp.append(int_exp_data/million)
            if ebit_data < 0:
                int_coverage.append(np.nan)
            else:
                int_coverage.append(data[0][1]['ebit']/int_exp_data)

        income_data = data[0][1]['netIncome']
        if income_data is None or income_data==0:
            income.append(np.nan)
            income_margin.append(np.nan)
        else:
            income.append(income_data/million)
            income_margin_data = 100*data[0][1]['netIncome']/data[0][1]['totalRevenue']
            if income_margin_data < 0:
                income_margin_data = np.nan
            income_margin.append(income_margin_data)


    annual_cf = yhfin.get_financial_stmts('annual', 'cash')
    da = []
    capex = []
    divd = []
    CFO = []
    CFF = []
    CFI = []
    ebitda = []
    ebitda_margin = []
    FCF = []
    net_debt = []
    FCFE = []
    stock_issuance = []

    for x in annual_cf['cashflowStatementHistory'][stock_ticker]:
        data = list(x.items())

        try:
            da.append(data[0][1]['depreciation']/million)
        except KeyError:
            da.append(np.nan)

        try:
            capex.append(data[0][1]['capitalExpenditures']/million)
        except KeyError:
            capex.append(np.nan)

        try:
            divd.append(data[0][1]['dividendsPaid']/million)
        except KeyError:
            divd.append(np.nan)

        try:
            CFO.append(data[0][1]['totalCashFromOperatingActivities']/million)
        except KeyError:
            CFO.append(np.nan)
        try:
            CFF.append(data[0][1]['totalCashFromFinancingActivities']/million)
        except KeyError:
            CFF.append(np.nan)
        try:
            CFI.append(data[0][1]['totalCashflowsFromInvestingActivities']/million)
        except KeyError:
            CFI.append(np.nan)

        try:
            stock_issuance.append(data[0][1]['issuanceOfStock']/million +
                                  data[0][1]['repurchaseOfStock']/million)
        except KeyError:
            stock_issuance.append(np.nan)

        try:
            net_debt.append(data[0][1]['netBorrowings']/million)
        except KeyError:
            net_debt.append(np.nan)

    ebitda = np.add(ebit,da)
    ebitda_filtered = []
    for ebitda_value in ebitda:
        if ebitda_value < 0 or ebitda_value == np.nan:
            ebitda_filtered.append(np.nan)
        else:
            ebitda_filtered.append(ebitda_value)
    ebitda_margin = map(lambda x : x*100, np.divide(ebitda_filtered,revenue))
    FCF = np.add(CFO,capex)
    FCFE = np.add(FCF,net_debt)
    df_cash = {'CF Operations': CFO,'CF Finance': CFF,
               'CF Investment': CFI,'CAPEX': capex,
               'Free Cash Flow': FCF,
               'Dividend': divd,
               'Stock Issue': stock_issuance,
               'Free Cash Flow to Equity': FCFE
    }


    index = [date]

    df_income = {'Revenue': revenue,'COGS': np.multiply(cogs,-1), 'Gross Profit': gross_profit,'EBITDA': ebitda,
          'D&A': np.multiply(da,-1),'EBIT': ebit, 'Interest Exp': int_exp,
          'Profit': income
    }

    df_income = pd.DataFrame(data=df_income,index=index)

    df_ratios = { 'GP Margin (%)': gross_profit_margin,
                 'EBITDA Margin (%)': ebitda_margin,
                 'EBIT Margin (%)': ebit_margin,
                 'Int Coverage (x)': np.multiply(int_coverage,-1),
                 'Profit Margin (%)': income_margin
    }

    df_ratios = pd.DataFrame(data=df_ratios,index=index)

    df_cash = pd.DataFrame(data=df_cash,index=index)

    st.subheader("Income Data in Millions")
    st.table(df_income.transpose().style.format("{:.2f}"))

    st.subheader("Key Ratios")
    st.table(df_ratios.transpose().style.format("{:.2f}"))

    st.subheader("Cash Flow Data in Millions")
    st.table(df_cash.transpose().style.format("{:.2f}"))




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

    st.title('Trading DashBoard')
    url = 'https://g.foolcdn.com/editorial/images/602904/why-tesla-stock-is-up-today.jpg'
    response = requests.get(url)
    img = Image.open(BytesIO(response. content))
    st.image(img, use_column_width=True)
    st.sidebar.header('Enter your Inputs')
    stock_ticker = st.sidebar.text_input("Stock Ticker", "TSM")
    stock = yf.Ticker(stock_ticker)
    period_view = st.sidebar.checkbox('period')
    mv_slow = st.sidebar.text_input("Moving Average (Slow)",5)
    mv_fast = st.sidebar.text_input("Moving Average (Fast)",25)
    rsi_period = st.sidebar.text_input("RSI (Period)",14)
    display_period(period_view,stock,rsi_period)

    FinanceStatement_Setup(stock_ticker.upper())

    #BackTest Component
    st.sidebar.header('Backtest Strategy')
    ticker = st.sidebar.text_input("Ticker","TSM")
    stock_bt = yf.Ticker(ticker)
    stock_close = stock_bt.history(period = "5d", interval = "1d")
    current_price = st.sidebar.text_input( "Current Price" ,"%.2f" %stock_close.Close[-1])
    backtest_options = ['SMA','RSI']
    backtest_options = st.sidebar.selectbox("Strategies", backtest_options)
    execution = ["Current Day Close Price", "Next Day Open Price"]
    sell_execution = ["Sell All at Next Open", "Customise Sell"]
    execution_type = st.sidebar.selectbox("Buy Execution Type",execution)
    sell_execution_type = st.sidebar.selectbox("Sell Execution Type",sell_execution)
    st.title("BackTest")
    backtest(backtest_options, stock_bt, ticker,execution_type,sell_execution_type)
