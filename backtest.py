import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import backtrader as bt
import backtrader.analyzers as btanalysis
from datetime import datetime, timedelta
import matplotlib.pyplot as pt
import RSI

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



def backtest(options,stock_bt,ticker,execution_type,sell_execution_type):
    cerebro = bt.Cerebro()
    sell_percentage = 0
    if sell_execution_type == "Customise Sell":
        sell_options = ["Current Day Close Price","Next Day Open Price"]
        sell_execution_type = st.sidebar.selectbox("Sell Execution Type", sell_options)
        sell_percentage = st.sidebar.text_input("Sell Percentage %",50)
        sell_percentage = float(sell_percentage)/100

    RSI_Period = st.sidebar.text_input("RSI Period", 14)
    backtest_start = st.sidebar.text_input("Start Period",datetime.strftime(datetime.today()-timedelta(365*4),"%Y-%m-%d"))
    backtest_end  = st.sidebar.text_input("End Period",datetime.strftime(datetime.today(),"%Y-%m-%d"))
    buy_n_hold  = st.sidebar.checkbox("Buy and Hold")
    trade_size = st.sidebar.text_input("Lot Size for 1 transaction", 100)
    initial_amount = st.sidebar.text_input("Initial Cash",100000)

    if (options=='SMA'):
        SMA_strategy(backtest_start, backtest_end,stock_bt,cerebro, initial_amount,trade_size,ticker,buy_n_hold,RSI_Period,execution_type,sell_execution_type,sell_percentage)

    if(options == 'RSI'):
        RSI_strategy(backtest_start, backtest_end,stock_bt,cerebro, initial_amount,trade_size,ticker,buy_n_hold,RSI_Period,execution_type,sell_execution_type,sell_percentage)



############# Strategies  ######################
def SMA_strategy(start, end, stock_bt,cerebro,initial_amt,trade_size, ticker, buy_n_hold, RSI_Period, execution_type, sell_execution_type, sell_percentage,):
    cross_over_fast =  st.sidebar.text_input("Average Cross Over Days",5)
    cross_over_slow = st.sidebar.text_input("Average Days",10)

    class Cross_MA(bt.Strategy):

        def __init__(self):
            slow = bt.ind.SMA(period = int(cross_over_slow))
            fast = bt.ind.SMA(period = int(cross_over_fast))
            self.crossver = bt.ind.CrossOver(fast,slow)
            self.curr =0

        def next(self):
            if self.crossver > 0:
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


    # data = bt.feeds.YahooFinanceData(dataname=ticker,
    #                              fromdate=start_date,
    #                              todate=end_date)


    dataframe = yf.download(ticker, start=start_date, end=end_date)

    data = bt.feeds.PandasData(dataname=dataframe, name=ticker)



    back = cerebro_run(cerebro, data, Cross_MA, initial_amt, trade_size)
    SMA_Visualisation(back,start,end,stock_bt,cerebro,initial_amt,RSI_Period)


def SMA_Visualisation(back,start,end,stock_bt,cerebro,initial_amt,RSI_Period):
    cashDelta_list = []
    date_list = []
    stock_qty = []
    print(back[0].analyzers.transaction.get_analysis().items())
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

    final_data['date'] = pd.to_datetime(final_data["date"].dt.strftime('%Y-%m-%d'))

    final_df = df.merge(final_data,"inner",'date')

    final_df['final_pnl'] = final_df['Stock_Quantity'] * final_df['close'] + final_df['trade_pnl']
    final_df = final_df.set_index('date')
    fig = pt.figure(figsize=(8, 5))
    st.subheader("Proft & Loss of Trade Simulation")
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

    RSI_df = RSI.RSI_function(stock_data,RSI_Period)

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


   # data = bt.feeds.YahooFinanceData(dataname=ticker,
   #                              fromdate=start_date,
   #                              todate=end_date)
    

 

    data = bt.feeds.PandasData(dataname=yf.download(ticker, start_date, end_date, auto_adjust=True))
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

    final_data['date'] = pd.to_datetime(final_data["date"].dt.strftime('%Y-%m-%d'))
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


    RSI_df = RSI.RSI_function(stock_data,RSI_Period)

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





