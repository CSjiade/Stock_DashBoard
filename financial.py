import pandas as pd
import numpy as np
import streamlit as st
from yahoofinancials import YahooFinancials
import yfinance as yf



def financeStatement_setup(stock_ticker):



    #yhfin = YahooFinancials(stock_ticker)

    yhfin = yf.Ticker(stock_ticker)


    print(yhfin.income_stmt)
    #print(yhfin.balance_sheet)
    #print(yhfin.cashflow)

    incomeData  = yhfin.income_stmt.loc[["Total Revenue", "Cost Of Revenue","Gross Profit","Selling General And Administration","Research And Development",
                                    "Operating Income","EBITDA","EBIT","Net Income"]]

    newColName = []
    for col in incomeData.columns:
        newCol = col.strftime('%Y-%m-%d')
        newColName.append(newCol)
        print(newCol)
    incomeData.columns =  newColName 

    for col in newColName:
        incomeData[col] = incomeData[col].apply(lambda x: x/1000000)

    incomeData = incomeData.apply(lambda x: x *-1 if x.name in [
              'Cost Of Revenue', 'Selling General And Administration', 'Research And Development'] else x, axis=1)
    st.subheader("Income Data in Millions")
    st.dataframe(incomeData)

    print(yhfin.balance_sheet)
