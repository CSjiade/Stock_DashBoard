import pandas as pd
import numpy as np
import streamlit as st
from yahoofinancials import YahooFinancials




def financeStatement_setup(stock_ticker):
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
    net_debt = []
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
