import numpy as np

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

