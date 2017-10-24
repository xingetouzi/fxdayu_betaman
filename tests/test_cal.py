import pandas as pd
from fxdayu_betaman.calculator import BaseCalculator
from fxdayu_betaman.loader import FileLoader
import pyfolio as pf

if __name__ == "__main__":
    pd.set_option("display.width", 160)
    df = FileLoader("test.xlsx").load()
    # print(df)
    calculator = BaseCalculator(df, 1000000)
    print(calculator.daily_market_value)
    #start = pd.Timestamp("2013-03-12 15:00:00")
    #end = pd.Timestamp("2013-03-15 15:00:00")
    # print(calculator.position_info_detail_by_time.loc[start:end])
    # print(calculator.market_value_by_time)
    # print(calculator.account_value_by_time)
    # print(calculator.portfolio_value_by_time)
    # print(calculator.cash)
    # print(calculator.net)
    # print(calculator.pnl_by_time)
    # print(calculator.float_pnl_by_time)
    # print(calculator.position)
    # print(calculator.average_price)
    # print(calculator.market_data)
