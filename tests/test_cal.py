import pandas as pd
from fxdayu_betaman.calculator import BaseCalculator
from fxdayu_betaman.loader import FileLoader

if __name__ == "__main__":
    pd.set_option("display.width", 160)
    df = FileLoader("test.xlsx").load()
    print(df)
    calculator = BaseCalculator(df, 1000000)
    # print(calculator.position)
    # print(calculator.pending_position)
    # print(calculator.market_value)
    # print(calculator.pnl)
    # print(calculator.market_data)
    # print
    start = pd.Timestamp("2013-03-12 15:00:00")
    end = pd.Timestamp("2013-03-15 15:00:00")
    df = calculator.position_info_detail_by_time.loc[start:end]
    print(df.loc[df["cumsum_quantity"] != 0])
    print(calculator.net)
    # print(calculator.pnl_by_time)
    # print(calculator.float_pnl_by_time)
    # print(calculator.net)
    # print(calculator.entry)
    # print(calculator.average_price)
    # print(calculator.market_data)
