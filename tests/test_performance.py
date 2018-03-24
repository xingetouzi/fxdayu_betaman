import pandas as pd
from fxdayu_betaman.calculator.analysis import Performance
from fxdayu_betaman.loader import FileLoader
import pymongo

if __name__ == "__main__":
    strategy_id = 2
    pd.set_option("display.width", 160)
    client = pymongo.MongoClient("mongodb://192.168.0.101,192.168.0.102")
    c = client.get_database("signal").get_collection("trade")
    trades = c.find({"strategy": strategy_id}, {"_id": 0})
    tdf = pd.DataFrame(list(trades))
    print(tdf)
    # if len(tdf):
    #     tdf["datetime"] = tdf["trading_datetime"]
    # calculator = Performance(tdf, 1000000)
    # # daily_returns = calculator.daily_returns
    # # print(daily_returns)
    # print(calculator.entry)
    # returns = daily_returns
    # b_r = calculator.benchmark_rets()
