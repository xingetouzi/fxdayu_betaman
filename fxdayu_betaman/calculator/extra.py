from datetime import timedelta

import pandas as pd

from fxdayu_betaman.calculator import BaseCalculator

EXCHANGE_MAP = {
    "XSHE": "SZSE",
    "XSHG": "SSE"
}


class GuoJinCalculator(BaseCalculator):
    datetime_format = "%Y%m%dT%H%M%S"

    def __init__(self, *args, name="untitled", **kwargs):
        self._name = name
        super(GuoJinCalculator, self).__init__(*args, **kwargs)

    def _get_market_from_order_book_id(self, order_book_id):
        return EXCHANGE_MAP[order_book_id.split(".")[-1]]

    def _get_code_from_order_book_id(self, order_book_id):
        return order_book_id.split(".")[0]

    def _strftime(self, dt):
        return dt.strftime(self.datetime_format)

    def get_order(self):
        df = self._trades

    def get_position(self):
        df = pd.concat([self._trades, self.position_info_detail], axis=1).sort_values(["order_book_id", "datetime"])
        temp = pd.DataFrame(index=df.index)
        temp["Market"] = df["order_book_id"].apply(self._get_market_from_order_book_id)
        temp["Code"] = df["order_book_id"].apply(self._get_code_from_order_book_id)
        temp["StartTime"] = df["datetime"]
        temp["ExpireTime"] = df["datetime"] + timedelta(days=1)
        temp["Position"] = df["cumsum_quantity"]
        temp["StartTime"] = temp["StartTime"].apply(self._strftime)
        temp["ExpireTime"] = temp["ExpireTime"].apply(self._strftime)
        return temp

    def get_risk_exp(self):
        series = self.net.resample("D").last().dropna() * self._account["total_value"]
        series.name = "RiskExp"
        series.index.name = "Date"
        df = series.reset_index()
        df["Date"] = df["Date"].apply(lambda x: x.strftime("%Y/%m/%d 10:00:00"))
        return df

    def save_risk_exp_csv(self, path=None):
        if path is None:
            path = "{}-RiskExp.csv".format(self._name)
        self.get_risk_exp().to_csv(path, index=False)

    def save_position_csv(self, path=None):
        if path is None:
            path = "{}-Positions.csv".format(self._name)
        self.get_position().to_csv(path, index=False)
