import datetime
import functools
import weakref
from enum import Enum

import pandas as pd
import numpy as np
from dateutil.parser import parse
from fxdayu_data import DataAPI

try:
    from functools import lru_cache, partial
except ImportError:
    from fastcache import lru_cache


def memorized_method(*lru_args, **lru_kwargs):
    def decorator(func):
        @functools.wraps(func)
        def wrapped_func(self, *args, **kwargs):
            # We're storing the wrapped method inside the instance. If we had
            # a strong reference to self the instance would never die.
            self_weak = weakref.ref(self)

            @functools.wraps(func)
            @lru_cache(*lru_args, **lru_kwargs)
            def cached_method(*args, **kwargs):
                return func(self_weak(), *args, **kwargs)

            return cached_method(*args, **kwargs)

        return wrapped_func

    return decorator


def side2sign(s, keepdim=False):
    series = (pd.Categorical(s, categories=["SELL", "BUY"]).codes << 1) - 1
    return np.squeeze(series) if not keepdim else series


class DIRECTION(Enum):
    NONE = "无方向"
    LONG = "多"
    SHORT = "空"
    UNKNOWN = "未知"


def sign2direction(x):
    return DIRECTION.LONG.value if x >= 0 else DIRECTION.SHORT.value


class BaseCalculator(object):
    def __init__(self, trades, accounts, extend=False, freq="D", dailySumTime=15):
        self._freq = freq
        self._dailySumTime = dailySumTime
        self._trades = trades
        self._extend = extend
        if isinstance(accounts, (float, int)):
            self._accounts = self._init_default_account(accounts)
        else:
            self._accounts = accounts
        self._account = self._accounts["STOCK"]
        self._normalize_data()

    def _normalize_data(self):
        self._trades["datetime"] = self._trades["datetime"].astype("datetime64[ns]")
        self._trades = self._trades.sort_values(by="datetime")
        self._trades.index = np.arange(len(self._trades))
        self._trades.index.name = "order_id"
        self._trades = self._trades.drop("order_id", axis=1, errors="ignore")

    def _init_default_account(self, init_value):
        return {
            "STOCK":
                {
                    "total_value": init_value
                }
        }

    @memorized_method()
    def get_trade_groupby(self, by=None):
        return self._trades.groupby(by)

    @staticmethod
    def _cumsum_position(df):
        series = (df["last_quantity"] * side2sign(df["side"])).cumsum()
        series.name = "cumsum_quantity"
        return series

    @staticmethod
    def _cal_entry(df):
        p = df["cumsum_quantity"]
        series = (p - p.shift(1).fillna(0)) > 0
        series.name = "entry"
        return series

    @property
    @memorized_method()
    def entry(self):
        return pd.concat([self.position["datetime"],
                          pd.DataFrame(self.position).groupby(level=0, group_keys=False).apply(self._cal_entry)],axis=1)


    @memorized_method()
    # TODO 未考虑反向开仓,未考虑分红派息导致的仓位和持仓成本变化
    def _position_info_detail(self):
        detail = []
        for ticker, trades in self._trades.groupby("order_book_id"):
            temp = pd.DataFrame(index=trades.index)
            temp["cumsum_quantity"] = (trades["last_quantity"] * side2sign(trades["side"])).cumsum()
            temp["position_side"] = (temp["cumsum_quantity"] >= 0).apply(sign2direction)
            market_values = []
            position_avx_prices = []
            profits = []
            position_avx_price = 0
            last_volume = 0
            for _, direction, volume in zip(trades.iterrows(), temp["cumsum_quantity"].values,
                                            temp["cumsum_quantity"].values):

                _, order = _
                point_value = getattr(order, "point_value", 1)
                # 计算证券市值、持仓均价、每笔收益
                if side2sign(order["side"]) >= 0: # 加仓，更新持仓均价 TODO 分红派息也要更新持仓均价
                    position_avx_price = (position_avx_price * last_volume + order["last_quantity"] * order["last_price"])/volume
                    profit = np.nan
                else: # 减仓、平仓，计算平仓收益
                    profit = (order["last_quantity"] * (order["last_price"] - position_avx_price))*point_value
                    if volume == 0: # 平仓，将持仓均价调为0
                        position_avx_price = 0
                last_volume = volume
                market_values.append(order["last_price"] * volume) # 更新市值
                position_avx_prices.append(position_avx_price) # 更新持仓均价
                profits.append(profit) # 更新平仓收益

            temp["position_id"] = (temp["cumsum_quantity"] == 0).cumsum().shift(1).fillna(0) + 1  # TODO 未考虑反向开仓
            temp["position_id"] = temp["position_id"].astype(int)
            temp["market_value"] = market_values
            temp["avg_price"] = position_avx_prices
            temp["profits"] = profits
            temp["pnl"] = temp["profits"].fillna(0).cumsum() - trades["transaction_cost"].cumsum()
            detail.append(temp)
        return pd.concat(detail, axis=0).sort_index()

    @staticmethod
    def _concat_market_data(df, index=None, market_data=None):
        result = pd.concat([df.drop("order_book_id", axis=1).set_index("datetime"), market_data.loc[df.name]], axis=1)
        return result.reindex(index).fillna(method="ffill").dropna()

    @memorized_method()
    def _position_info_detail_by_time(self):
        details = pd.concat([self._trades, self._position_info_detail()], axis=1)
        market_data = self.market_data
        index = sorted(pd.concat([details["datetime"], market_data.reset_index()["datetime"]]).unique())
        fields = ["cumsum_quantity", "pnl", "market_value", "position_side", "avg_price", "order_book_id", "datetime"]
        df = details[fields].reset_index().groupby("order_book_id").apply(partial(self._concat_market_data,
                                                                                  index=index,
                                                                                  market_data=market_data))
        groupby = df.reset_index().groupby(["datetime", "order_book_id"]).last()
        groupby["float_pnl"] = groupby["pnl"] + (groupby["close"] - groupby["avg_price"]) * groupby["cumsum_quantity"]
        return groupby

    @memorized_method()
    def _position_info_detail_by_symbol(self):
        return pd.concat([self._trades[["order_book_id", "datetime"]], self._position_info_detail()], axis=1) \
            .reset_index().set_index(["order_book_id", "order_id"]).sort_index()

    @property
    def position_info_detail(self):
        return self._position_info_detail()

    @property
    def position_info_detail_by_symbol(self):
        return self._position_info_detail_by_symbol()

    @property
    def position_info_detail_by_time(self):
        return self._position_info_detail_by_time()

    @property
    def position(self):
        return self.position_info_detail_by_symbol[["datetime","cumsum_quantity"]]

    @property
    def market_value(self):
        return self.position_info_detail_by_symbol[["datetime","market_value"]]

    @property
    def pnl(self):
        return self.position_info_detail_by_symbol[["datetime","pnl"]]

    @property
    def profits(self):
        return self.position_info_detail_by_symbol[["datetime","profits"]]

    @property
    def average_price(self):
        return self.position_info_detail_by_symbol[["datetime","avg_price"]]

    @property
    def pending_position(self):
        df = self.position.groupby(level=0).last()
        return df[df != 0].dropna()

    def _by_order2time(self, series):
        pass

    @property
    def market_value_by_time(self):
        return self.position_info_detail_by_time["market_value"]

    @property
    def daily_market_value(self):
        market_value_by_time = self.market_value_by_time.unstack()
        series = market_value_by_time[market_value_by_time.index.hour==self._dailySumTime].stack()
        series.name = "daily_market_value"
        return series

    @property
    def portfolio_value_by_time(self):
        series = self.market_value_by_time.groupby(level=0).sum()
        series.name = "portfolio_value"
        return series

    @property
    def position_by_time(self):
        series = self.position_info_detail_by_time["cumsum_quantity"]
        return series.loc[series != 0]

    @property
    def pnl_by_time(self):
        df = self.position_info_detail_by_time
        return df["pnl"].groupby(level=0).sum()

    @property
    def float_pnl_by_time(self):
        df = self.position_info_detail_by_time
        return df["float_pnl"].groupby(level=0).sum()

    @property
    def account_value_by_time(self):
        series = self.float_pnl_by_time + self._account["total_value"]
        series.name = "account_value"
        return series

    @property
    def daily_account_value(self):
        account_value_by_time = self.account_value_by_time
        series = account_value_by_time[account_value_by_time.index.hour==self._dailySumTime]
        series.name = "daily_account_value"
        return series

    @property
    def net(self):
        series = self.account_value_by_time/self._account["total_value"]
        series.name = "net"
        return series

    @property
    def cash(self):
        series = self.account_value_by_time - self.portfolio_value_by_time
        series.name = "cash"
        return series

    @property
    def returns(self):
        series = self.account_value_by_time.pct_change()
        series.name = "returns"
        return series

    def daily_returns(self):
        series = self.daily_account_value.pct_change()
        series.name = "daily_returns"
        return series

    @property
    @memorized_method()
    def date_range(self):
        dt = self._trades["datetime"]
        if "datetime" in str(dt.dtype):
            start = dt.iloc[0].to_pydatetime()
        elif dt.dtype == np.dtype.str:
            start = parse(self._trades["datetime"].iloc[0]).date()
        if len(self.pending_position) and self._extend:
            end = datetime.datetime.now()
        else:
            dt = self._trades["datetime"]
            if "datetime" in str(dt.dtype):
                end = dt.iloc[-1].to_pydatetime()
            elif dt.dtype == np.dtype.str:
                end = parse(self._trades["datetime"].iloc[-1]).date()
        end += datetime.timedelta(days=1)
        return start, end

    @property
    @memorized_method()
    def universe(self):
        return self._trades["order_book_id"].unique()

    @property
    @memorized_method()
    def market_data(self):
        start, end = self.date_range
        df = DataAPI.candle(self.universe, fields=["close"], start=start, end=end,
                            freq=self._freq).loc[:, :, "close"].T

        df.index.name = "order_book_id"
        series = df.stack()
        series.name = "close"
        return series
