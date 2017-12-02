import os
import datetime
import functools
import weakref
from enum import Enum

import pandas as pd
import numpy as np
from dateutil.parser import parse
import pyfolio as pf
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
    def __init__(self,
                 trades,
                 accounts,
                 extend=False,
                 freq="D",
                 dailySumTime=15,
                 startSumDate=None,
                 endSumDate=None):
        self._freq = freq
        self._dailySumTime = dailySumTime
        self._startSumDate = startSumDate
        if isinstance(self._startSumDate, str):
            self._startSumDate = parse(self._startSumDate)
        self._endSumDate = endSumDate
        if isinstance(self._endSumDate, str):
            self._endSumDate = parse(self._endSumDate)
        if self._endSumDate:
            self._endSumDate = self._endSumDate.replace(hour=23, minute=59, second=59)
        self._trades = trades
        self._extend = extend
        if isinstance(accounts, (float, int)):
            self._accounts = self._init_default_account(accounts)
        else:
            self._accounts = accounts
        self._account = self._accounts["STOCK"]
        self._normalize_data()

    def _normalize_data(self):
        if len(self._trades):
            self._trades["datetime"] = self._trades["datetime"].astype("datetime64[ns]")
            self._trades = self._trades.sort_values(by="datetime")
            if self._endSumDate:
                self._trades = self._trades[self._trades["datetime"] <= self._endSumDate]
            if self._startSumDate:
                self._trades = self._trades[self._trades["datetime"] >= self._startSumDate]
            self._trades.index = np.arange(len(self._trades))
            self._trades.index.name = "order_id"
            self._trades = self._trades.drop("order_id", axis=1, errors="ignore")

    @staticmethod
    def _init_default_account(init_value):
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
                          pd.DataFrame(self.position).groupby(level=0, group_keys=False).apply(self._cal_entry)],
                         axis=1)

    @staticmethod
    def _reform_stock_trades(code, trades):
        # 给股票的交割单加上分红送股的日子
        _trades = trades.reset_index().set_index('datetime')
        _trades.index = pd.to_datetime(_trades.index)
        start_date, end_date = _trades.index[0], _trades.index[-1] + datetime.timedelta(days=1)

        _bonus = DataAPI.bonus(code).copy()
        cols = [i for i in ['split_factor', 'cash_before_tax'] if i in _bonus.columns]
        if 'cash_before_tax' in cols:
            _bonus['cash_before_tax'] = _bonus['cash_before_tax'] / _bonus['round_lot']

        _trades = _trades.append(pd.DataFrame(index=_bonus['closure_date'].dropna())).append(
            _bonus[cols].rename(columns={'split_factor': 'last_quantity', 'cash_before_tax': 'last_price'}))

        return _trades.reset_index().sort_values(['index', 'order_id']).set_index('index')[start_date: end_date]

    @staticmethod
    def _trades_analyze(trades):
        position_avg_price = 0
        holding_position = 0
        record_closure_position = 0
        _dividend_cash = 0

        for dt, order in trades.iterrows():
            # 这里的计算可能会发现问题
            point_value = getattr(order, "point_value", 1)
            if ~np.isnan(order['order_id']):

                direction = side2sign(order["side"])
                profit = order["last_quantity"] * (order["last_price"] - position_avg_price) * point_value * \
                    (-direction) if direction * holding_position < 0 else np.nan
                # if _dividend_cash:
                #     profit += _dividend_cash
                #     _dividend_cash = 0

                overall_cost = position_avg_price * holding_position + order["last_quantity"] * order["last_price"] \
                    if direction * holding_position >= 0 else 0

                holding_position += order['last_quantity'] * direction
                position_avg_price = overall_cost / holding_position if overall_cost != 0 else (
                    0 if holding_position == 0 else position_avg_price)

                market_value = order["last_price"] * holding_position

                yield order['order_id'], int(holding_position), market_value, position_avg_price,\
                      profit, order['transaction_cost'], dt
            else:
                if np.isnan(order['last_price']) and np.isnan(order['last_quantity']):
                    if holding_position > 0:
                        record_closure_position = holding_position
                    continue

                if ~np.isnan(order['last_price']):
                    position_avg_price -= order['last_price'] if position_avg_price != 0 else 0
                    if not np.allclose(record_closure_position, 0):
                        _dividend_cash = order['last_price'] * record_closure_position
                        # if np.allclose(_dividend_cash, 0):
                        #     print('++++++++++++++++++++++may have error++++++++++++++++++++++++++++++++',
                        #           order['last_price'], record_closure_position)
                        record_closure_position = 0

                if ~np.isnan(order['last_quantity']):
                    position_avg_price = position_avg_price / order['last_quantity']
                    holding_position = holding_position * order['last_quantity']

                holding_position = int(holding_position)
                if _dividend_cash:
                    yield -1, holding_position, holding_position*position_avg_price, position_avg_price, \
                          _dividend_cash, 0, dt
                    _dividend_cash = 0

    @memorized_method()
    # TODO 未考虑反向开仓,未考虑分红派息导致的仓位、市值、和持仓成本变化
    def _position_info_detail(self):
        detail = []
        for ticker, trades in self._trades.groupby("order_book_id"):
            if DataAPI.bonus(ticker).size != 0:
                trades = self._reform_stock_trades(ticker, trades)
            # order_id cumsum_quantity Xposition_side  position_id  market_value  avg_price  profits  pnl
            temp = pd.DataFrame.from_records(self._trades_analyze(trades),
                 columns=['order_id', 'cumsum_quantity', 'market_value', 'avg_price',
                          'profits', 'transaction_cost', 'datetime'])

            temp["position_id"] = (temp["cumsum_quantity"] == 0).cumsum().shift(1).fillna(0) + 1  # TODO 未考虑反向开仓
            temp["position_id"] = temp["position_id"].astype(int)
            temp["position_side"] = (temp["cumsum_quantity"] >= 0).apply(sign2direction)
            temp["pnl"] = temp["profits"].fillna(0).cumsum() - temp["transaction_cost"].cumsum()
            temp['order_book_id'] = ticker

            detail.append(temp.set_index('order_id'))

        return pd.concat(detail, axis=0).sort_index()

    @staticmethod
    def _concat_market_data(df, index=None, market_data=None):
        df1 = df.drop("order_book_id", axis=1).set_index("datetime").groupby(level=0).last()
        df2 = market_data.loc[df.name]
        result = pd.concat([df1, df2], axis=1)
        return result.reindex(index).fillna(method="ffill").dropna()

    @memorized_method()
    def _position_info_detail_by_time(self):
        details = pd.concat([self._trades.drop(["datetime", "order_book_id"], axis=1),
                             self._position_info_detail()], axis=1)
        market_data = self.market_data
        index = sorted(pd.concat([details["datetime"], market_data.reset_index()["datetime"]]).unique())
        fields = ["cumsum_quantity", "pnl", "market_value", "position_side", "avg_price", "order_book_id", "datetime"]
        #print(details[fields].reset_index().set_index(["order_book_id", "datetime"]).sort_index().head(20),
        #      '======================================================\n')
        df = details[fields].reset_index().groupby("order_book_id").apply(partial(self._concat_market_data,
                                                                                  index=index,
                                                                                  market_data=market_data))
        #print(df.head(20))
        groupby = df.reset_index().groupby(["datetime", "order_book_id"]).last()
        groupby["float_pnl"] = groupby["pnl"] + (groupby["close"] - groupby["avg_price"]) * groupby["cumsum_quantity"]
        return groupby

    @memorized_method()
    def _position_info_detail_by_symbol(self):
        return self._position_info_detail().reset_index().set_index(["order_book_id", "order_id"]).sort_index()

    @property
    @memorized_method()
    def transactions(self):
        transactions = self._trades.copy()
        transactions["amount"] = (transactions["last_quantity"] * side2sign(transactions["side"]))
        transactions = transactions[["datetime", "order_book_id", "amount", "last_price"]]
        transactions.rename(columns={"order_book_id": "symbol", "last_price": "price"}, inplace=True)
        transactions.set_index("datetime", inplace=True)
        return transactions

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
        return self.position_info_detail_by_symbol[["datetime", "cumsum_quantity"]]

    @property
    def market_value(self):
        return self.position_info_detail_by_symbol[["datetime", "market_value"]]

    @property
    def pnl(self):
        return self.position_info_detail_by_symbol[["datetime", "pnl"]]

    @property
    def profits(self):
        return self.position_info_detail_by_symbol[["datetime", "profits"]]

    @property
    def average_price(self):
        return self.position_info_detail_by_symbol[["datetime", "avg_price"]]

    @property
    def pending_position(self):
        df = self.position.groupby(level=0).last()
        return df[df != 0].dropna()

    def _by_order2time(self, series):
        pass

    @property
    def market_value_by_time(self):
        return self.position_info_detail_by_time.apply(lambda x: x["close"] * x["cumsum_quantity"], axis=1)

    @property
    @memorized_method()
    def daily_market_value(self):
        market_value_by_time = self.market_value_by_time.unstack()
        series = market_value_by_time[market_value_by_time.index.hour == self._dailySumTime]
        series.index = series.index.normalize()
        series.name = "daily_market_value"
        return series.stack()

    @property
    @memorized_method()
    def daily_mv_df(self):
        market_value_by_time = self.market_value_by_time.unstack()
        daily_mv_df = market_value_by_time[market_value_by_time.index.hour == self._dailySumTime]
        daily_mv_df.index = daily_mv_df.index.normalize()
        daily_mv_df = pd.concat([daily_mv_df, self.daily_cash], axis=1)
        daily_mv_df = daily_mv_df.rename(columns={"daily_cash": "cash"})
        return daily_mv_df

    @property
    def security_value_by_time(self):
        series = self.market_value_by_time.groupby(level=0).sum()
        series.name = "security_value"
        return series

    @property
    def daily_security_value(self):
        security_value_by_time = self.security_value_by_time
        series = security_value_by_time[security_value_by_time.index.hour == self._dailySumTime]
        series.index = series.index.normalize()
        series.name = "daily_security_value"
        return series

    @property
    def position_by_time(self):
        series = self.position_info_detail_by_time["cumsum_quantity"]
        return series.loc[series != 0]

    @property
    def daily_position(self):
        position_by_time = self.position_by_time.unstack()
        series = position_by_time[position_by_time.index.hour == self._dailySumTime]
        series.index = series.index.normalize()
        series.name = "daily_position"
        return series.stack()

    @property
    def pnl_by_time(self):
        df = self.position_info_detail_by_time
        return df["pnl"].groupby(level=0).sum()

    @property
    def daily_pnl(self):
        pnl_by_time = self.pnl_by_time
        series = pnl_by_time[pnl_by_time.index.hour == self._dailySumTime]
        series.index = series.index.normalize()
        series.name = "daily_pnl"
        return series

    @property
    def float_pnl_by_time(self):
        df = self.position_info_detail_by_time
        return df["float_pnl"].groupby(level=0).sum()

    @property
    def daily_float_pnl(self):
        float_pnl_by_time = self.float_pnl_by_time
        series = float_pnl_by_time[float_pnl_by_time.index.hour == self._dailySumTime]
        series.index = series.index.normalize()
        series.name = "daily_float_pnl"
        return series

    @property
    def account_value_by_time(self):
        series = self.float_pnl_by_time + self._account["total_value"]
        series.name = "account_value"
        return series

    @property
    def daily_account_value(self):
        account_value_by_time = self.account_value_by_time
        series = account_value_by_time[account_value_by_time.index.hour == self._dailySumTime]
        series.index = series.index.normalize()
        series.name = "daily_account_value"
        return series

    @property
    def net(self):
        series = self.account_value_by_time / self._account["total_value"]
        series.name = "net"
        return series

    @property
    def daily_net(self):
        net = self.net
        series = net[net.index.hour == self._dailySumTime]
        series.index = series.index.normalize()
        series.name = "daily_net"
        return series

    @property
    def cash(self):
        series = self.account_value_by_time - self.security_value_by_time
        series.name = "cash"
        return series

    @property
    @memorized_method()
    def daily_cash(self):
        cash = self.cash
        series = cash[cash.index.hour == self._dailySumTime]
        series.index = series.index.normalize()
        series.name = "daily_cash"
        return series

    @property
    def returns(self):
        series = self.account_value_by_time.pct_change()
        series.name = "returns"
        return series

    @property
    @memorized_method()
    def daily_returns(self):
        series = self.daily_account_value.pct_change()
        if len(series):
            series[0] = self.daily_account_value[0] / self._account["total_value"] - 1
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
        start -= datetime.timedelta(days=1)

        if self._endSumDate is None:
            if len(self.pending_position) and self._extend:
                end = datetime.datetime.now()
            else:
                dt = self._trades["datetime"]
                if "datetime" in str(dt.dtype):
                    end = dt.iloc[-1].to_pydatetime()
                elif dt.dtype == np.dtype.str:
                    end = parse(self._trades["datetime"].iloc[-1]).date()
            end += datetime.timedelta(days=1)
        else:
            end = self._endSumDate
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

    @memorized_method()
    def benchmark_rets(self, code="000300.XSHG", freq="D", isIndex=True):
        start, end = self.date_range
        adjust = None
        if not isIndex:
            adjust = "before"
        benchmark = DataAPI.candle(code, freq=freq, fields="close", start=start, end=end, adjust=adjust)
        benchmark_rets = benchmark["close"].pct_change()
        benchmark_rets.name = "benchmark_rets"
        if freq == "D":
            benchmark_rets.index = benchmark_rets.index.normalize()
        return benchmark_rets

    @memorized_method()
    def market_data_panel(self, freq="D"):
        start, end = self.date_range
        df = DataAPI.candle(self.universe, start=start, end=end,
                            freq=freq).transpose(2, 1, 0)
        if freq == "D":
            df.major_axis = df.major_axis.normalize()
        return df

    @memorized_method()
    def symbol_sector_map(self, symbols=None):
        """
        获取行业分类信息
        :param symbols: 一组股票代码(list),形式为通用标准(编码.交易所 如["000001.XSHE","600000.XSHG"]),默认为策略的所有历史持有股票
        :return: 应应的sina的行业分类map。
        - Example:
            {'AAPL' : 'Technology'
             'MSFT' : 'Technology'
             'CHK' : 'Natural Resources'}
        """
        if symbols is None:
            symbols = self.universe
        sina_industy_class = DataAPI.info.classification()
        sina_industy_class.set_index("code", inplace=True)
        symbol_sector_map = {}
        for symbol in symbols:
            try:
                symbol_sector_map[symbol] = sina_industy_class.loc[symbol, "classification"]
            except:
                symbol_sector_map[symbol] = "行业未知"
        return symbol_sector_map

    @property
    @memorized_method()
    def positions_alloc(self):
        return pf.pos.get_percent_alloc(self.daily_mv_df)
