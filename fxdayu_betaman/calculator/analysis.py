# enconding = utf-8
from __future__ import division
from math import copysign
import warnings
from collections import deque, OrderedDict
import empyrical
import numpy as np
import pandas as pd
import scipy as sp
from pyfolio import capacity
from pyfolio import pos
from pyfolio import timeseries
from pyfolio import txn
from pyfolio import utils
from pyfolio.utils import (APPROX_BDAYS_PER_MONTH,
                           MM_DISPLAY_UNIT)

from fxdayu_betaman.calculator import BaseCalculator

##### TODO 因子贡献分析(pf.perf_attrib)，风险分析（pf.risk 行业及风格因子暴露)

STAT_FUNCS_PCT = [
        'Annual return',
        'Cumulative returns',
        'Annual volatility',
        'Max drawdown',
        'Daily value at risk',
        'Daily turnover'
    ]


class Performance(BaseCalculator):

    # 月度收益
    def monthly_returns(self,returns):
        """
        Returns by month.

        Parameters
        ----------
        returns : pd.Series
            Daily returns of the strategy, noncumulative.
             - See full explanation in pyfolio.tears.create_full_tear_sheet.

        Returns
        -------
        monthly_ret_table

        """
        monthly_ret_table = empyrical.aggregate_returns(returns, 'monthly')
        monthly_ret_table = monthly_ret_table.unstack().round(3).fillna(0)
        return monthly_ret_table

    # 年度收益率
    def yearly_returns(self,returns):
        """
        Returns by year.

        Parameters
        ----------
        returns : pd.Series
            Daily returns of the strategy, noncumulative.
             - See full explanation in pyfolio.tears.create_full_tear_sheet.

        Returns
        -------
        Returns by year
        """
        ann_ret_df = pd.DataFrame(
            empyrical.aggregate_returns(
                returns,
                'yearly'))
        ann_ret_df.columns=["annual_returns"]
        return ann_ret_df

    # 累计收益曲线
    def cum_rets(self,returns):
       """
       Cumulative returns.

       Parameters
       ----------
       returns : pd.Series
           Daily returns of the strategy, noncumulative.
            - See full explanation in pyfolio.tears.create_full_tear_sheet.

       Returns
       -------
       Cumulative returns
       """
       cum_rets = empyrical.cum_returns(returns, starting_value=1.0)
       return cum_rets

    # 持仓股票数量
    def holdings(self,positions):
        """
        total amount of stocks with an active position, either short
        or long. Displays daily total, daily average per month, and
        all-time daily average.

        Parameters
        ----------
        positions : pd.DataFrame, optional
            Daily net position values.
             - See full explanation in pyfolio.tears.create_full_tear_sheet.

        Returns
        -------
        holdings
        """
        positions = positions.copy().drop('cash', axis='columns')
        df_holdings = positions.replace(0, np.nan).count(axis=1)
        df_holdings_by_month = df_holdings.resample('1M').mean()
        holdings = {"holdings by day":df_holdings,
                    'holdings_by_month':df_holdings_by_month,
                    'avg holdings':df_holdings.mean()}
        return holdings

    # 最大回撤及恢复期
    def drawdown_periods(self, returns, top=10):
        """
        Top drawdown periods.

        Parameters
        ----------
        returns : pd.Series
            Daily returns of the strategy, noncumulative.
             - See full explanation in tears.create_full_tear_sheet.
        top : int, optional
            Amount of top drawdowns periods to plot (default 10).

        Returns
        -------
        drawdowns
        """

        drawdowns = timeseries.gen_drawdown_table(returns, top=top)
        return drawdowns

    # 收益回吐比例
    def drawdown_underwater(self, returns):
        """
        Plots how far underwaterr returns are over time, or plots current
        drawdown vs. date.

        Parameters
        ----------
        returns : pd.Series
            Daily returns of the strategy, noncumulative.
             - See full explanation in tears.create_full_tear_sheet.

        Returns
        -------
        drawdown_underwater
        """
        df_cum_rets = empyrical.cum_returns(returns, starting_value=1.0)
        running_max = np.maximum.accumulate(df_cum_rets)
        underwater = - ((running_max - df_cum_rets) / running_max)
        return underwater


    # 绩效指标
    def perf_stats(self,
                   returns,
                   factor_returns,
                   positions=None,
                   transactions=None,
                   live_start_date=None,
                   bootstrap=False):
        """
        Performance metrics of the strategy.

        - Shows amount of time the strategy has been run in backtest and
          out-of-sample (in live trading).

        - Shows Omega ratio, max drawdown, Calmar ratio, annual return,
          stability, Sharpe ratio, annual volatility, alpha, and beta.

        Parameters
        ----------
        returns : pd.Series
            Daily returns of the strategy, noncumulative.
             - See full explanation in pyfolio.tears.create_full_tear_sheet.
        factor_returns : pd.Series
            Daily noncumulative returns of the benchmark.
             - This is in the same style as returns.
        positions : pd.DataFrame
            Daily net position values.
             - See full explanation in create_full_tear_sheet.
        live_start_date : datetime, optional
            The point in time when the strategy began live trading, after
            its backtest period.
        bootstrap : boolean (optional)
            Whether to perform bootstrap analysis for the performance
            metrics.
             - For more information, see pyfolio.timeseries.perf_stats_bootstrap

        Returns
        -------
        Performance metrics of the strategy
        """

        perf = {}

        if bootstrap:
            perf_func = timeseries.perf_stats_bootstrap
        else:
            perf_func = timeseries.perf_stats

        perf_stats_all = perf_func(
            returns,
            factor_returns=factor_returns,
            positions=positions,
            transactions=transactions)

        if live_start_date is not None:
            live_start_date = empyrical.utils.get_utc_timestamp(live_start_date)
            returns_is = returns[returns.index < live_start_date]
            returns_oos = returns[returns.index >= live_start_date]

            positions_is = None
            positions_oos = None
            transactions_is = None
            transactions_oos = None

            if positions is not None:
                positions_is = positions[positions.index < live_start_date]
                positions_oos = positions[positions.index >= live_start_date]
                if transactions is not None:
                    transactions_is = transactions[(transactions.index <
                                                    live_start_date)]
                    transactions_oos = transactions[(transactions.index >
                                                     live_start_date)]

            perf_stats_is = perf_func(
                returns_is,
                factor_returns=factor_returns,
                positions=positions_is,
                transactions=transactions_is)

            perf_stats_oos = perf_func(
                returns_oos,
                factor_returns=factor_returns,
                positions=positions_oos,
                transactions=transactions_oos)

            perf["In-sample months"] = int(len(returns_is) / APPROX_BDAYS_PER_MONTH)
            perf["Out-of-sample months"] = int(len(returns_oos) / APPROX_BDAYS_PER_MONTH)

            perf_stats = pd.concat(OrderedDict([
                ('In-sample', perf_stats_is),
                ('Out-of-sample', perf_stats_oos),
                ('All', perf_stats_all),
            ]), axis=1)
        else:
            perf["Backtest months"] = int(len(returns) / APPROX_BDAYS_PER_MONTH)
            perf_stats = pd.DataFrame(perf_stats_all, columns=['Backtest'])

        for column in perf_stats.columns:
            for stat, value in perf_stats[column].iteritems():
                if stat in STAT_FUNCS_PCT:
                    perf_stats.loc[stat, column] = str(np.round(value * 100,
                                                                1)) + '%'

        perf["perf_stats"] = perf_stats
        return perf

    # 相对某个风险因子的beta收益（如：对大盘的beta）
    def rolling_beta(self,
                     returns,
                     factor_returns):
        """
        the rolling 6-month and 12-month beta versus date.

        Parameters
        ----------
        returns : pd.Series
            Daily returns of the strategy, noncumulative.
             - See full explanation in tears.create_full_tear_sheet.
        factor_returns : pd.Series, optional
            Daily noncumulative returns of the benchmark.
             - This is in the same style as returns.

        Returns
        -------
        the rolling 6-month and 12-month beta versus date.
        """
        rolling_beta = {}
        rolling_beta["6-month"] = timeseries.rolling_beta(
            returns, factor_returns, rolling_window=APPROX_BDAYS_PER_MONTH * 6)
        rolling_beta["12-month"] = timeseries.rolling_beta(
            returns, factor_returns, rolling_window=APPROX_BDAYS_PER_MONTH * 12)
        return rolling_beta

    # 滚动波动率 默认6个月
    def rolling_volatility(self,
                           returns,
                           rolling_window=APPROX_BDAYS_PER_MONTH * 6):
        """
        rolling volatility versus date.

        Parameters
        ----------
        returns : pd.Series
            Daily returns of the strategy, noncumulative.
             - See full explanation in tears.create_full_tear_sheet.
        rolling_window : int, optional
            The days window over which to compute the volatility.

        Returns
        -------
        rolling volatility versus date
        """
        rolling_vol_ts = timeseries.rolling_volatility(
            returns, rolling_window)
        return rolling_vol_ts

    def rolling_sharpe(self,
                       returns,
                       rolling_window=APPROX_BDAYS_PER_MONTH * 6,):
        """
        The rolling Sharpe ratio versus date.

        Parameters
        ----------
        returns : pd.Series
            Daily returns of the strategy, noncumulative.
             - See full explanation in tears.create_full_tear_sheet.
        rolling_window : int, optional
            The days window over which to compute the sharpe ratio.


        Returns
        -------
        The rolling Sharpe ratio versus date
        """
        rolling_sharpe_ts = timeseries.rolling_sharpe(
            returns, rolling_window)
        return rolling_sharpe_ts

    # 杠杆率（纯多头策略等同于仓位比重）
    def gross_leverage(self,positions):
        """
        Gross leverage versus date.

        Gross leverage is the sum of long and short exposure per share
        divided by net asset value.

        Parameters
        ----------
        positions : pd.DataFrame
            Daily net position values.
             - See full explanation in create_full_tear_sheet.

        Returns
        -------
        Gross leverage versus date
        """
        gl = timeseries.gross_lev(positions)
        return gl

    # 仓位比重(持仓市值/总市值)
    def exposures(self,positions):
        """
        The long and net and short exposure.

        Parameters
        ----------
        positions : pd.DataFrame
            Daily net position values.
             - See full explanation in create_full_tear_sheet.

        Returns
        -------
        The long and net and short exposure.
        """
        pos_no_cash = positions.drop('cash', axis=1)
        l_exp = pos_no_cash[pos_no_cash > 0].sum(axis=1) / positions.sum(axis=1)
        s_exp = pos_no_cash[pos_no_cash < 0].sum(axis=1) / positions.sum(axis=1)
        net_exp = pos_no_cash.sum(axis=1) / positions.sum(axis=1)
        exp = {"long_exp":l_exp,"short_exp":s_exp,"net_exp":net_exp}
        return exp

    # 历史重/持仓股(某一日达到该日top要求即入选)
    def top_positions(self,
                      positions,
                      top=10):
        """
        The exposures of the top 10 and all held positions of
        all time.

        Parameters
        ----------

        positions : pd.DataFrame
            Daily net position values.
             - See full explanation in create_full_tear_sheet.
        top : int, optional
            How many of each to find (default 10).

        Returns
        -------
        The exposures of the top 10 and all held positions of
        all time.

        """
        positions_alloc = pos.get_percent_alloc(positions)
        positions_alloc.columns = positions_alloc.columns.map(utils.format_asset)
        df_top_long, df_top_short, df_top_abs = pos.get_top_long_short_abs(
            positions_alloc,top)
        _, _, df_top_abs_all = pos.get_top_long_short_abs(
            positions_alloc, top=9999)
        return {"top_long":df_top_long,
                "top_short":df_top_short,
                "top_abs":df_top_abs,
                "all":df_top_abs_all}

    # 每天持有个股的最大头寸占比和头寸占比众数（分多空头)
    def max_median_position_concentration(self,positions):
        """
        The max and median of long and short position concentrations
        over the time.

        Parameters
        ----------
        positions : pd.DataFrame
            The positions that the strategy takes over time.

        Returns
        -------
        每天持有个股的最大头寸占比和头寸占比众数（分多空头)
        """
        alloc_summary = pos.get_max_median_position_concentration(positions)
        return alloc_summary

    # 行业风险暴露（行业资金分配）
    def sector_exposures(self, positions, symbol_sector_map):
        """
        The sector exposures of the portfolio over time.

        Parameters
        ----------
        positions : pd.DataFrame
            Contains position values or amounts.
            - Example
                index         'AAPL'         'MSFT'        'CHK'        cash
                2004-01-09    13939.380     -15012.993    -403.870      1477.483
                2004-01-12    14492.630     -18624.870    142.630       3989.610
                2004-01-13    -13853.280    13653.640     -100.980      100.000
        symbol_sector_map : dict or pd.Series
            Security identifier to sector mapping.
            Security ids as keys/index, sectors as values.
            - Example:
                {'AAPL' : 'Technology'
                 'MSFT' : 'Technology'
                 'CHK' : 'Natural Resources'}

        Returns
        -------
        sector_exp : pd.DataFrame
            Sectors and their allocations.
            - Example:
                index         'Technology'    'Natural Resources' cash
                2004-01-09    -1073.613       -403.870            1477.4830
                2004-01-12    -4132.240       142.630             3989.6100
                2004-01-13    -199.640        -100.980            100.0000
        """
        sector_alloc = pos.get_sector_exposures(positions, symbol_sector_map)
        return sector_alloc

    # 按样本内外的日度、月度、周度来呈现收益分布特征
    def return_quantiles(self,
                         returns,
                         live_start_date=None):
        """
        Daily, weekly, and monthly return series

        Parameters
        ----------
        returns : pd.Series
            Daily returns of the strategy, noncumulative.
             - See full explanation in tears.create_full_tear_sheet.
        live_start_date : datetime, optional
            The point in time when the strategy began live trading, after
            its backtest period.

        Returns
        -------
        Daily, weekly, and monthly return series
        """
        is_returns = returns if live_start_date is None \
            else returns.loc[returns.index < live_start_date]
        is_weekly = empyrical.aggregate_returns(is_returns, 'weekly')
        is_monthly = empyrical.aggregate_returns(is_returns, 'monthly')

        return_quantiles = {"is":{'Daily':is_returns, 'Weekly':is_weekly, 'Monthly':is_monthly}}
        if live_start_date is not None:
            oos_returns = returns.loc[returns.index >= live_start_date]
            oos_weekly = empyrical.aggregate_returns(oos_returns, 'weekly')
            oos_monthly = empyrical.aggregate_returns(oos_returns, 'monthly')
            return_quantiles["oos"] = {'Daily':oos_returns, 'Weekly':oos_weekly, 'Monthly':oos_monthly}

        return return_quantiles

    # 换手率
    def turnover(self,
                 transactions,
                 positions):

        """
        turnover vs. date.

        Turnover is the number of shares traded for a period as a fraction
        of total shares.

        Displays daily total, daily average per month, and all-time daily
        average.

        Parameters
        ----------
        transactions : pd.DataFrame
            Prices and amounts of executed trades. One row per trade.
             - See full explanation in tears.create_full_tear_sheet.
        positions : pd.DataFrame
            Daily net position values.
             - See full explanation in tears.create_full_tear_sheet.

        Returns
        -------
        turnover vs. date.
        """
        df_turnover = txn.get_turnover(positions, transactions)
        df_turnover_by_month = df_turnover.resample("M").mean()
        return {"Daily":df_turnover,
                "Monthly":df_turnover_by_month,
                "Mean":df_turnover.mean()}

    # 不同滑点下的净值曲线
    # 注：1bps = 0.0001
    # A股一个点是0.01 即100bps
    def slippage_sweep(self,
                       returns,
                       transactions,
                       positions,
                       slippage_params=(100, 200, 300, 400, 500)):
        """
        Equity curves at different per-dollar slippage assumptions.
        注：1bps = 0.0001
        A股一个点是0.01 即100bps
        Parameters
        ----------
        returns : pd.Series
            Timeseries of portfolio returns to be adjusted for various
            degrees of slippage.
        transactions : pd.DataFrame
            Prices and amounts of executed trades. One row per trade.
             - See full explanation in tears.create_full_tear_sheet.
        positions : pd.DataFrame
            Daily net position values.
             - See full explanation in tears.create_full_tear_sheet.
        slippage_params: tuple
            Slippage pameters to apply to the return time series (in
            basis points).

        Returns
        -------
        Equity curves at different per-dollar slippage assumptions.
        """

        turnover = txn.get_turnover(positions, transactions,
                                    period=None, average=False)

        slippage_sweep = pd.DataFrame()
        for bps in slippage_params:
            adj_returns = txn.adjust_returns_for_slippage(returns, turnover, bps)
            label = str(bps) + " bps"
            slippage_sweep[label] = empyrical.cum_returns(adj_returns, 1)

        return slippage_sweep

    # 滑点敏感度 —— 不同滑点下对应的年化收益率
    def slippage_sensitivity(self,
                             returns,
                             transactions,
                             positions,
                             slippage_range = range(100,5000,100)
                             ):

        """
        Curve relating per-dollar slippage to average annual returns.

        Parameters
        ----------
        returns : pd.Series
            Timeseries of portfolio returns to be adjusted for various
            degrees of slippage.
        transactions : pd.DataFrame
            Prices and amounts of executed trades. One row per trade.
             - See full explanation in tears.create_full_tear_sheet.
        positions : pd.DataFrame
            Daily net position values.
             - See full explanation in tears.create_full_tear_sheet.

        Returns
        -------
        avg_returns_given_slippage
        """
        turnover = txn.get_turnover(positions, transactions,
                                    period=None, average=False)
        avg_returns_given_slippage = pd.Series()
        for bps in slippage_range:
            adj_returns = txn.adjust_returns_for_slippage(returns, turnover, bps)
            avg_returns = empyrical.annual_return(adj_returns)
            avg_returns_given_slippage.loc[bps] = avg_returns
        return avg_returns_given_slippage

    # 考虑资金量带来市场冲击后的修正夏普率
    # 默认入场资金从100000 - 300000000， 步长1000000
    # 返回指定资金量下的修正夏普率
    def capacity_sweep(self,
                       returns, transactions, market_data,
                       bt_starting_capital,
                       min_pv=100000,
                       max_pv=300000000,
                       step_size=1000000):
        """
        考虑资金量（每百万元）带来市场冲击后的修正夏普率

        Parameters
        ----------
        returns : pd.Series
            Timeseries of portfolio returns to be adjusted for various
            degrees of slippage.
        transactions : pd.DataFrame
            Prices and amounts of executed trades. One row per trade.
             - See full explanation in tears.create_full_tear_sheet.
        market_data : pd.Panel, optional
            Panel with items axis of 'price' and 'volume' DataFrames.
            The major and minor axes should match those of the
            the passed positions DataFrame (same dates and symbols).
        min_pv:int
            最小入场资金
        max_pv:int
            最大入场资金
        step_size:int
            步长

        Returns
        -------
        考虑资金量（每百万元）带来市场冲击后的修正夏普率（adj_sharpe）
        资金量为index，单位为百万元
        """
        txn_daily_w_bar = capacity.daily_txns_with_bar_data(transactions,
                                                            market_data)

        captial_base_sweep = pd.Series()
        for start_pv in range(min_pv, max_pv, step_size):
            adj_ret = capacity.apply_slippage_penalty(returns,
                                                      txn_daily_w_bar,
                                                      start_pv,
                                                      bt_starting_capital)
            sharpe = empyrical.sharpe_ratio(adj_ret)
            if sharpe < -1:
                break
            captial_base_sweep.loc[start_pv] = sharpe
        captial_base_sweep.index = captial_base_sweep.index / MM_DISPLAY_UNIT
        return captial_base_sweep

    # 日交易手数和交易金额
    def daily_volume(self,
                     transactions):
        """
        Trading volume per day vs. date.

        Parameters
        ----------

        transactions : pd.DataFrame
            Prices and amounts of executed trades. One row per trade.
             - See full explanation in tears.create_full_tear_sheet.

        Returns
        -------
        策略日交易手数和交易金额
        """
        daily_txn = txn.get_txn_vol(transactions)
        return daily_txn

    def _groupby_consecutive(self,
                             txn,
                             max_delta=pd.Timedelta('8h')):
        """Merge transactions of the same direction separated by less than
        max_delta time duration.

        Parameters
        ----------
        transactions : pd.DataFrame
            Prices and amounts of executed round_trips. One row per trade.
            - See full explanation in tears.create_full_tear_sheet

        max_delta : pandas.Timedelta (optional)
            Merge transactions in the same direction separated by less
            than max_delta time duration.


        Returns
        -------
        transactions : pd.DataFrame

        """
        def vwap(transaction):
            if transaction.amount.sum() == 0:
                warnings.warn('Zero transacted shares, setting vwap to nan.')
                return np.nan
            return (transaction.amount * transaction.price).sum() / \
                transaction.amount.sum()

        out = []
        for sym, t in txn.groupby('symbol'):
            t = t.sort_index()
            t.index.name = 'dt'
            t = t.reset_index()

            t['order_sign'] = t.amount > 0
            t['block_dir'] = (t.order_sign.shift(
                1) != t.order_sign).astype(int).cumsum()

            temp = [t.dt.iloc[i] - t.dt.shift(1).iloc[i] > max_delta for i in t.dt.index]
            temp = pd.Series(temp,index = t.dt.index)
            t['block_time'] =pd.Series(temp,index = t.dt.index).astype(int).cumsum()
            grouped_price = (t.groupby(('block_dir',
                                       'block_time'))
                              .apply(vwap))
            grouped_price.name = 'price'
            grouped_rest = t.groupby(('block_dir', 'block_time')).agg({
                'amount': 'sum',
                'symbol': 'first',
                'dt': 'first'})

            grouped = grouped_rest.join(grouped_price)

            out.append(grouped)

        out = pd.concat(out)
        out = out.set_index('dt')
        return out

    # 提取往返交易（把每日同品种同向订单合并，然后匹配隔日不同方向的开单为一个round trip）
    def extract_round_trips(self,
                            transactions,
                            portfolio_value=None):
        """Group transactions into "round trips". First, transactions are
        grouped by day and directionality. Then, long and short
        transactions are matched to create round-trip round_trips for which
        PnL, duration and returns are computed. Crossings where a position
        changes from long to short and vice-versa are handled correctly.

        Under the hood, we reconstruct the individual shares in a
        portfolio over time and match round_trips in a FIFO-order.

        For example, the following transactions would constitute one round trip:
        index                  amount   price    symbol
        2004-01-09 12:18:01    10       50      'AAPL'
        2004-01-09 15:12:53    10       100      'AAPL'
        2004-01-13 14:41:23    -10      100      'AAPL'
        2004-01-13 15:23:34    -10      200       'AAPL'

        First, the first two and last two round_trips will be merged into a two
        single transactions (computing the price via vwap). Then, during
        the portfolio reconstruction, the two resulting transactions will
        be merged and result in 1 round-trip trade with a PnL of
        (150 * 20) - (75 * 20) = 1500.

        Note, that round trips do not have to close out positions
        completely. For example, we could have removed the last
        transaction in the example above and still generated a round-trip
        over 10 shares with 10 shares left in the portfolio to be matched
        with a later transaction.

        Parameters
        ----------
        transactions : pd.DataFrame
            Prices and amounts of executed round_trips. One row per trade.
            - See full explanation in tears.create_full_tear_sheet

        portfolio_value : pd.Series (optional)
            Portfolio value (all net assets including cash) over time.
            Note that portfolio_value needs to beginning of day, so either
            use .shift() or positions.sum(axis='columns') / (1+returns).

        Returns
        -------
        round_trips : pd.DataFrame
            DataFrame with one row per round trip.  The returns column
            contains returns in respect to the portfolio value while
            rt_returns are the returns in regards to the invested capital
            into that partiulcar round-trip.
        """

        transactions = self._groupby_consecutive(transactions)
        roundtrips = []

        for sym, trans_sym in transactions.groupby('symbol'):
            trans_sym = trans_sym.sort_index()
            price_stack = deque()
            dt_stack = deque()
            trans_sym['signed_price'] = trans_sym.price * \
                np.sign(trans_sym.amount)
            trans_sym['abs_amount'] = trans_sym.amount.abs().astype(int)
            for dt, t in trans_sym.iterrows():
                if t.price < 0:
                    warnings.warn('Negative price detected, ignoring for'
                                  'round-trip.')
                    continue

                indiv_prices = [t.signed_price] * t.abs_amount
                if (len(price_stack) == 0) or \
                   (copysign(1, price_stack[-1]) == copysign(1, t.amount)):
                    price_stack.extend(indiv_prices)
                    dt_stack.extend([dt] * len(indiv_prices))
                else:
                    # Close round-trip
                    pnl = 0
                    invested = 0
                    cur_open_dts = []

                    for price in indiv_prices:
                        if len(price_stack) != 0 and \
                           (copysign(1, price_stack[-1]) != copysign(1, price)):
                            # Retrieve first dt, stock-price pair from
                            # stack
                            prev_price = price_stack.popleft()
                            prev_dt = dt_stack.popleft()

                            pnl += -(price + prev_price)
                            cur_open_dts.append(prev_dt)
                            invested += abs(prev_price)

                        else:
                            # Push additional stock-prices onto stack
                            price_stack.append(price)
                            dt_stack.append(dt)

                    roundtrips.append({'pnl': pnl,
                                       'open_dt': cur_open_dts[0],
                                       'close_dt': dt,
                                       'long': price < 0,
                                       'rt_returns': pnl / invested,
                                       'symbol': sym,
                                       })

        roundtrips = pd.DataFrame(roundtrips)
        temp = [roundtrips['close_dt'].iloc[i] - roundtrips['open_dt'].iloc[i] for i in roundtrips['close_dt'].index]
        roundtrips['duration'] = pd.Series(temp, index=roundtrips['close_dt'].index)

        if portfolio_value is not None:
            # Need to normalize so that we can join
            pv = pd.DataFrame(portfolio_value,
                              columns=['portfolio_value'])\
                .assign(date=portfolio_value.index)

            roundtrips['date'] = roundtrips.close_dt.apply(lambda x:
                                                           x.replace(hour=0,
                                                                     minute=0,
                                                                     second=0))

            tmp = roundtrips.join(pv, on='date', lsuffix='_')

            roundtrips['returns'] = tmp.pnl / tmp.portfolio_value
            roundtrips = roundtrips.drop('date', axis='columns')

        return roundtrips

    # 每支股票的收益贡献
    def profit_attribution(self,round_trips):
        """
        The share of total PnL contributed by each
        traded name.

        Parameters
        ----------
        round_trips : pd.DataFrame
            DataFrame with one row per round trip trade.
            - See full explanation in round_trips.extract_round_trips

        Returns
        -------

        """

        total_pnl = round_trips['pnl'].sum()
        pnl_attribution = round_trips.groupby('symbol')['pnl'].sum() / total_pnl
        pnl_attribution.name = ''

        pnl_attribution.index = pnl_attribution.index.map(utils.format_asset)
        return pnl_attribution.sort_values(inplace=False,ascending=False)


    # 下盈利单的可能性分布
    def prob_profit_trade(self,round_trips):
        """
        Probability distribution for the event of making
        a profitable trade.

        Parameters
        ----------
        round_trips : pd.DataFrame
            DataFrame with one row per round trip trade.
            - See full explanation in round_trips.extract_round_trips

        Returns
        -------
        Probability distribution for the event of making
        a profitable trade.
        """

        x = np.linspace(0, 1., 500)
        round_trips['profitable'] = round_trips.pnl > 0
        dist = sp.stats.beta(round_trips.profitable.sum(),
                             (~round_trips.profitable).sum())
        y = dist.pdf(x)
        return {"Probability":x,
                "Belief":y,
                "lower_perc":dist.ppf(.025),
                'upper_perc':dist.ppf(.975)}


