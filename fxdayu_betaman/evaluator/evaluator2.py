# 重构主要是从全市场算ic，然后算各细分范围的ic效果

import jaqs_fxdayu
from jaqs_fxdayu.research import SignalDigger
from jaqs_fxdayu.research.signaldigger import performance as pfm
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from scipy.stats import spearmanr, ttest_1samp
from functools import partial
from collections import namedtuple
from jaqs.trade import common


index_map = {
    "hs300": "000300.SH_member",
    "zz500": '000905.SH_member',
    "sz50": '000016.SH_member',
}


class Evaluator:

    def __init__(self, dv, signal, limit_rules="A-share default"):
        """
        :param dv: jaqs.dataview
        :param signal: pd.DataFrame
        :param limit_rules: 限制条件 dict {mask,can_enter,can_exit}
        """

        self.dv = dv
        self._signal = signal if isinstance(signal, pd.Series) else signal.stack()
        self._signal = self._signal.rename("signal")
        self.methods = {
            "mad": self._mad,
            "winsorize": self._winsorize,
            "barra": self._barra,
            "standard_scale": lambda x: pd.Series(scale(x), index=x.index, name="signal"),
        }
        self._cap = None
        self._industry_standard = None
        self._style = None
        self.data = None
        self.residuals = None

        # =========================================================================================
        for field in ["trade_status", "close_adj", "high_adj", "low_adj"]:
            if field not in self.dv.fields:
                raise ValueError("请确保dv中必须提供的字段-%s存在!" % (field,))

        if isinstance(limit_rules, str):
            if limit_rules == "A-share default":
                self.limit_rules = self._a_share_default_rule()
            else:
                raise ValueError("limit rules only support 'A-share default' now")
        elif isinstance(limit_rules, dict):
            for rule in limit_rules.keys():
                if not (rule in ["mask", "can_enter", "can_exit"]):
                    raise ValueError("limit_rule的keys只能为'mask','can_enter','can_exit'")
            self.limit_rules = limit_rules

    def _a_share_default_rule(self):
        trade_status = self.dv.get_ts('trade_status')
        mask_sus = trade_status.fillna("") == u'停牌'
        # 涨停
        up_limit = self.dv.add_formula('up_limit', '(close_adj - Delay(close_adj, 1)) / Delay(close_adj, 1) > 0.095',
                                       is_quarterly=False)
        # 跌停
        down_limit = self.dv.add_formula('down_limit','(close_adj - Delay(close_adj, 1)) / Delay(close_adj, 1) < -0.095',
                                         is_quarterly=False)
        can_enter = np.logical_and(up_limit < 1, ~mask_sus)  # 未涨停未停牌
        can_exit = np.logical_and(down_limit < 1, ~mask_sus)  # 未跌停未停牌
        return {
            "can_enter": can_enter,
            "can_exit": can_exit,
            "mask": None,
        }

    def generate_residuals(self, style="float_mv", industry_standard="sw1", cap="float_mv",
                    preprocessing=("mad", "neutralization", "standard_scale")):  # 行业范围
        """
        主要用作对因子对中性化。。。
        :param period: 评估周期 int
        :param benchmark: 指数价格 pd.DataFrame/pd.Series
        :param commission:双边手续费率 float 默认0.0008
        :param industry_standard:行业标准 str/pd.DataFrame/pd.Series
        :param cap:流通市值 str/pd.DataFrame/pd.Series
        :param time:list of tuple, e.g. [('20170101', '20170201')]
        :param comp:指数成分 只针对该成分股票进行评估 str/pd.DataFrame/pd.Series
        :param industry:行业成分 list 只针对该行业成分股票进行评估 行业元素需包含在设置的行业标准中
        :return:
        """

        # self.style_factor = None  # 还没想好方案, 市值
        if isinstance(style, str):
            if not (style in self.dv.fields):
                raise ValueError("请确保dv中必须提供的市值因子标准(cap)字段-%s存在!" % (style,))
            self._style = self.dv.get_ts(style).stack().apply(np.log)  # Attention!
        elif style is None:
            raise ValueError("not yet ready for complete Barra style factors")
        elif isinstance(style, pd.DataFrame):  # 为什么需要的DataFrame我给忘了
            self._style = style.stack()
        elif isinstance(style, pd.Series):
            pass
        else:
            raise ValueError("cap should be one of str, DataFrame and Series")

        self._cap = self.dv.get_ts(cap)
        if isinstance(cap, str):
            if not (cap in self.dv.fields):
                raise ValueError("请确保dv中必须提供的市值因子标准(cap)字段-%s存在!" % (cap,))
            self._cap = self.dv.get_ts(cap).stack()
        elif isinstance(cap, pd.DataFrame):
            self._cap = self._cap.stack()
        elif isinstance(cap, pd.Series):
            pass
        else:
            raise ValueError("cap should be one of str, DataFrame and Series")
        self._cap = self._cap.reindex_like(self._signal).apply(np.log)

        if isinstance(industry_standard, str):  # str的话只支持一个指标
            if not (industry_standard in self.dv.fields):
                raise ValueError("请确保dv中必须提供的行业分类标准(industry_standard)字段-%s存在!" % (industry_standard,))
            self._industry_standard = self.dv.get_ts(industry_standard).stack()
        elif isinstance(industry_standard, pd.DataFrame):
            self._industry_standard = industry_standard.stack()
        elif isinstance(industry_standard, pd.Series):
            pass
        else:
            raise ValueError("industry_standard should be one of str, DataFrame and Series")

        def drop_sw1_nan(s):
            """
            主要是有一支'000748.SZ'的股票 
            """
            return s[s != "nan"]

        self._industry_standard = drop_sw1_nan(self._industry_standard).rename("industry")

        self.data = pd.concat([self._signal, self._style, self._industry_standard], axis=1).dropna()
        # ===========================================上面预处理======================================================

        self.residuals = pd.concat(self._generate_cross_sectional_residual(preprocessing))

    def _generate_cross_sectional_residual(self, preprocessing):
        """
        input dataframe should has two columns,[signal, return], cross setion data, so no multiindex
        ['signal', 'return', 'upside_ret', 'downside_ret']
        """
        for _, X in self.data.groupby(level=0):
            signal = X.pop("signal")
            X = pd.concat([X, pd.get_dummies(X.pop("industry"))], axis=1)
            for method in preprocessing:
                if method != "neutralization":
                    signal = self.methods[method](signal)
                else:
                    signal = pd.Series(self._ols_by_numpy(X.values, signal), index=signal.index, name=signal.name)
            yield signal

    @staticmethod
    def _ols_by_numpy(x, y):
        """
        x 应该是3000*40的格式
        """
        if len(x.shape) == 1:
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A, y)[0]
            resid = y - (m * x + c)
            return resid
        elif len(x.shape) == 2:
            A = np.hstack([x, np.ones(len(x)).reshape(-1, 1)])
            *m, c = np.linalg.lstsq(A, y)[0]
            resid = y - (x@m + c)
            return resid

    @staticmethod
    def _mad(series):
        median = series.median()
        mad = (series-median).abs().median()
        return series.clip(median-5*mad, median+5*mad)  # .rename("signal")

    @staticmethod
    def _winsorize(series):
        q = series.quantile([0.025, 0.975])
        series[series < q.iloc[0]] = q.iloc[0]
        series[series > q.iloc[1]] = q.iloc[1]
        return series  # .rename("signal")

    @staticmethod
    def _barra0(series, cap_series):
        """
        MultiIndex; 同时应该包含两个columns,[factor, cap(market_value)]

        """
        df = pd.concat([series, cap_series.reindex(index=series.index)], axis=1)
        first_step_factor = df.groupby(level=0).apply(
            lambda x: x.iloc[:, 0] - (x.iloc[:, 0] * x.iloc[:, 1] / (x.iloc[:, 1].sum())) / x.iloc[:, 0].std()
        )
        if isinstance(first_step_factor, pd.Series):
            first_step_factor.index = first_step_factor.index.droplevel(0)
        else:
            first_step_factor = first_step_factor.squeeze()
        return first_step_factor.clip(-3.5, 3.5)  # .rename("signal")

    def _barra(self, series):
        return self._barra0(series, self._cap)

    def generate_dimensions(self, period,
                            time=None,  # 时间范围
                            comp=None,  # 指数成分范围
                            industry=None,
                            benchmark=None, commission=0.0008):

        data = self.residuals.unstack()

        if comp is not None:  # ("hs300", "zz500")
            if isinstance(comp, str):
                field = index_map[comp] if comp in index_map.keys() else comp
                if not (field in self.dv.fields):
                    raise ValueError("请确保dv中必须提供的指数成分数据(comp)字段-%s存在!" % (field,))
                member = self.dv.get_ts(field)
                data = data[member == 1].dropna(how="all", axis=0).dropna(how="all", axis=1)
            elif isinstance(comp, pd.DataFrame):
                data = data.reindex_like(comp)
                data = data[comp == 1].dropna(how="all", axis=0).dropna(how="all", axis=1)
            elif isinstance(comp, pd.Series):
                data = data[comp == 1].dropna(how="all", axis=0).dropna(how="all", axis=1)
            else:
                raise ValueError("comp should be one of str, DataFrame and Series")

        if industry is not None:
            assert isinstance(industry, list), "industry should be a list"
            data = data[self._industry_standard.unstack().isin(industry)].dropna(how="all", axis=0).dropna(how="all", axis=1)

        data = self._generate_data(data,
                                   period=period,
                                   benchmark=benchmark,
                                   commission=commission,
                                   **self.limit_rules).drop("quantile", axis=1)

        def del_all_zero_date(df):
            index = df["return"].groupby(level=0).transform(lambda x: x if (x != 0).any() else None).dropna().index
            return df.reindex(index)

        data = del_all_zero_date(data)

        if time is not None:
            assert isinstance(time, list), "time should be a list of tuple, e.g. [('20170101', '20170201')]"
            data = pd.concat(data.loc[(slice(start_time, end_time),), :] for start_time, end_time in time)

        return Dimensions(data, period)

    # 后面
    def _generate_data(self,
                       signal,
                       period,
                       mask=None,
                       can_enter=None,
                       can_exit=None,
                       benchmark=None,
                       commission=0.0008):
        '''
        :param signal:
        :param period:
        :param mask:
        :param can_enter:
        :param can_exit:
        :param benchmark:
        :param commission:
        :return:
        '''

        obj = SignalDigger()
        # 处理因子 计算目标股票池每只股票的持有期收益，和对应因子值的quantile分类
        obj.process_signal_before_analysis(signal=signal,
                                           price=self.dv.get_ts("close_adj").reindex_like(signal),
                                           high=self.dv.get_ts("high_adj").reindex_like(signal),
                                           low=self.dv.get_ts("low_adj").reindex_like(signal),
                                           n_quantiles=5,
                                           mask=mask.reindex_like(signal) if mask is not None else None,
                                           can_enter=can_enter.reindex_like(signal) if can_enter is not None else None,  # 是否能进场
                                           can_exit=can_exit.reindex_like(signal) if can_exit is not None else None,  # 是否能出场
                                           period=period,  # 持有期
                                           benchmark_price=benchmark,  # 基准价格 可不传入，持有期收益（return）计算为绝对收益
                                           commission=commission,
                                           )
        return obj.signal_data


class Dimensions:

    def __init__(self, signal, period):
        self.signal_data = signal
        self.period = period

    def _result(self, df):
        # for date, dataframe in grouped:
        grouped = df.groupby(level=0)
        return pd.DataFrame.from_records(
            (self._cross_sectional_function(dataframe) for _, dataframe in grouped),
            # columns=["回归系数t值", "回归系数", "回归系数 p值", "IC", "IC p值",
            #          "最大回报IC", "最大回报IC p值", "最低回报IC", "最低回报IC p值"]
            columns=["IC", "IC p值", "最大回报IC", "最大回报IC p值", "最低回报IC", "最低回报IC p值"]
        )

    def _cross_sectional_function(self, X):
        upside_ret = X.pop("upside_ret")
        downside_ret = X.pop("downside_ret")
        ret = X.pop("return")
        signal_series = X.pop("signal")
        return (*self.two_column_rank_ic(ret, signal_series)), \
                (*self.two_column_rank_ic(upside_ret, signal_series)), \
                (*self.two_column_rank_ic(downside_ret, signal_series))

    def __call__(self,
                 p_threshold=0.05,
                 n_quantiles=10,
                 calc_full_report=False):

        '''
        :param regression_method: 回归方法 目前仅支持 ols wls
        :param preprocessing: tuple,分别代表因子数据预处理的方法,按顺序依次执行.一般依次为去极值,中性化,标准化
                                      支持neutralization_both, neutralization_cap, neutralization_industry三种中性化方法
        :param p_threshold: float 显著性水平标准 通常为0.05
        :param n_quantiles: int quantile 分类数 默认10
        :param calc_full_report: bool 是否计算完整报告 --因子日度ic 收益等
        :return:
        '''

        signal_data = self.signal_data
        result = self._result(signal_data)

        # 预测能力绩效评估表
        coef_df = pfm.calc_ic_stats_table(result[["IC", "最大回报IC", "最低回报IC"]]).rename(
            columns={"IC Mean": "Mean", "IC Std.": "Std.", "t-stat(IC)": "t-stat",
                     "p-value(IC)": "p-value", "IC Skew": "Skew", "IC Kurtosis": "Kurtosis"})

        # 稳定性绩效评估表
        stability_df = pd.concat(map(partial(self._stability_df_transform, pvalue_threshold=p_threshold),
                                     (result[["IC", "IC p值"]], result[["最大回报IC", "最大回报IC p值"]],
                                      result[["最低回报IC", "最低回报IC p值"]])), axis=1)


        # 划分 quantile 计算投资组合收益
        try:
            signal_data["quantile"] = signal_data["signal"].dropna().groupby(level=0).apply(pd.qcut, q=n_quantiles, labels=np.arange(n_quantiles)+1)
        except ValueError:
            print("quantile cut do not work")
            signal_data["quantile"] = 1
        signal_data["bins"] = signal_data["signal"].dropna().groupby(level=0).apply(pd.cut, bins=n_quantiles, labels=np.arange(n_quantiles)+1)

        # 下面先对因子做处理，可能要改，因为不确定能不能跟前面的整合
        # copy_signal_data = signal_data.copy()
        # copy_signal_data["signal"] = self.signal_for_ic
        # copy_signal_data["signal"] = copy_signal_data["signal"].groupby(level=0).transform(
        #     lambda x: self.combined_method(
        #         list(map(lambda y: self.methods[y], profit_signal_preprocessing)), x))
        # 持有收益评估
        profit_df = self._profit_df(signal_data)
        # 最大收益评估
        up_space_df = self._profit_df(signal_data.drop("return", axis=1).rename(columns={"upside_ret": "return"}))
        # 最大风险评估
        down_space_df = self._profit_df(signal_data.drop("return", axis=1).rename(columns={"downside_ret": "return"}))

        # profit, up_space和down_sapce里计算的因子是没做过变换的原因子
        output = namedtuple("Output", ["coef", "stability", "profit", "up_space", "down_space", "full_report"])

        return output(
            coef=coef_df, stability=stability_df,
            profit=profit_df.rename("收益").to_frame(),
            up_space=up_space_df.rename("潜在收益").to_frame(),
            down_space=down_space_df.rename("潜在风险").to_frame(),
            full_report=self.create_full_report() if calc_full_report else None,
        )

    def _profit_df(self, signal_data):

        def positive_negative_signal(df):
            assert (df["signal"] > 0).all() or (df["signal"] < 0).all(), "the signal have different signs"
            weight_sign = "+" if (df["signal"] > 0).all() else "-"
            weighted = df.groupby(level=0).apply(lambda x: np.average(x["return"], weights=x["signal"].abs())).\
                rename("weighted")
            simple = df["return"].groupby(level=0).mean().rename("simple")
            return pd.concat([weighted, simple], axis=1), weight_sign

        df1, weight1 = positive_negative_signal(signal_data[signal_data["signal"] > 0])
        df2, weight2 = positive_negative_signal(signal_data[signal_data["signal"] < 0])

        def top_bottom_signal(df, way="bins"):
            top = df.groupby(level=0).apply(lambda x: x[x[way] == 5]["return"].mean()).rename("top")
            bottom = df.groupby(level=0).apply(lambda x: x[x[way] == 1]["return"].mean()).rename("bottom")
            return pd.concat([top, bottom], axis=1)

        q_way = top_bottom_signal(signal_data, way="quantile") if "quantile" in signal_data.columns else None
        b_way = top_bottom_signal(signal_data)

        series1 = self.pairwise_describe_and_mean_difference(df1, ["正signal: ", "加权", "简单"])
        series1 = series1.append(pd.Series(weight1, index=pd.MultiIndex.from_tuples([("正signal", "signal符号")])))
        if df2.size != 0:
            series2 = self.pairwise_describe_and_mean_difference(df2, ["负signal: ", "加权", "简单"])
            series2 = series2.append(pd.Series(weight2, index=pd.MultiIndex.from_tuples([("负signal", "signal符号")])))
        else:
            series2 = None

        series3 = self.pairwise_describe_and_mean_difference(b_way, ["bins: ","top", "bottom"])
        if q_way is not None:
            series4 = self.pairwise_describe_and_mean_difference(q_way, ["quantile: ", "top", "bottom"])
        else:
            series4 = None

        return pd.concat([series1, series2, series3, series4])

    @staticmethod
    def _stability_df_transform(df, pvalue_threshold=0.05):
        """
        input dataframe should include [value, p]
        """
        name_column, p_column = df.columns
        series = pd.Series(name=name_column)
        length = df.shape[0]
        df = df[df[p_column] <= pvalue_threshold].drop(p_column, axis=1).squeeze(1)
        ratio = (df.shape[0])/length
        series["正相关显著比例"] = (df > 0).mean()
        series["负相关显著比例"] = (df < 0).mean()
        series["同向显著次数占比"] = ((df * df.shift(-1)).dropna() > 0).mean()
        series["状态切换次数占比"] = ((df * df.shift(-1)).dropna() < 0).mean()
        series["显著比例较高的方向"] = "+" if series["正相关显著比例"] > series["负相关显著比例"] else "-"
        series["abs(正-负)"] = abs((series["正相关显著比例"] - series["负相关显著比例"]))
        series["同向-切换"] = series["同向显著次数占比"] - series["状态切换次数占比"]
        series[series.apply(np.isreal)] *= ratio
        return series

    @staticmethod
    def two_column_rank_ic(ret, factor):
        temp = spearmanr(ret, factor)
        return temp.correlation, temp.pvalue

    @staticmethod
    def single_series_describe(series, first_level_index):
        new_series = series.describe().loc[["mean", "std"]].rename({"mean": "均值", "std": "标准差"})
        new_series.loc["均值/标准差"] = new_series["均值"]/new_series["标准差"]
        new_series.index = pd.MultiIndex.from_product([[first_level_index], new_series.index])
        return new_series

    def pairwise_describe_and_mean_difference(self, df, first_level_index):
        """
        first_level_index should be in forms of ["正signal:","加权","简单"]
        """
        ratio = (1.0 * common.CALENDAR_CONST.TRADE_DAYS_PER_YEAR / self.period)

        assert len(first_level_index) == 3
        # self.error, self.error2 = df, first_level_index
        to_concated1 = self.single_series_describe(df.iloc[:, 0],
                                                   first_level_index=first_level_index[0]+first_level_index[1])
        to_concated2 = self.single_series_describe(df.iloc[:, 1],
                                                   first_level_index=first_level_index[0]+first_level_index[2])
        series = df.iloc[:, 0] - df.iloc[:, 1]
        mean, std = series.mean() * ratio, series.std() * np.sqrt(ratio)
        t, p = ttest_1samp(series, 0)
        to_concated3 = pd.Series([mean, std, mean/std, t, p],
                                 index=pd.MultiIndex.from_product(
                                 [[first_level_index[0]+first_level_index[1]+"-"+first_level_index[2]],
                                  ["均值", "标准差", "均值/标准差", "t值", "p值"]]))
        return pd.concat([to_concated1, to_concated2, to_concated3])

    def create_returns_report(self):
        """
        Creates a tear sheet for returns analysis of a signal.

        """
        n_quantiles = self.signal_data['quantile'].max()

        # ----------------------------------------------------------------------------------
        # Daily Signal Return Time Series
        # Use regression or weighted average to calculate.
        period_wise_long_ret = \
            pfm.calc_period_wise_weighted_signal_return(self.signal_data, weight_method='long_only')
        period_wise_short_ret = \
            pfm.calc_period_wise_weighted_signal_return(self.signal_data, weight_method='short_only')
        cum_long_ret = pfm.period_wise_ret_to_cum(period_wise_long_ret, period=self.period, compound=True)
        cum_short_ret = pfm.period_wise_ret_to_cum(period_wise_short_ret, period=self.period, compound=True)
        # period_wise_ret_by_regression = perf.regress_period_wise_signal_return(signal_data)
        # period_wise_ls_signal_ret = \
        #     pfm.calc_period_wise_weighted_signal_return(signal_data, weight_method='long_short')
        # daily_ls_signal_ret = pfm.period2daily(period_wise_ls_signal_ret, period=period)
        # ls_signal_ret_cum = pfm.daily_ret_to_cum(daily_ls_signal_ret)

        # ----------------------------------------------------------------------------------
        # Period-wise Quantile Return Time Series
        # We calculate quantile return using equal weight or market value weight.
        # Quantile is already obtained according to signal values.

        # quantile return
        period_wise_quantile_ret_stats = pfm.calc_quantile_return_mean_std(self.signal_data, time_series=True)
        cum_quantile_ret = pd.concat({k: pfm.period_wise_ret_to_cum(v['mean'], period=self.period, compound=True)
                                      for k, v in period_wise_quantile_ret_stats.items()},
                                     axis=1)

        # top quantile minus bottom quantile return
        period_wise_tmb_ret = pfm.calc_return_diff_mean_std(period_wise_quantile_ret_stats[n_quantiles],
                                                            period_wise_quantile_ret_stats[1])
        cum_tmb_ret = pfm.period_wise_ret_to_cum(period_wise_tmb_ret['mean_diff'], period=self.period, compound=True)

        self.returns_report_data = {'period_wise_quantile_ret': period_wise_quantile_ret_stats,
                                    'cum_quantile_ret': cum_quantile_ret,
                                    'cum_long_ret': cum_long_ret,
                                    'cum_short_ret': cum_short_ret,
                                    'period_wise_tmb_ret': period_wise_tmb_ret,
                                    'cum_tmb_ret': cum_tmb_ret}

    def create_information_report(self):
        """
        Creates a tear sheet for information analysis of a signal.

        """
        ic = pfm.calc_signal_ic(self.signal_data)
        ic.index = pd.to_datetime(ic.index, format="%Y%m%d")
        monthly_ic = pfm.mean_information_coefficient(ic, "M")
        self.ic_report_data = {'daily_ic': ic,
                               'monthly_ic': monthly_ic}

    def create_full_report(self):
        """
        Creates a full tear sheet for analysis and evaluating single
        return predicting (alpha) signal.

        """
        # signal quantile description statistics
        self.create_returns_report()
        self.create_information_report()
        # we do not do turnover analysis for now
        # self.create_turnover_report(signal_data)

        res = dict()
        res.update(self.returns_report_data)
        res.update(self.ic_report_data)
        return res

