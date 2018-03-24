# from jaqs.data import DataView  # 可以视为一个轻量级的数据库，数据格式基于pandas，方便数据的调用和处理
# from jaqs.data import RemoteDataService  # 数据服务，用于下载数据
from jaqs.research import SignalDigger
from jaqs.research.signaldigger import performance as pfm
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
import statsmodels.api as sm
from scipy.stats import spearmanr, ttest_1samp
from functools import partial
from collections import namedtuple
from jaqs.trade import common


class Dimensions:

    def __init__(self, dv, signal_data, industry_standard, period):
        self.signal_data = signal_data
        self.sw1 = industry_standard
        self.cap = dv.get_ts("float_mv").apply(np.log)  # TODO not sure about this indicator
        self.period = period
        self.signal_series = pd.Series()

        self.methods = {
            "mad": self._mad,
            "barra": partial(self._barra, cap_series=self.cap.stack()),
            "standard_scale": lambda x: pd.Series(scale(x), index=x.index, name="signal"),
        }

    def __call__(self, signal_data=None, regression_preprocessing=("mad",),
                 regression_method="wls", regression_weights="capitalization",
                 rank_ic_preprocessing=("mad", "neutralization_both", "standard_scale"),
                 p_threshold=0.05):

        """
        rank_ic_preprocesssing 支持neutralization_both, neutralization_cap, neutralization_industry
        """

        if signal_data is None:
            signal_data = self.signal_data
        self.regression_preprocessing = regression_preprocessing
        if regression_method == "ols":
            self.regression_method = regression_method
        elif regression_method == "wls":
            self.regression_method = regression_method
            self.regression_weights = regression_weights
        else:
            raise ValueError("regression method should be ols or wls")
        self.rank_ic_preprocessing = rank_ic_preprocessing
        self.p_threshold = p_threshold

        result = self._result(signal_data)

        def _restructure(s):
            return pd.Series(s, index=pd.MultiIndex.from_tuples(s.index)).rename("signal")
        self.signal_series = _restructure(self.signal_series)

        coef_df = pfm.calc_ic_stats_table(result[["回归系数", "IC", "最大回报IC", "最低回报IC"]]).rename(
            columns={"IC Mean": "Mean", "IC Std.": "Std.", "t-stat(IC)": "t-stat",
                     "p-value(IC)": "p-value", "IC Skew": "Skew", "IC Kurtosis": "Kurtosis"})

        stability_df = pd.concat(map(self._stability_df_transform,
                                     (result[["回归系数", "回归系数 p值"]], result[["IC", "IC p值"]],
                        result[["最大回报IC", "最大回报IC p值"]], result[["最低回报IC", "最低回报IC p值"]])), axis=1)

        signal_data.update(self.signal_series)
        self.signal_data_after = signal_data

        try:
            signal_data["quantile"] = signal_data["signal"].groupby(level=0).apply(pd.qcut, q=5, labels=[1,2,3,4,5])
        except ValueError:
            print("quantile cut do not work")
        signal_data["bins"] = signal_data["signal"].groupby(level=0).apply(pd.cut, bins=5, labels=[1,2,3,4,5])

        # 下面先对因子做处理，可能要改，因为不确定能不能跟前面的整合
        # copy_signal_data = signal_data.copy()
        # copy_signal_data["signal"] = self.signal_for_ic
        # copy_signal_data["signal"] = copy_signal_data["signal"].groupby(level=0).transform(
        #     lambda x: self.combined_method(
        #         list(map(lambda y: self.methods[y], profit_signal_preprocessing)), x))

        profit_df = self._profit_df(signal_data)
        up_space_df = self._profit_df(signal_data.drop("return", axis=1).rename({"upside_ret": "return"}, axis=1))
        down_space_df = self._profit_df(signal_data.drop("return", axis=1).rename({"downside_ret": "return"}, axis=1))

        # profit, up_space和down_sapce里计算的因子是没做过变换的原因子
        output = namedtuple("Output", ["coef", "stability", "profit", "up_space", "down_space", "full_report"])

        return output(
            coef=coef_df, stability=stability_df,
            profit=profit_df.rename("收益").to_frame(),
            up_space=up_space_df.rename("潜在收益").to_frame(),
            down_space=down_space_df.rename("潜在风险").to_frame(),
            full_report= self.create_full_report()
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

    def _stability_df_transform(self, df, pvalue_threshold=0.05):
        """
        input dataframe should include [value, p] 
        """
        name_column, p_column = df.columns
        series = pd.Series(name=name_column)
        length = df.shape[0]
        df = df[df[p_column] <= pvalue_threshold].drop(p_column, axis=1).squeeze()
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

    def _result(self, df):
        grouped = df.groupby(level=0)
        return pd.DataFrame.from_records(
            (self._cross_sectional_function(date, dataframe) for date, dataframe in grouped),
            columns=["回归系数t值", "回归系数", "回归系数 p值", "IC", "IC p值",
                     "最大回报IC", "最大回报IC p值", "最低回报IC", "最低回报IC p值"]
        )

    # @staticmethod
    # def _mad(df):
    #     """
    #     这个原本的df应该是0轴为时间， 1轴为各股票
    #     """
    #     median = df.median(axis=1)
    #     mad = df.sub(median, axis="index").abs().median(axis=1)  # TODO axis not sure
    #     return df.clip(median-5*mad, median+5*mad, axis=0).apply(scale, axis=1)

    @staticmethod
    def _mad(series):
        median = series.median()
        mad = (series-median).abs().median()
        return series.clip(median-5*mad, median+5*mad).rename("signal")

    @staticmethod
    def _barra(series, cap_series):
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
        return first_step_factor.clip(-3.5, 3.5).rename("signal")

    def combined_method(self, methods: list, value):
        if len(methods) == 0:
            return value
        return methods.pop()(self.combined_method(methods, value))

    def _cross_sectional_function(self, date, df):
        """
        input dataframe should has two columns,[signal, return], cross setion data, so no multiindex
        ['signal', 'return', 'upside_ret', 'downside_ret']
        """
        df.index = df.index.droplevel(0)
        cap = self.cap.loc[date, :].rename("cap")
        industry_df = pd.get_dummies(self.sw1.loc[date, :].dropna())
        X = pd.concat([df, cap, industry_df], axis=1).dropna()

        upside_ret = X.pop("upside_ret")
        downside_ret = X.pop("downside_ret")
        ret = X.pop("return")
        signal_series = X["signal"]

        for method in self.rank_ic_preprocessing:
            if method in self.methods.keys():
                signal_series = self.methods[method](signal_series)
            elif method.startswith("neutralization"):
                X1 = X.copy()
                X1.update(signal_series)
                signal_series = self._ic_regression(X1, method)
                signal_series.index = pd.MultiIndex.from_product([[date], signal_series.index])
            else:
                pass

        # ic_neutral_way = [x for x in self.rank_ic_preprocessing if x.startswith("neutralization")]
        # if len(ic_neutral_way) != 0:
        #     ic_other_way = list(set(self.rank_ic_preprocessing)-set(ic_neutral_way))
        #     ic_other_way = list(map(lambda x: self.methods[x], ic_other_way))
        #     signal_ic = self._ic_regression(X, ic_neutral_way[0])
        #     signal_ic.index = pd.MultiIndex.from_product([[date], signal_ic.index])
        #     signal_ic = self.combined_method(ic_other_way, signal_ic)
        # else:
        #     ic_other_way = ic_neutral_way
        #     ic_other_way = list(map(lambda x: self.methods[x], ic_other_way))
        #     signal_ic = self.combined_method(ic_other_way, X["signal"])

        self.signal_series = self.signal_series.append(signal_series)

        X["signal"] = self.combined_method(
            list(map(lambda x: self.methods[x], self.regression_preprocessing)), X["signal"])

        if self.regression_method == "wls":
            wls_model = sm.WLS(ret, X, weights=X["cap"])
            regression_results = wls_model.fit()
        else:
            ols_model = sm.OLS(ret, X)
            regression_results = ols_model.fit()
        # ic = self._ic_calculator(ret, df, cap, industry_df)
        return regression_results.tvalues[0], regression_results.params[0], regression_results.pvalues[0],\
               (*self.two_column_rank_ic(ret, signal_series)), \
               (*self.two_column_rank_ic(upside_ret, signal_series)), \
               (*self.two_column_rank_ic(downside_ret, signal_series))
        # TODO not sure about the pvalues

    @staticmethod
    def two_column_rank_ic(ret, factor):
        temp = spearmanr(ret, factor)
        return temp.correlation, temp.pvalue

    def _ic_regression(self, X, ic_neutral_way="neutralization_both"):
        """
        先对
        """
        independent_variable = X.iloc[:, 1:] if ic_neutral_way == "neutralization_both" else \
            X.iloc[:, 2:] if ic_neutral_way == "neutralization_industry" else sm.add_constant(X.iloc[:, 1])
        # self.jesus = (X["signal"], independent_variable)
        model = sm.OLS(X["signal"], independent_variable).fit()

        return model.resid

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

        assert len(first_level_index)==3
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
        qstb = calc_quantile_stats_table(self.signal_data)

        self.create_returns_report()
        self.create_information_report()
        # we do not do turnover analysis for now
        # self.create_turnover_report(signal_data)

        res = dict()
        res.update(self.returns_report_data)
        res.update(self.ic_report_data)
        return res


class Evaluator:

    """
    可以开个接口来操作是否对因子做一些中性化什么的
    """

    def __init__(self, dv, signal):
        self.dv = dv
        self.signal = signal

    def _generate_data(self, signal, period, benchmark):
        # 定义信号过滤条件-非指数成分
        def mask_index_member():
            df_index_member = self.dv.get_ts('index_member')
            mask_index_member = df_index_member == 0
            return mask_index_member

        # 定义可买卖条件——未停牌、未涨跌停
        def limit_up_down():
            trade_status = self.dv.get_ts('trade_status')
            mask_sus = trade_status == u'停牌'
            # 涨停
            self.dv.add_formula('up_limit', '(close - Delay(close, 1)) / Delay(close, 1) > 0.095',
                                is_quarterly=False, add_data=True)
            # 跌停
            self.dv.add_formula('down_limit', '(close - Delay(close, 1)) / Delay(close, 1) < -0.095',
                                is_quarterly=False, add_data=True)
            can_enter = np.logical_and(self.dv.get_ts('up_limit') < 1, ~mask_sus)  # 未涨停未停牌
            can_exit = np.logical_and(self.dv.get_ts('down_limit') < 1, ~mask_sus)  # 未跌停未停牌
            return can_enter, can_exit

        mask = mask_index_member()
        can_enter, can_exit = limit_up_down()

        obj = SignalDigger(output_folder='./output',
                           output_format='pdf')

        # 处理因子 计算目标股票池每只股票的持有期收益，和对应因子值的quantile分类
        obj.process_signal_before_analysis(signal=signal,
                                           price=self.dv.get_ts("close_adj").reindex_like(signal),
                                           high=self.dv.get_ts("high_adj").reindex_like(signal),  # 可为空
                                           low=self.dv.get_ts("low_adj").reindex_like(signal),  # 可为空
                                           group=None,  # 可为空
                                           n_quantiles=5,  # quantile分类数
                                           mask=mask,  # 过滤条件
                                           can_enter=can_enter,  # 是否能进场
                                           can_exit=can_exit,  # 是否能出场
                                           period=period,  # 持有期
                                           benchmark_price=benchmark,  # 基准价格 可不传入，持有期收益（return）计算为绝对收益
                                           commission=0.0008,
                                           )
        return obj.signal_data

    def __call__(self, period, comp=None,
                 industry_standard="sw1", industry=None,
                 time=None, benchmark=None):

        data = self._generate_data(self.signal, period=period, benchmark=benchmark).drop("quantile", axis=1)

        def del_all_zero_date(df):
            index = df["return"].groupby(level=0).transform(lambda x: x if (x != 0).all() else None).dropna().index
            return df.reindex(index)
        data = del_all_zero_date(data)

        if isinstance(industry_standard, str):
            industry_standard = self.dv.get_ts(industry_standard)
        elif isinstance(industry, pd.DataFrame):
            industry_standard = industry_standard
        elif isinstance(industry_standard, pd.Series):
            industry_standard = industry_standard.unstack()
        else:
            raise ValueError("industry_standard should be one of str, DataFrame and Series")

        if comp is not None:  # ("hs300", "zz500")
            if comp == "hs300":
                member = self.dv.get_ts('000300.SH_member').stack()
                data = data[member == 1]
            elif comp == "zz500":
                member = self.dv.get_ts('000905.SH_member').stack()
                data = data[member == 1]
            elif isinstance(comp, pd.DataFrame):
                data = data[comp.stack() == 1]
            elif isinstance(comp, pd.Series):
                data = data[comp == 1]
            else:
                raise ValueError("only support 'hs300', 'zz500' now")

        if industry is not None:
            assert isinstance(industry, list), "industry should be a list"
            data = data[industry_standard.stack().isin(industry)]

        if time is not None:
            assert isinstance(time, list), "time should be a list of tuple, e.g. [('20170101', '20170201')]"
            data = pd.concat(data.loc[(slice(start_time, end_time),), :] for start_time, end_time in time)

        return Dimensions(self.dv, data, industry_standard=industry_standard, period=period)

    def _style_factor(self):
        pass


# if __name__ == "__main__":
#     warnings.filterwarnings("ignore")
#     dataview_folder = './Factor'
#
#     if not (os.path.isdir(dataview_folder)):
#         os.makedirs(dataview_folder)
#
#     # 数据下载
#     def save_dataview():
#         data_config = {
#             "remote.data.address": "tcp://data.tushare.org:8910",  # "tcp://192.168.0.102:23000"
#             "remote.data.username": "18566262672",
#             "remote.data.password": "eyJhbGciOiJIUzI1NiJ9.eyJjcmVhdGVfdGltZSI6IjE1MTI3MDI3NTAyMTIiLCJpc3MiOiJhdXRoMCIsImlkIjoiMTg1NjYyNjI2NzIifQ.O_-yR0zYagrLRvPbggnru1Rapk4kiyAzcwYt2a3vlpM"
#         }
#         ds = RemoteDataService()
#         ds.init_from_config(data_config)
#
#         dv = DataView()
#         props = {'start_date': 20140101, 'end_date': 20171001, 'universe': '000300.SH',
#                  'fields': "pb,pe,ps,float_mv,sw1",
#                  'freq': 1}
#         dv.add_comp_info("000300.SH", ds)
#
#         dv.init_from_config(props, ds)
#         dv.prepare_data()
#         dv.save_dataview(dataview_folder)  # 保存数据文件到指定路径，方便下次直接加载

def calc_quantile_stats_table(signal_data):
    quantile_stats = signal_data.groupby('quantile').agg(['min', 'max', 'mean', 'std', 'count'])['signal']
    quantile_stats['count %'] = quantile_stats['count'] / quantile_stats['count'].sum() * 100.
    return quantile_stats