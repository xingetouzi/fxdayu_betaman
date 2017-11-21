import pandas as pd
#from fxdayu_betaman.calculator import BaseCalculator
from fxdayu_betaman.fxdayu_betaman.loader import FileLoader
import pyfolio as pf
import matplotlib.pyplot as plt
from datetime import datetime
#from fxdayu_betaman.calculator import analysis

if __name__ == "__main__":
    pd.set_option("display.width", 160)
    df = FileLoader("test.xlsx").load()
    print(df)
    #calculator = BaseCalculator(df, 1000000)
    #daily_returns = calculator.daily_returns
    #returns = daily_returns
    #b_r = calculator.benchmark_rets()
    # positions = calculator.daily_mv_df
    # print(positions)
    #transactions = calculator.transactions
    #positions_alloc = pf.pos.get_percent_alloc(positions)
    #symbol_sector_map = calculator.symbol_sector_map()
    #sector_alloc = analysis.sector_exposures(positions,symbol_sector_map)
    #market_data = calculator.market_data_panel()
    #print(calculator.daily_returns)
    #print(analysis.return_quantiles(daily_returns,datetime(2016,1,1)))
    #print(analysis.perf_stats(daily_returns,
                              #b_r,
                              #live_start_date=datetime(2016,1,1),
                              #bootstrap=True))

    #print(daily_returns,len(daily_returns))
    #print(b_r,len(b_r))
    # print(analysis.annual_returns(daily_returns))
    # print(analysis.holdings(calculator.daily_mv_df))
    # print(analysis.cum_rets(daily_returns))
    # print(analysis.drawdown_periods(daily_returns))
    # print (analysis.capacity_sweep(returns,transactions,market_data,1000000))
    #print(analysis.drawdown_underwater(daily_returns))
    #pf.plot_holdings(daily_returns,calculator.daily_mv_df)
    #pf.plot_drawdown_periods(daily_returns)
    #pf.plot_drawdown_underwater(daily_returns)
    #pf.plot_returns(daily_returns,live_start_date=datetime(2016,1,1))
    #pf.plot_gross_leverage(daily_returns,positions)
    #pf.plot_exposures(daily_returns, positions)
    #pf.plot_exposures(daily_returns, positions_alloc)
    #pf.pos.get_top_long_short_abs(positions)
    #pf.show_and_plot_top_positions(daily_returns,positions)
    #pf.plot_max_median_position_concentration(positions)
    #pf.plot_return_quantiles(daily_returns,live_start_date=datetime(2016,1,1))
    #pf.plot_turnover(returns, transactions, positions)
    #pf.plot_slippage_sweep(returns, transactions, positions)
    #pf.plot_slippage_sensitivity(returns, transactions, positions)
    #pf.show_worst_drawdown_periods(returns)
    #round_trips = pf.round_trips.extract_round_trips(transactions)
    #print(analysis.profit_attribution(round_trips))
    #print(analysis.prob_profit_trade(round_trips))
    #pf.plot_prob_profit_trade(round_trips)
    #plt.show()
    #for i in [True,False]:
        #pf.show_perf_stats(daily_returns,
                           #b_r,
                           #live_start_date=datetime(2016,1,1),bootstrap=i)
    # print(calculator.daily_market_value)
    #start = pd.Timestamp("2013-03-12 15:00:00")
    #end = pd.Timestamp("2013-03-15 15:00:00")
    #print(calculator.position_info_detail_by_time)
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
