from fxdayu_betaman.calculator.extra import GuoJinCalculator
from fxdayu_betaman.loader import FileLoader

if __name__ == "__main__":
    df = FileLoader("test.xlsx").load()
    print(df)
    calculator = GuoJinCalculator(df, 1000000)
    # print(calculator.position)
    # print(calculator.pending_position)
    # print(calculator.market_value)
    # print(calculator.pnl)
    # print(calculator.market_data)
    # print(calculator.position_by_time)
    print(calculator.get_order())
    print(calculator.get_position())
    print(calculator.get_risk_exp())
    calculator.save_position_csv()
    calculator.save_risk_exp_csv()
    # print(calculator.entry)
    # print(calculator.average_price)
    # print(calculator.market_data)