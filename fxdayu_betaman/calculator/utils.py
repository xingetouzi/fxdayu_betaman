#  _split_factor = SimpleFactorStore(_p('split_factor.bcolz'))

import bcolz
import numpy as np


class SimpleFactorStore(object):
    def __init__(self, f):
        table = bcolz.open(f, 'r')
        self._index = table.attrs['line_map']
        self._table = table[:]

    def get_factors(self, order_book_id):
        try:
            s, e = self._index[order_book_id]
            return self._table[s:e]
        except KeyError:
            return None

class DividendStore(object):
    def __init__(self, f):
        ct = bcolz.open(f, 'r')
        self._index = ct.attrs['line_map']
        self._table = np.empty((len(ct), ), dtype=np.dtype([
            ('announcement_date', '<u4'), ('book_closure_date', '<u4'),
            ('ex_dividend_date', '<u4'), ('payable_date', '<u4'),
            ('dividend_cash_before_tax', np.float), ('round_lot', '<u4')
        ]))
        self._table['announcement_date'][:] = ct['announcement_date']
        self._table['book_closure_date'][:] = ct['closure_date']
        self._table['ex_dividend_date'][:] = ct['ex_date']
        self._table['payable_date'][:] = ct['payable_date']
        self._table['dividend_cash_before_tax'] = ct['cash_before_tax'][:] / 10000.0
        self._table['round_lot'][:] = ct['round_lot']

    def get_dividend(self, order_book_id):
        try:
            s, e = self._index[order_book_id]
        except KeyError:
            return None

        return self._table[s:e]

# if __name__ == '__main__':
#     sf = SimpleFactorStore('C:\\Users\Tianhang\.rqalpha\\bundle\split_factor.bcolz')
#       ds = DividendStore('C:\\Users\Tianhang\.rqalpha\\bundle\original_dividends.bcolz')
#     sf.get_factors('000001.XSHE')