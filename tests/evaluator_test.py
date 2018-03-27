import jaqs_fxdayu
jaqs_fxdayu.patch_all()
from jaqs.data import DataView
from jaqs.data import RemoteDataService # 远程数据服务类
from fxdayu_betaman.evaluator import Evaluator

# step 1 其中,username password分别对应官网注册的账号和序列号
data_config = {
"remote.data.address": "tcp://192.168.0.102:23000",
# "remote.data.address": "tcp://data.tushare.org:8910",
"remote.data.username": "18566262672",
"remote.data.password": "eyJhbGciOiJIUzI1NiJ9.eyJjcmVhdGVfdGltZSI6IjE1MTI3MDI3NTAyMTIiLCJpc3MiOiJhdXRoMCIsImlkIjoiMTg1NjYyNjI2NzIifQ.O_-yR0zYagrLRvPbggnru1Rapk4kiyAzcwYt2a3vlpM"
}

# step 2
ds = RemoteDataService()
ds.init_from_config(data_config)
dv = DataView()
#
# step 3
props = {'start_date': 20170501, 'end_date': 20171001, 'universe':'000300.SH',
         'fields': "pb,float_mv,sw1",
         'freq': 1}

dv.init_from_config(props, ds)
dv.prepare_data()
# import os
# dataview_folder = './data'
#
# if not (os.path.isdir(dataview_folder)):
#     os.makedirs(dataview_folder)
#
# dv.save_dataview(dataview_folder)
# dv.load_dataview(dataview_folder)

evaluator = Evaluator(dv,dv.get_ts("pb"))
dms = evaluator(period=15,
                benchmark=None,
                commission=0.0008,
                industry_standard="sw1", # 行业标准
                cap="float_mv", # 流通市值
                time=[(20170601,20170901)], # 时间范围
                comp=dv.get_ts("index_member"), # 指数成分范围
                industry=['480000','430000'])

report = dms(regression_method="wls",
             preprocessing=("winsorize", "neutralization_both", "standard_scale"),
             p_threshold=0.05,
             n_quantiles=10,
             calc_full_report=True)

print(report.full_report)
