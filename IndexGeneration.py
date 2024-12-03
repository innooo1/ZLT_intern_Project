# -*- coding = utf-8 -*-
# @Time: 2024-11-27
# @Author: Zhang Zhiyan
# @File：Index_Generation.py
# @Desc:
# @Software: PyCharm


import pandas as pd
import numpy as np
from datetime import datetime
import os
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")


class DataHandler:
    """数据处理函数"""
    @staticmethod
    def process_dates(start_date, end_date):
        return pd.date_range(start=start_date, end=end_date).strftime('%Y%m%d').tolist()

    @staticmethod
    def get_eod_data():
        df_eod = pd.read_csv('input/A_EOD_from_20141020.csv')
        df_dvd = pd.read_csv('input/A_DVD_from_20141020.csv')
        df_dvd['TRADE_DT'] = df_dvd['EX_DT'] + 1

        df = pd.merge(df_eod, df_dvd, "left", on=('TRADE_DT', 'S_CODE'))
        df = df.fillna(0)
        df['adj_pre_close'] = df['pre_close'] - df['PRE_TAX']
        df = df.drop('PRE_TAX', axis='columns')
        return df

    @staticmethod
    def get_3s_data(date, stk_code):
        folder_path = 'original/{}'.format(date) #修改日内3s数据文件夹路径
        code, mkt = stk_code.split('.')
        file_path = os.path.join(folder_path, '{}_{}_{}.parquet'.format(mkt, code, date))
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            return df
        else:
            print(f"No data found for {stk_code} on {date}.")
            return pd.DataFrame()

    @staticmethod
    def convert_time_format(time_int):
        if len(str(time_int)) == 8:
            time_str = '0' + str(time_int)
        else:
            time_str = str(time_int)
        hour = time_str[:2]
        minute = time_str[2:4]
        second = time_str[4:6]
        return f"{hour}:{minute}:{second}"


class IndexDataProcessor:
    """wind指数逐日计算，以20141020为基期，1000点为基点"""
    def __init__(self,
                 code:str,
                 start_date: str,
                 end_date: str):
        self.code = code
        self.start_date = pd.to_datetime(start_date, format='%Y%m%d')
        self.end_date = pd.to_datetime(end_date, format='%Y%m%d')
        self.eod_df = DataHandler.get_eod_data()
        self.eod_df['TRADE_DT'] = pd.to_datetime(self.eod_df['TRADE_DT'].astype(str), format='%Y%m%d')

    def get_index_stock_weight(self):
        df = pd.read_csv(f'input/{self.code}_info.csv')
        df = df.loc[df['I_CODE'] == '8841388.WI'][['S_CODE', 'IN_DT', 'OUT_DT']]
        today_date = int(datetime.now().strftime('%Y%m%d'))
        df['OUT_DT'] = df['OUT_DT'].fillna(today_date + 1)
        df['IN_DT'] = pd.to_datetime(df['IN_DT'].astype(int).astype(str), format='%Y%m%d')
        df['OUT_DT'] = pd.to_datetime(df['OUT_DT'].astype(int).astype(str), format='%Y%m%d')

        dt_df = pd.read_csv('input/trade_date.csv')
        dt_df['TRADE_DAYS'] = pd.to_datetime(dt_df['TRADE_DAYS'].astype(int).astype(str), format='%Y%m%d')

        results = []
        filtered_dates = dt_df[(dt_df['TRADE_DAYS'] >= self.start_date) & (dt_df['TRADE_DAYS'] <= self.end_date)]
        target_dates = pd.DatetimeIndex(filtered_dates['TRADE_DAYS'])

        for date in target_dates:
            valid_stocks = df[(df['IN_DT'] <= date) & (df['OUT_DT'] > date)]
            valid_stocks = valid_stocks.drop_duplicates(subset=['S_CODE'])

            for _, row in valid_stocks.iterrows():
                results.append({'S_CODE': row['S_CODE'], 'TRADE_DT': date})

        result = pd.DataFrame(results)
        return result

    def calculate_total_return_index(self):
        stk_df = self.get_index_stock_weight()
        df_cal = pd.merge(stk_df, self.eod_df, "left", on=('TRADE_DT', 'S_CODE'))
        df_cal = df_cal[['TRADE_DT', 'S_CODE', 'pre_close', 'adj_pre_close', 'open', 'close']]
        df_cal['prc_rt'] = df_cal['close'] / df_cal['pre_close']
        df_cal['ttl_rtn_rt'] = df_cal['close'] / df_cal['adj_pre_close']

        result_df = df_cal.groupby('TRADE_DT').agg({
            'prc_rt': 'sum',
            'ttl_rtn_rt': 'sum'
        }).reset_index()

        result_df.loc[0, 'prc_index'] = 1000.0
        result_df.loc[0, 'ttl_rtn_index'] = 1000.0

        for i in range(1, len(result_df)):
            result_df.loc[i, 'prc_index'] = result_df.loc[i, 'prc_rt']\
                                              / result_df.loc[i - 1, 'prc_rt']\
                                              * result_df.loc[i - 1, 'prc_index']

            result_df.loc[i, 'ttl_rtn_index'] = result_df.loc[i, 'ttl_rtn_rt'] / \
                                                result_df.loc[i - 1, 'ttl_rtn_rt'] \
                                                * result_df.loc[i - 1, 'ttl_rtn_index']
        return result_df


class GetDivider:
    """
    除数计算过程：
    --> 获取成分股自由流通股数变化超5%日期
    --> 获取factor、成分股组成变化日期
    --> 利用收盘价和调整收盘价对应计算用股本计算市值
    --> 在变动日根据指数类型（价格/全收益）进行除数迭代
    """
    def __init__(self,
                 code: str,
                 eod: pd.DataFrame,
                 idx_type: int):
        self.eod_df = eod
        self.code = code
        self.idx_type = idx_type

    def get_stk_shr_changed_day(self):
        # 获取自由流通股变动超过5%的日期
        df = pd.read_csv('input/A_SHR.csv')
        df['CHANGE_DT'] = pd.to_datetime(df['CHANGE_DT'], format='%Y%m%d')
        date_range = pd.date_range(start='2000-01-01', end=pd.Timestamp('today'))
        stock_codes = df['S_CODE'].unique()

        new_index = pd.MultiIndex.from_product([stock_codes, date_range], names=['S_CODE', 'CHANGE_DT'])
        full_df = pd.DataFrame(index=new_index).reset_index()

        merged_df = pd.merge(full_df, df, on=['S_CODE', 'CHANGE_DT'], how='left')
        merged_df = merged_df.sort_values(by=['S_CODE', 'CHANGE_DT'])
        merged_df['SHR'] = merged_df['SHR'].ffill()

        final_df = merged_df[['CHANGE_DT', 'S_CODE', 'SHR']].fillna(0)
        final_df = final_df.rename(columns={'CHANGE_DT': 'TRADE_DT'})

        df_com = pd.read_csv(f'input/{self.code}_info.csv')
        df_com['TRADE_DT'] = pd.to_datetime(df_com['TRADE_DT'], format='%Y%m%d')
        df_merged = pd.merge(final_df, df_com, how='right', on=('S_CODE', 'TRADE_DT'))

        # calculation
        df_cal = df_merged[df_merged['TRADE_DT'] >= '20141020'][['TRADE_DT', 'S_CODE', 'SHR']]
        df_cal = df_cal.sort_values(by=['S_CODE', 'TRADE_DT'])
        df_cal['PREV_SHR'] = df_cal.groupby('S_CODE')['SHR'].shift(1)
        df_cal['SHR_RATIO'] = (df_cal['SHR'] - df_cal['PREV_SHR']) / df_cal['PREV_SHR']

        df_cal['stk_changed'] = 0
        df_cal.loc[df_cal['SHR_RATIO'].abs() > 0.05, 'stk_changed'] = 1
        df_cal = df_cal[['TRADE_DT', 'stk_changed']].drop_duplicates().sort_values(by=['TRADE_DT'])
        df_cal = df_cal.loc[df_cal.groupby('TRADE_DT')['stk_changed'].idxmax()].reset_index()

        return df_cal

    def get_initial_value(self):
        if self.idx_type == 1:
            base_value_list = {'000300.SH': 2454.71, '000852.SH': 6154.52}  # 2014.10.20
        else:
            base_value_list = {'000300.SH': 2825.99, '000852.SH': 6415.20}

        base_value = base_value_list[self.code]
        df = pd.read_csv('input/{}_info.csv'.format(self.code))
        df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'].astype(str), format='%Y%m%d')
        df = df[df['TRADE_DT'] >= '20141020'][['TRADE_DT', 'S_CODE', 'FACTOR', 'STK_NUM', 'CAL_VALUE']]

        df = pd.merge(df, self.eod_df, "left", on=('TRADE_DT', 'S_CODE'))

        # calculation
        df = df.sort_values(by=['S_CODE', 'TRADE_DT'])
        df['PREV_STK'] = df.groupby('S_CODE')['STK_NUM'].shift(1)
        df['STK_RATIO'] = (df['STK_NUM'] - df['PREV_STK']) / df['PREV_STK']

        df['PREV_F'] = df.groupby('S_CODE')['FACTOR'].shift(1)
        df['F_diff'] = df['FACTOR'] - df['PREV_F']

        df['PRC_VALUE'] = df['STK_NUM'] * df['pre_close'] * df['FACTOR']
        df['ADJ_VALUE'] = df['STK_NUM'] * df['adj_pre_close'] * df['FACTOR']

        # find the changed day caused by stock_num
        df_stk_adj = self.get_stk_shr_changed_day()

        # find the stock adjusted day
        df_com_adj = df.groupby('TRADE_DT')['S_CODE'].agg(lambda x: frozenset(x)).reset_index()
        df_com_adj['com_changed'] = (df_com_adj['S_CODE'] != df_com_adj['S_CODE'].shift(1)).astype(int)

        # find the factor changed day
        df_factor_adj = df.copy()
        df_factor_adj['fct_changed'] = 0
        df_factor_adj.loc[df_factor_adj['F_diff'].abs() > 0, 'fct_changed'] = 1
        df_factor_adj = df_factor_adj[['TRADE_DT', 'fct_changed']].drop_duplicates().sort_values(by=['TRADE_DT'])
        df_factor_adj = df_factor_adj.loc[df_factor_adj.groupby('TRADE_DT')['fct_changed'].idxmax()]

        # get the trade value
        df_v = df.groupby('TRADE_DT').agg({'PRC_VALUE': 'sum',
                                           'ADJ_VALUE': 'sum'}).reset_index()

        # get the changed day
        df_adj1 = pd.merge(df_stk_adj, df_com_adj, "outer", on='TRADE_DT')
        df_adj1['stk_com_changed'] = df_adj1[['stk_changed', 'com_changed']].max(axis=1)
        df_adj1 = df_adj1[['TRADE_DT', 'stk_com_changed']]
        df_adj = pd.merge(df_adj1, df_factor_adj, "outer", on='TRADE_DT')
        df_adj['changed'] = df_adj[['stk_com_changed', 'fct_changed']].max(axis=1)

        df_cal = pd.merge(df_v, df_adj, 'left', on='TRADE_DT')[['TRADE_DT', 'PRC_VALUE', 'ADJ_VALUE', 'changed']]

        # calculate the div
        df_cal['DIV'] = 0.0
        if self.idx_type == 1:
            df_cal.loc[0, 'DIV'] = (df_cal.loc[0, 'PRC_VALUE'] / base_value) * 1000
            for i in range(1, len(df_cal)):
                if df_cal.loc[i, 'changed'] == 0:
                    df_cal.loc[i, 'DIV'] = df_cal.loc[i - 1, 'DIV']
                else:
                    df_cal.loc[i, 'DIV'] = df_cal.loc[i, 'PRC_VALUE'] / df_cal.loc[i - 1, 'PRC_VALUE'] * df_cal.loc[
                        i - 1, 'DIV']
        else:
            df_cal.loc[0, 'DIV'] = (df_cal.loc[0, 'ADJ_VALUE'] / base_value) * 1000
            for i in range(1, len(df_cal)):
                if df_cal.loc[i, 'changed'] == 0:
                    df_cal.loc[i, 'DIV'] = df_cal.loc[i - 1, 'DIV']
                else:
                    df_cal.loc[i, 'DIV'] = df_cal.loc[i, 'ADJ_VALUE'] / df_cal.loc[i - 1, 'ADJ_VALUE'] * df_cal.loc[
                        i - 1, 'DIV']
        df_cal = df_cal[['TRADE_DT', 'DIV']]

        return df_cal


class IndexGeneration():
    """
    指数生成过程：
    --> 获取目标区间成分股
    --> 计算目标区间内每个交易日指数
    --> 区分自由流通市值加权和等权
    --> 区分价格指数与全收益指数
    --> 输出预测结果
    """
    def __init__(self,
                 code: str,
                 idx_type: int,
                 start_date: str,
                 end_date: str,
                 eod,
                 div):
        self.code = code
        self.idx_type = idx_type
        self.start_date = pd.to_datetime(start_date, format='%Y%m%d')
        self.end_date = pd.to_datetime(end_date, format='%Y%m%d')
        self.eod_df = eod
        self.div_df = div
        self.components = None

    def load_components(self):
        self.components = self.get_index_stock_weight()

    def get_index_stock_weight(self):
        df = pd.read_csv('input/{}_info.csv'.format(self.code))

        if self.code == '8841388.WI':
            df = df.loc[df['I_CODE'] == '8841388.WI'][['S_CODE', 'IN_DT', 'OUT_DT']]
            today_date = int(datetime.now().strftime('%Y%m%d'))
            df['OUT_DT'] = df['OUT_DT'].fillna(today_date + 1)
            df['IN_DT'] = pd.to_datetime(df['IN_DT'].astype(int).astype(str), format='%Y%m%d')
            df['OUT_DT'] = pd.to_datetime(df['OUT_DT'].astype(int).astype(str), format='%Y%m%d')

            # 获取交易日信息
            dt_df = pd.read_csv('input/trade_date.csv')
            dt_df['TRADE_DAYS'] = pd.to_datetime(dt_df['TRADE_DAYS'].astype(int).astype(str), format='%Y%m%d')

            # 获取目标区间内每个交易日的成分股信息
            results = []
            filtered_dates = dt_df[(dt_df['TRADE_DAYS'] >= self.start_date) & (dt_df['TRADE_DAYS'] <= self.end_date)]
            target_dates = pd.DatetimeIndex(filtered_dates['TRADE_DAYS'])

            for date in target_dates:
                valid_stocks = df[(df['IN_DT'] <= date) & (df['OUT_DT'] > date)]
                valid_stocks = valid_stocks.drop_duplicates(subset=['S_CODE'])

                for _, row in valid_stocks.iterrows():
                    results.append({
                        'S_CODE': row['S_CODE'],
                        'TRADE_DT': date
                    })
            result = pd.DataFrame(results)

        else:
            df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'].astype(str), format='%Y%m%d')
            result = df[(df['TRADE_DT'] >= self.start_date) & (df['TRADE_DT'] <= self.end_date)]
            result = result[['S_CODE', 'TRADE_DT', 'STK_NUM', 'FACTOR']]

        return result

    def calculate_index_mkt_weight(self, date):
        stk_df = self.components[self.components['TRADE_DT'] == date].reset_index(drop=True)
        contributions = []
        self.eod_df = self.eod_df[self.eod_df['TRADE_DT'] == date]

        rnt_df = pd.DataFrame()
        for _, row in stk_df.iterrows():
            stock_code = row['S_CODE']
            stk_num = row['STK_NUM']
            factor = row['FACTOR']
            stk_trade_df = DataHandler.get_3s_data(date, stock_code)
            if stk_trade_df is None or stk_trade_df.empty:
                continue

            # 排除异常行情计算每3s的收益率和max，min点位
            sub_stk_df = stk_trade_df[['resample_time', 'last_prc']]
            sub_stk_df.loc[:, 'cal_prc'] = sub_stk_df['last_prc'] * stk_num * factor
            time_df = sub_stk_df[['resample_time']].rename(columns={'resample_time': 'time'})
            sub_stk_df = sub_stk_df[['cal_prc']].rename(columns={'cal_prc': stock_code})

            # 拼接last_prc
            if rnt_df.empty:
                rnt_df = sub_stk_df
            else:
                rnt_df = pd.concat([rnt_df, sub_stk_df], axis=1)

            # 正常计算volume, turnover
            if self.idx_type == 2:
                stk_adj_pre_close = self.eod_df.loc[self.eod_df['S_CODE'] == stock_code, 'adj_pre_close'].values[0]
                stk_trade_df['prev_close'] = stk_adj_pre_close

            cal_data = {
                'time': stk_trade_df['resample_time'],
                'ttl_prev_close': stk_trade_df['prev_close'] * stk_num * factor,
                'ttl_open': stk_trade_df['open'] * stk_num * factor,
                'volume': stk_trade_df['volume'],
                'turnover': stk_trade_df['turnover']
            }
            contributions.append(pd.DataFrame(cal_data))

        if rnt_df.empty:
            print('Warn!!! The result is empty.')
            return pd.DataFrame()
        elif not contributions:
            print('Warn!!! The result is empty.')
            return pd.DataFrame()

        rnt_df = pd.concat([time_df, rnt_df], axis=1)
        rnt_df['total_sum'] = rnt_df.iloc[:, 1:].sum(axis=1)
        fst_prc = rnt_df.loc[0, 'total_sum']
        part_sum = []
        for i in range(len(rnt_df)):
            if i == 0:
                # 第一行的 sum1 等于 sum2
                part_sum.append(rnt_df['total_sum'].iloc[i])
            else:
                # 计算上一行不为0的值该行的和
                previous_row = rnt_df.iloc[i - 1, 1:-1]  # 第 i-1 行的所有股票代码列
                non_zero_columns = previous_row[previous_row != 0].index
                current_sum = rnt_df.loc[i, non_zero_columns].sum()  # 对这些列的第 i 行进行加总
                part_sum.append(current_sum)

        rnt_df['part_sum'] = part_sum
        rnt_df['3s_rnt_rate'] = rnt_df['part_sum'] / rnt_df['total_sum'].shift(1)
        rnt_df.loc[0, '3s_rnt_rate'] = 1

        rnt_df['rnt_rate'] = rnt_df['3s_rnt_rate'].cumprod()
        rnt_df['high'] = rnt_df['rnt_rate'].copy()
        rnt_df['low'] = rnt_df['rnt_rate'].copy()

        # 逐行更新high和low
        for i in range(len(rnt_df)):
            if i > 0:
                rnt_df.loc[i, 'high'] = max(rnt_df.loc[i - 1, 'high'], rnt_df.loc[i, 'rnt_rate'])
                rnt_df.loc[i, 'low'] = min(rnt_df.loc[i - 1, 'low'], rnt_df.loc[i, 'rnt_rate'])

        result_1 = rnt_df[['time', '3s_rnt_rate', 'rnt_rate', 'high', 'low']]

        contributions_df = pd.concat(contributions, ignore_index=True)
        result_2 = contributions_df.groupby('time').agg({
            'ttl_prev_close': 'sum',
            'ttl_open': 'sum',
            'volume': 'sum',
            'turnover': 'sum'
        }).reset_index()
        result_2.rename(columns={
            'ttl_prev_close': 'prev_close',
            'ttl_open': 'open'
        }, inplace=True)
        # result['3s_return_rate'] = (result['last_prc'] / result['last_prc'].shift(1).fillna(1)) - 1
        date = pd.to_datetime(date, format='%Y%m%d')
        div = self.div_df.loc[self.div_df['TRADE_DT'] == date, 'DIV'].values[0]

        open = result_2.loc[len(result_2) - 1, 'open']
        result_2['ori_open'] = open / div
        result_2['open'] = 1.0
        result_2['prev_close'] = result_2.loc[len(result_2) - 1, 'prev_close'] / open
        # result.iloc[:, 1:6] = ((result.iloc[:, 1:6] / div) * 1000).round(4)

        result = pd.merge(result_1, result_2, "left", on='time').reset_index()
        result['close'] = result.loc[len(result) - 1, 'rnt_rate']
        result = result[
            ['time', '3s_rnt_rate', 'rnt_rate', 'prev_close', 'open', 'close', 'high', 'low', 'volume', 'turnover']]

        return result

    def get_pre_index(self):
        if self.idx_type == 1:
            df = pd.read_csv('input/8841388.WI_EOD_from_19991231.csv')[['TRADE_DT','preclose']]
            df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'].astype(str), format='%Y%m%d')
        else:
            df = pd.read_csv('output/8841388.WI_pre_index_from_20141020.csv')
            df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'].astype(str), format='%Y-%m-%d')
        return df

    def calculate_index_equal_weight(self, date):
        stk_df = self.components[self.components['TRADE_DT'] == date].reset_index(drop=True)
        pre_index_df = self.get_pre_index()
        date_c = pd.to_datetime(date, format='%Y%m%d')
        self.eod_df = self.eod_df[self.eod_df['TRADE_DT'] == date_c]

        # 获取当日各成分股的高频数据
        contributions = []
        stk_num = 0
        for _, row in stk_df.iterrows():
            stock_code = row['S_CODE']
            stk_trade_df = DataHandler.get_3s_data(date, stock_code)
            if stk_trade_df.shape[0] != 4744:
                print(stock_code, ': data has missing value')
                continue

            if stk_trade_df.loc[0, 'open'] != stk_trade_df.loc[3000, 'open']:
                print(stock_code, ': open data has missing value')
                continue

            if stk_trade_df.loc[0, 'last_prc'] == 0:
                print(stock_code, ': last_prc data has missing value')
                continue

            if stk_trade_df is None or stk_trade_df.empty:
                continue

            if self.idx_type == 2:
                stk_adj_pre_close = self.eod_df.loc[self.eod_df['S_CODE'] == stock_code, 'adj_pre_close'].values[0]
                stk_trade_df['prev_close'] = stk_adj_pre_close

            stk_num += 1
            cal_data = {
                'time': stk_trade_df['resample_time'],
                # '3s_data': stk_trade_df['last_prc'],
                'prev_close': stk_trade_df['prev_close'] / stk_trade_df['prev_close'],
                'open': stk_trade_df['open'] / stk_trade_df['prev_close'],
                'high': stk_trade_df['high'] / stk_trade_df['prev_close'],
                'low': stk_trade_df['low'] / stk_trade_df['prev_close'],
                'last_prc': stk_trade_df['last_prc'] / stk_trade_df['prev_close'],
                'volume': stk_trade_df['volume'],
                'turnover': stk_trade_df['turnover']
            }

            contributions.append(pd.DataFrame(cal_data))

        if not contributions:
            print('Warning!!! The result is empty.')
            return pd.DataFrame()

        contributions_df = pd.concat(contributions, ignore_index=True)
        result = contributions_df.groupby('time').agg({
            # '3s_data': 'sum',
            'open': 'sum',
            'high': 'sum',
            'low': 'sum',
            'last_prc': 'sum',
            'volume': 'sum',
            'turnover': 'sum'
        }).reset_index()

        # result['3s_return_rate'] = (result['last_prc'] / result['last_prc'].shift(1).fillna(1)) - 1

        if self.idx_type == 1:
            pre_close = pre_index_df.loc[pre_index_df['TRADE_DT'] == date, 'preclose'].iloc[0]
        else:
            pre_close = pre_index_df.loc[pre_index_df['TRADE_DT'] == str(int(date) - 1), 'ttl_rtn_index'].iloc[0]

        result.iloc[:, 1:6] = ((result.iloc[:, 1:6] / stk_num) * pre_close).round(4)
        result['close'] = result.loc[len(result) - 1, 'last_prc']
        result['open'] = result.loc[len(result) - 1, 'open']
        result = result[['time', 'prev_close', 'open', 'close', 'last_prc', 'high', 'low', 'volume', 'turnover']]

        return result

    def calculate_daily_data(self):
        self.load_components()
        # 获取目标区间内每个交易日的成分股信息
        dt_df = pd.read_csv('input/trade_date.csv')
        dt_df['TRADE_DAYS'] = pd.to_datetime(dt_df['TRADE_DAYS'].astype(int).astype(str), format='%Y%m%d')
        filtered_dates = dt_df[(dt_df['TRADE_DAYS'] >= self.start_date) & (dt_df['TRADE_DAYS'] <= self.end_date)]
        date_range = pd.DatetimeIndex(filtered_dates['TRADE_DAYS']).strftime('%Y%m%d').tolist()
        if self.code == '8841388.WI':
            for target_date in date_range:
                index_df = self.calculate_index_equal_weight(target_date)
                print(index_df.tail())
        else:
            for target_date in date_range:
                index_df = self.calculate_index_mkt_weight(target_date)
                print(index_df.tail())

        if self.idx_type == 1:
            index_df.to_csv(f'output/3s_predict_data/PriceIndex/{self.code}_{target_date}.csv', index=False)
        elif self.idx_type == 2:
            index_df.to_csv(f'output/3s_predict_data/TotalReturnIndex/{self.code}_{target_date}.csv', index=False)
        else:
            index_df.to_csv(f'output/3s_predict_data/StandarlizedIndex/{self.code}_{target_date}.csv', index=False)


class Visualizer:
    """绘制对比图"""
    @staticmethod
    def draw_tendency(code: str,
                      idx_type: int,
                      df_3s: pd.DataFrame,
                      df_g: pd.DataFrame = None,
                      start_time: str = None,
                      end_time: str = None
                      ):
        df_3s['time'] = pd.to_datetime(df_3s['time'])
        date = pd.to_datetime(start_time).date()
        tp = 'Price' if idx_type == 1 else 'Total Return' if idx_type == 2 else 'Standarlied'

        if start_time is not None:
            start_time = pd.to_datetime(start_time)
            df_3s = df_3s[df_3s['time'] >= start_time]

        if end_time is not None:
            end_time = pd.to_datetime(end_time)
            df_3s = df_3s[df_3s['time'] <= end_time]

        if df_g is not None:
            open = df_g.loc[0,'last_prc']
            df_g['rnt_rate'] = df_g['last_prc'] / open
            df_g['time'] = df_g['time'].apply(DataHandler.convert_time_format)
            df_g['date'] = df_g['date'].astype(str)
            df_g['time'] = df_g['time'].astype(str)
            df_g['datetime'] = pd.to_datetime(df_g['date'] + ' ' + df_g['time'], format='%Y%m%d %H:%M:%S')
            df_g = df_g[df_g['datetime'] >= start_time]
            df_g = df_g[df_g['datetime'] <= end_time]

            df_g['datetime'] = df_g['datetime'].astype('datetime64[ns]')  # 确保为纳秒
            df_3s['time'] = df_3s['time'].astype('datetime64[ns]')
            # 对齐两个 DataFrame 的时间
            df_merged = pd.merge_asof(df_3s.sort_values('time'),
                                      df_g.sort_values('datetime'),
                                      left_on='time',
                                      right_on='datetime',
                                      direction='backward',
                                      suffixes=('_pre', '_g'))
            df_merged.to_csv(f'output/{index_code}_merged_20240913.csv')
            df_merged = df_merged[['time_pre', 'rnt_rate_pre', 'rnt_rate_g']]
            # 创建带有两个 Y 轴的子图
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # 添加 3 秒数据的价格趋势
            fig.add_trace(
                go.Scatter(x=df_merged['time_pre'], y=df_merged['rnt_rate_pre'], mode='lines',
                           name='3s Predicted Price Trend')
                ,secondary_y=False
            )

            # 添加每分钟数据的价格趋势
            fig.add_trace(
                go.Scatter(x=df_merged['time_pre'], y=df_merged['rnt_rate_g'], mode='lines', name='True Price Trend')
                ,secondary_y=True
            )

            # 更新布局
            fig.update_layout(
                title='{}\'s {} Trend Over Time'.format(code.upper(), tp),
                xaxis_title=f'Time(Date：{date})',
                yaxis_title='3s Predicted Price',
                yaxis2_title='True Price',
                xaxis_tickformat='%H:%M',
                xaxis=dict(
                    tickmode='array',
                    tickvals=df_merged['time_pre'][::20],  # 根据需要选择刻度
                    ticktext=df_merged['time_pre'].dt.strftime('%H:%M')[::20]
                ),
                legend=dict(x=0, y=1)
            )

            # 更新 Y 轴设置
            fig.update_yaxes(title_text="3s Predicted Price", secondary_y=False, tickformat='.2f')
            fig.update_yaxes(title_text="True Price", secondary_y=True, tickformat='.2f')

        else:
            # 如果没有，绘制单一的 3 秒数据图表
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=df_3s['time'], y=df_3s['rnt_rate'], mode='lines', name='3s Price Trend')
            )

            # 更新布局
            fig.update_layout(
                title='{}\'s {} Trend Over Time'.format(code.upper(), tp),
                xaxis_title=f'Time (Date: {date})',
                yaxis_title='3s Price',
                xaxis_tickformat='%H:%M',
                xaxis=dict(
                    tickmode='array',
                    tickvals=df_3s['time'][::20],  # 根据需要选择刻度
                    ticktext=df_3s['time'].dt.strftime('%H:%M')[::20]
                ),
                legend=dict(x=0, y=1)
            )
            fig.update_yaxes(title_text="3s Price", tickformat='.2f')
        # 显示图表
        fig.show()


if __name__ == "__main__":
    start_dt = '20240913'
    end_dt = '20240916'
    index_code_list = [
        '000852.SH'
        #  , '000300.SH'
        # , '8841388.WI'
    ]
    index_type_list= [3]
    today = datetime.now().strftime('%Y%m%d')

    # 处理eod和wind等权指数
    processor = IndexDataProcessor('8841388.WI', '20141020', today)

    if not os.path.isfile('output/A_EOD_PRO_from_20141020.csv'):
        processor.eod_df.to_csv('output/A_EOD_PRO_from_20141020.csv', index=False)

    for index_code in index_code_list:
        for index_type in index_type_list:
            print(f"""
            Processing...
            Index: {index_code}
            Type: {index_type}
                  """)
            if index_code == '8841388.WI' and not os.path.isfile('output/8841388.WI_pre_index_from_20141020.csv'):
                pre_index_df = processor.calculate_total_return_index()
                pre_index_df.to_csv('output/8841388.WI_pre_index_from_20141020.csv', index=False)

            eod_df = pd.read_csv('output/A_EOD_PRO_from_20141020.csv')
            eod_df['TRADE_DT'] = pd.to_datetime(eod_df['TRADE_DT'].astype(str), format='%Y-%m-%d')

            # 计算除数
            if index_code == '8841388.WI':
                div_df = None
            else:
                divider = GetDivider(index_code, eod_df, index_type)
                div_df = divider.get_initial_value()

            # 计算指数
            index = IndexGeneration(index_code, index_type, start_dt, end_dt, eod_df, div_df)
            index.calculate_daily_data()

            # 可视化对比
            visualizer = Visualizer()
            if index_type == 1:
                df_3s = pd.read_csv(f'output/3s_predict_data/PriceIndex/{index_code}_20240913.csv')
            elif index_type == 2:
                df_3s = pd.read_csv(f'output/3s_predict_data/TotalReturnIndex/{index_code}_20240913.csv')
            else:
                df_3s = pd.read_csv(f'output/3s_predict_data/StandarlizedIndex/{index_code}_20240913.csv')


            if index_code == '8841388.WI':
                visualizer.draw_tendency(index_code, index_type, df_3s, start_time='2024-09-13 09:25:00', end_time='2024-09-13 15:00')

            else:
                df_g = pd.read_csv('input/secbar_data/{}_secbar_{}.csv'.format(index_code, '20240913'))
                df_g[['date','time']] = df_g[['date','time']].astype(str)
                visualizer.draw_tendency(index_code, index_type, df_3s, df_g, start_time='2024-09-13 09:35:00', end_time='2024-09-13 11:30:00')