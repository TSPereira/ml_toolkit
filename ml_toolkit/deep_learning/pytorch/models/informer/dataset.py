import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from .....utils.os_utl import check_options


class InformerTemporalDataset(Dataset):
    _time_tick_map = dict(S=0, M=1, H=2, d=3, w=4, m=5)

    # todo implement checks on lengths passed
    @check_options(time_tick=('S', 'M', 'H', 'd', 'm'))
    def __init__(self, df, enc_seq_len, dec_seq_len, pred_len, target, time_tick='M', scale=True):
        self.enc_seq_len = enc_seq_len
        self.dec_seq_len = dec_seq_len
        self.pred_len = pred_len
        self.time_tick = time_tick
        self._time_tick = self._time_tick_map[time_tick]
        self.target = target if isinstance(target, (list, tuple, set)) else [target]
        self.scale = scale

        self._prepare_data(df)

    # todo add support for numpy array
    def _prepare_data(self, df):
        scaler = StandardScaler()
        data = df.values.astype(float)
        self.data = scaler.fit_transform(data) if self.scale else data
        self.target_idx = df.columns.get_indexer_for(self.target)

        data_stamp = df.index.to_frame(index=False, name='date')
        data_stamp['date'] = pd.to_datetime(data_stamp['date'])

        data_stamp['month'] = data_stamp['date'].dt.month

        if self._time_tick < 5:
            data_stamp['week'] = data_stamp['date'].dt.isocalendar().week

        if self._time_tick < 4:
            data_stamp['day'] = data_stamp['date'].dt.day
            data_stamp['weekday'] = data_stamp['date'].dt.weekday

        if self._time_tick < 3:
            data_stamp['hour'] = data_stamp['date'].dt.hour

        if self._time_tick < 2:
            data_stamp['minute'] = data_stamp['date'].dt.minute

        if self._time_tick < 1:
            data_stamp['second'] = data_stamp['date'].dt.second

        data_stamp.drop('date', axis=1, inplace=True)
        self.data_stamp = data_stamp.values.astype(float)

    def __getitem__(self, index):
        if index >= len(self):
            raise StopIteration

        s_begin = index
        s_end = s_begin + self.enc_seq_len
        r_begin = s_end - self.dec_seq_len
        r_end = s_end + self.pred_len

        x_enc = self.data[s_begin:s_end]
        x_dec = np.zeros_like(self.data[-self.pred_len:])
        x_dec = np.concatenate([self.data[r_begin:s_end], x_dec])
        y = self.data[s_end:r_end, self.target_idx]

        x_enc_stamp = self.data_stamp[s_begin:s_end]
        x_dec_stamp = self.data_stamp[r_begin:r_end]

        return x_enc, x_dec, x_enc_stamp, x_dec_stamp, y

    def __len__(self):
        return len(self.data) - self.enc_seq_len - self.pred_len + 1


if __name__ == '__main__':
    # todo convert to test
    lgth = 200
    data = pd.DataFrame({'A': np.sin(range(lgth))}, index=pd.date_range(start='01/01/2019', periods=lgth, freq='H'))
    dt = InformerTemporalDataset(data, 20, 5, 5, 'A', 'M', scale=False)

    from torch.utils.data import DataLoader

    dl = DataLoader(dt, 2)
    a = next(iter(dl))

    print(1)
