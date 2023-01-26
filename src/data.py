from __future__ import annotations

import urllib
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
import patoolib

import pytorch_lightning
import torch

from src import GLOBALS


@dataclass
class M4Info:
    frequency: str

    n_series: int
    seasonality: int

    outsample_size: int  # horizon size
    insample_size: int  # input window size
    max_length: int = field(init=False)  # maximum total length = outsample_size + insample_size

    def __post_init__(self):
        self.max_length = self.outsample_size + self.insample_size


info = {
    "Hourly": M4Info(
        frequency="Hourly",
        n_series=414,
        seasonality=24,
        outsample_size=48,
        insample_size=4 * 48,
    ),
    "Daily": M4Info(
        frequency="Daily",
        n_series=4227,
        seasonality=1,
        outsample_size=14,
        insample_size=3 * 14,
    ),
    "Weekly": M4Info(
        frequency="Weekly",
        n_series=359,
        seasonality=1,
        outsample_size=13,
        insample_size=4 * 13,
    ),
    "Monthly": M4Info(
        frequency="Monthly",
        n_series=48_000,
        seasonality=12,
        outsample_size=18,
        insample_size=3 * 18,
    ),
    "Quarterly": M4Info(
        frequency="Quarterly",
        n_series=24_000,
        seasonality=4,
        outsample_size=8,
        insample_size=3 * 8,
    ),
    "Yearly": M4Info(
        frequency="Yearly",
        n_series=23_000,
        seasonality=1,
        outsample_size=6,
        insample_size=3 * 6,
    ),
}

# Change this global variable to modify behavior of various data dependent classes
INFO = info["Monthly"]


@dataclass
class Batch:
    """
    Custom batch class for transformer sequence prediction.
    """

    data: torch.Tensor
    idx: torch.Tensor
    mask: torch.Tensor
    nan_mask: torch.Tensor

    @property
    def teacher_forcing_inputs(self):
        return self.data

    @property
    def autoregressive_inputs(self):
        return self.data[: INFO.insample_size]

    @property
    def targets(self):
        return self.data[INFO.insample_size :]

    @property
    def mean(self):
        data = self.autoregressive_inputs[-INFO.outsample_size :]
        return torch.mean(data, dim=0, keepdim=True)

    def clone(self) -> Batch:
        return Batch(
            data=self.data.clone().detach(),
            idx=self.idx,
            mask=self.mask,
            nan_mask=self.nan_mask,
        )

    @staticmethod
    def collate(batches: List[Batch]) -> Batch:
        return Batch(
            data=torch.cat([batch.data for batch in batches], dim=1),
            idx=torch.cat([batch.idx for batch in batches]),
            mask=torch.cat([batch.mask for batch in batches]),
            nan_mask=torch.cat([batch.nan_mask for batch in batches]),
        )

    def __len__(self):
        return self.data.size(1)

    @property
    def shape(self):
        return self.data.shape


class ConcatDataFrameDataset(torch.utils.data.ConcatDataset):
    def __init__(self, df, start, end, length):
        super(torch.utils.data.ConcatDataset, self).__init__()
        self.datasets = [
            DataFrameTransformerForecastingDataset(df, col, start, end, length)
            for col in df.columns
        ]
        self.cumulative_sizes = self.cumsum(self.datasets)


class DataFrameTransformerForecastingDataset(torch.utils.data.Dataset):
    def __init__(self, df, key, start, end, length):
        self.key = key
        self.df = df

        self.length = length
        self.start = start
        self.end = end

        chunk_size = length[key]
        if chunk_size <= 0:
            self.n_examples = 0
        elif chunk_size < INFO.max_length:
            self.n_examples = 1
        else:
            self.n_examples = (chunk_size - INFO.max_length) + 1

    @property
    def chunk(self):
        column = self.df[self.key]
        values = column[self.start[self.key] : self.end[self.key]]
        return values.to_numpy()

    @property
    def idx(self):
        return torch.full((1,), self.key, dtype=torch.int32)

    def __len__(self):
        return self.n_examples

    def __getitem__(self, i):
        data = self.chunk[i : i + INFO.max_length]
        length = data.size

        assert length > INFO.outsample_size

        M = INFO.max_length

        mask = torch.zeros((M, M), dtype=torch.bool)
        a, b = torch.triu_indices(M, M, offset=1)
        mask[a, b] = True

        pad_length = M - length
        left_pad = pad_length

        if pad_length != 0:
            padded = np.full(M, np.nan, dtype=np.float32)
            padded[left_pad:M] = data
            data = padded

        na = np.isnan(data)

        nan_mask = torch.from_numpy(np.bitwise_and.outer(~na, na))
        mask |= nan_mask

        mask = mask.unsqueeze(0)  # add batch dim
        nan_mask = nan_mask.unsqueeze(0)

        data = torch.from_numpy(data)[:, None, None]  # seq, batch, channel

        return Batch(data=data, mask=mask, nan_mask=nan_mask, idx=self.idx)


def read_csv(path):
    df = pd.read_csv(
        path,
        index_col=0,
        header=0,
    ).transpose()
    df.reset_index(inplace=True, drop=True)
    df = df.astype(np.float32, copy=False)
    return df


BASE_URL = "https://github.com/Mcompetitions/M4-methods/raw/master/"
URL_TEMPLATE = BASE_URL + "Dataset/{}/{}-{}.csv"
NAIVE_URL = BASE_URL + "Point%20Forecasts/submission-Naive2.rar"


def download(frequency):
    data_dir = GLOBALS.root / "data" / "M4"
    data_dir.mkdir(exist_ok=True, parents=True)

    train_url = URL_TEMPLATE.format("Train", frequency, "train")
    test_url = URL_TEMPLATE.format("Test", frequency, "test")

    train_download_path = data_dir / train_url.split("/")[-1]
    test_download_path = data_dir / test_url.split("/")[-1]

    naive_download_path = data_dir / "submission-Naive2.rar"
    naive_path = data_dir / "submission-Naive2.csv"

    for url, path in [
        (train_url, train_download_path),
        (test_url, test_download_path),
        (NAIVE_URL, naive_download_path),
    ]:
        if not path.exists():
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, str(path))

    if not naive_path.exists():
        patoolib.extract_archive(
            str(naive_download_path),
            outdir=str(data_dir),
            interactive=False,
        )

    return train_download_path, test_download_path, naive_path


def concat_train_test(train, test):
    train_test_boundary = train.isna().idxmax()
    train_test_boundary[train_test_boundary == 0] = train.shape[0]

    nan_padding = test.copy()
    nan_padding[:] = np.nan

    data = pd.concat([train, nan_padding], ignore_index=True)

    for i in range(INFO.n_series):
        key = f"{INFO.frequency[0]}{i+1}"
        split = train_test_boundary[key]
        data[key].iloc[split : split + INFO.outsample_size] = test[key]

    return data, train_test_boundary


def compute_sMAPE(targets, predictions):
    error = targets - predictions
    smape_ratio = error.abs() / (targets.abs() + predictions.abs())
    return 200 * (smape_ratio).mean()


def compute_stats(train, test, naive2):
    naive_error = train.diff()
    s_naive_error = train.diff(periods=INFO.seasonality)

    naive_mae = naive_error.abs().mean()
    seasonal_naive_mae = s_naive_error.abs().mean()

    naive_smape = compute_sMAPE(train, train.shift(-1))
    s_naive_smape = compute_sMAPE(train, train.shift(-INFO.seasonality))

    naive2 = naive2.filter(regex=f"{INFO.frequency[0]}.*")
    naive2 = naive2.iloc[: INFO.outsample_size]  # remove nans

    naive2_error = naive2 - test
    naive2_mase = (naive2_error.abs() / seasonal_naive_mae).mean()

    naive2_smape = compute_sMAPE(test, naive2)

    return pd.DataFrame(
        {
            "naive2_MASE": naive2_mase,
            "naive2_sMAPE": naive2_smape,
            "naive_MAE": naive_mae,
            "naive_sMAPE": naive_smape,
            "s_naive_MAE": seasonal_naive_mae,
            "s_naive_sMAPE": s_naive_smape,
        }
    )


class M4DataModule(pytorch_lightning.LightningDataModule):
    train: torch.utils.data.Dataset
    val: torch.utils.data.Dataset
    test: torch.utils.data.Dataset

    def __init__(self, frequency: str):
        super().__init__()

        global INFO
        INFO = info[frequency]

        data_dir = GLOBALS.root / "data" / "M4"
        self.cache_path = data_dir / f"{INFO.frequency}.pickle"
        self.statistics_cache_path = data_dir / f"{INFO.frequency}_statistics.pickle"

        self.train_cache_path = data_dir / f"{INFO.frequency}_train.pickle"
        self.test_cache_path = data_dir / f"{INFO.frequency}_test.pickle"

        self.setup_done = False

    @property
    def INFO(self):
        return INFO

    def prepare_data(self):
        if (
            self.cache_path.exists()
            and self.statistics_cache_path.exists()
            and self.train_cache_path.exists()
            and self.test_cache_path.exists()
        ):
            return

        print("Downloading ...")
        train_path, test_path, naive_path = download(INFO.frequency)

        print("Reading csv ...")
        train_data = read_csv(train_path)
        test_data = read_csv(test_path)

        train_data.to_pickle(self.train_cache_path)
        test_data.to_pickle(self.test_cache_path)

        print("Processing data ...")

        data, train_test_boundary = concat_train_test(train_data, test_data)
        data.to_pickle(self.cache_path)

        naive2 = read_csv(naive_path)

        statistics = compute_stats(train_data, test_data, naive2)
        statistics["length"] = train_test_boundary + INFO.outsample_size
        statistics.to_pickle(self.statistics_cache_path)

    def setup(self, stage=None):
        if self.setup_done:
            return

        print("Loading data ...")
        df = pd.read_pickle(self.cache_path)
        df.columns = pd.RangeIndex(df.shape[1])

        self.statistics = pd.read_pickle(self.statistics_cache_path)

        length = self.statistics["length"]
        length.index = pd.RangeIndex(df.shape[1])

        zeros = length.copy()
        zeros[:] = 0

        test_split_end = length
        test_split_start = length - INFO.max_length
        test_split_start = test_split_start.clip(0)

        val_split_end = test_split_end - INFO.outsample_size

        val_min_length = val_split_end.quantile(0.25)

        enough_for_val = val_split_end >= val_min_length

        val_split_start = val_split_end - enough_for_val * INFO.max_length
        val_split_start = val_split_start.clip(0)

        train_split_end = val_split_end - enough_for_val * INFO.outsample_size
        train_split_start = zeros

        self.train = ConcatDataFrameDataset(
            df,
            train_split_start,
            train_split_end,
            train_split_end - train_split_start,
        )

        self.val = ConcatDataFrameDataset(
            df,
            val_split_start,
            val_split_end,
            val_split_end - val_split_start,
        )

        self.test = ConcatDataFrameDataset(
            df,
            test_split_start,
            test_split_end,
            test_split_end - test_split_start,
        )

        self.setup_done = True

    def train_dataloader(self):
        weights = np.concatenate(
            [np.full(len(dataset), 1 / len(dataset)) for dataset in self.train.datasets]
        )

        epoch_length = 128
        batch_size = 1024

        sampler = torch.utils.data.WeightedRandomSampler(
            weights,
            num_samples=batch_size * epoch_length,
            replacement=True,
        )

        return torch.utils.data.DataLoader(
            batch_size=batch_size,
            dataset=self.train,
            collate_fn=Batch.collate,
            drop_last=True,
            sampler=sampler,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            batch_size=1024,
            dataset=self.val,
            collate_fn=Batch.collate,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            batch_size=1024,
            dataset=self.test,
            collate_fn=Batch.collate,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
