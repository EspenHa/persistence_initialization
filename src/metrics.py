import numpy as np
import pandas as pd

from src.data import M4DataModule


def compute_metrics(dm: M4DataModule, predictions):
    dm.prepare_data()

    stats = pd.read_pickle(dm.statistics_cache_path)
    s_naive_MAE = stats["s_naive_MAE"].values[:, None]
    naive2_MASE = stats["naive2_MASE"].values.mean()
    naive2_sMAPE = stats["naive2_sMAPE"].values.mean()

    true_values = pd.read_pickle(dm.test_cache_path).values.T

    abs_error = np.abs(true_values - predictions)
    smape = np.mean(200 * abs_error / (np.abs(true_values) + np.abs(predictions)))
    mase = np.mean(abs_error / s_naive_MAE)
    owa = 0.5 * (smape / naive2_sMAPE + mase / naive2_MASE)

    return {"sMAPE": smape.item(), "MASE": mase.item(), "OWA": owa.item()}
