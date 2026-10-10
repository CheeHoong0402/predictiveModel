"""Disjoint future-outcome blocks, with a fixed origin and expanding past."""
import pandas as pd


def build_folds(data_start, data_end, config):
    c = config['evaluation']
    start, end = pd.Timestamp(data_start), pd.Timestamp(data_end)
    holdout = pd.Timestamp(c['holdout_start'])
    validation_start = max(pd.Timestamp(c['start_date']), start + pd.Timedelta(days=c['minimum_history_days']))
    rows = []
    window = c['validation_window_days']
    while validation_start + pd.Timedelta(days=window - 1) < holdout and validation_start + pd.Timedelta(days=window - 1) <= end:
        train_end = validation_start - pd.Timedelta(days=1)
        train_start = start if c['history_mode'] == 'expanding' else max(start, train_end - pd.Timedelta(days=c['rolling_history_days'] - 1))
        rows.append(dict(validation_fold=len(rows) + 1, training_start=train_start, training_end=train_end,
                         forecast_origin=train_end, validation_start=validation_start,
                         validation_end=validation_start + pd.Timedelta(days=window - 1)))
        validation_start += pd.Timedelta(days=window)
    if not rows:
        raise ValueError('Insufficient pre-holdout history for a complete validation block')
    result = pd.DataFrame(rows)
    assert (result.training_end < result.validation_start).all()
    assert (result.validation_end < holdout).all()
    assert (result.validation_start.iloc[1:].to_numpy() > result.validation_end.iloc[:-1].to_numpy()).all()
    return result
