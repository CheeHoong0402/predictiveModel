"""Metric definitions, coverage-aware rankings and fold stability summaries."""
import numpy as np
import pandas as pd

LABELS = ['DOWN', 'FLAT', 'UP']


def regression_metrics(frame, task):
    valid_actual = frame.actual.notna() & np.isfinite(frame.actual)
    eligible = frame.loc[valid_actual]
    good = eligible.loc[np.isfinite(eligible.prediction)]
    result = dict(target_count=len(eligible), count=len(good), excluded_predictions=len(eligible) - len(good),
                  prediction_coverage=len(good) / len(eligible) if len(eligible) else np.nan)
    metric_names = ['mae', 'rmse', 'bias', 'wape', 'mase', 'mase_count', 'mape', 'smape',
                    'direction_accuracy', 'pinball_q10', 'pinball_q50', 'pinball_q90', 'interval_coverage_80', 'mean_interval_width']
    result.update({name: np.nan for name in metric_names})
    if good.empty:
        return result
    error = good.prediction.to_numpy() - good.actual.to_numpy()
    actual = good.actual.to_numpy()
    result.update(mae=np.abs(error).mean(), rmse=np.sqrt(np.mean(error ** 2)), bias=error.mean())
    if task == 'demand':
        denominator = np.abs(actual).sum()
        result['wape'] = np.abs(error).sum() / denominator if denominator > 0 else np.nan
        # H * mean absolute seasonal-naive daily error, estimated solely from
        # that fold's training history. Undefined zero scales are not imputed.
        scale = good.mase_scale.to_numpy(float)
        mask = np.isfinite(scale) & (scale > 0)
        result['mase'] = np.mean(np.abs(error[mask]) / scale[mask]) if mask.any() else np.nan
        result['mase_count'] = int(mask.sum())
    else:
        nonzero = actual > 0
        result['mape'] = np.mean(np.abs(error[nonzero]) / actual[nonzero]) if nonzero.any() else np.nan
        denominator = np.abs(actual) + np.abs(good.prediction.to_numpy())
        mask = denominator > 0
        result['smape'] = np.mean(2 * np.abs(error[mask]) / denominator[mask]) if mask.any() else np.nan
        if 'actual_direction' in good:
            both = good.actual_direction.notna() & good.predicted_direction.notna()
            result['direction_accuracy'] = good.loc[both, 'actual_direction'].eq(good.loc[both, 'predicted_direction']).mean()
    for q in [.1, .5, .9]:
        column = f'q{int(q * 100)}'
        if column in good:
            finite = np.isfinite(good[column])
            residual = good.loc[finite, 'actual'] - good.loc[finite, column]
            result['pinball_' + column] = np.maximum(q * residual, (q - 1) * residual).mean()
    if {'q10', 'q90'}.issubset(good):
        interval = good[good.q10.notna() & good.q90.notna()]
        if len(interval):
            result['interval_coverage_80'] = ((interval.actual >= interval.q10) & (interval.actual <= interval.q90)).mean()
            result['mean_interval_width'] = (interval.q90 - interval.q10).mean()
    return result


def direction_metrics(actual, predicted):
    data = pd.DataFrame({'actual': actual, 'predicted': predicted}).dropna()
    matrix = pd.crosstab(data.actual, data.predicted).reindex(index=LABELS, columns=LABELS, fill_value=0)
    per_class = []
    for label in LABELS:
        tp = matrix.loc[label, label]
        predicted_n, actual_n = matrix[label].sum(), matrix.loc[label].sum()
        per_class.append(dict(label=label, support=int(actual_n), precision=tp / predicted_n if predicted_n else 0.,
                              recall=tp / actual_n if actual_n else np.nan))
    pc = pd.DataFrame(per_class)
    metrics = dict(count=len(data), accuracy=data.actual.eq(data.predicted).mean(),
                   balanced_accuracy=pc.recall.mean(), macro_precision=pc.precision.mean(), macro_recall=pc.recall.mean())
    return metrics, pc, matrix


def metric_table(frame, task, group_columns):
    rows = []
    for key, group in frame.groupby(group_columns, dropna=False, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        rows.append(dict(task=task, **dict(zip(group_columns, key)), **regression_metrics(group, task)))
    return pd.DataFrame(rows)


def rankings_and_champions(demand, price, config):
    rows, champions = [], []
    for task, frame, metric in [('demand', demand, 'wape'), ('price', price, 'mae')]:
        overall = metric_table(frame, task, ['horizon', 'baseline'])
        folds = metric_table(frame, task, ['horizon', 'baseline', 'validation_fold'])
        for _, row in overall.iterrows():
            f = folds[(folds.horizon == row.horizon) & (folds.baseline == row.baseline)]
            scores = f[metric].dropna()
            rows.append(dict(task=task, horizon=row.horizon, baseline=row.baseline, primary_metric=metric,
                pooled_error=row[metric], mean_fold_error=scores.mean(), std_fold_error=scores.std(),
                median_fold_error=scores.median(), best_fold_error=scores.min(), worst_fold_error=scores.max(),
                folds=len(scores), prediction_coverage=row.prediction_coverage,
                eligible=bool(row.prediction_coverage >= config['evaluation']['minimum_prediction_coverage'] and
                              len(scores) >= config['evaluation']['minimum_champion_folds'])))
    ranking = pd.DataFrame(rows).sort_values(['task', 'horizon', 'eligible', 'mean_fold_error', 'worst_fold_error', 'pooled_error', 'baseline'],
                                           ascending=[True, True, False, True, True, True, True]).reset_index(drop=True)
    ranking['rank'] = ranking.groupby(['task', 'horizon']).cumcount() + 1
    for (task, h), group in ranking.groupby(['task', 'horizon']):
        eligible = group[group.eligible]
        if eligible.empty:
            champions.append(dict(task=task, horizon=h, baseline=None, status='NO_FULL_COVERAGE_CHAMPION'))
        else:
            row = eligible.iloc[0].to_dict()
            row['status'] = 'DEVELOPMENT_CHAMPION'
            row['champion_name'] = f'champion_{task}_baseline_{h}d'
            champions.append(row)
    return ranking, pd.DataFrame(champions)
