"""Export dashboard-ready figures using Matplotlib's default colour cycle."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def save_figures(demand, price, dm, pm, champions, confusion, shocks, directory):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    def save(fig, name):
        fig.savefig(directory / (name + '.png'), dpi=150, bbox_inches='tight')
        fig.savefig(directory / (name + '.svg'), bbox_inches='tight')
        plt.close(fig)
    for task, metrics, measure, unit in [('demand', dm, 'wape', 'WAPE (%)'), ('price', pm, 'mae', 'MAE (MYR)')]:
        fig, ax = plt.subplots(figsize=(12, 6), layout='constrained')
        for name, group in metrics.groupby('baseline'):
            g = group.dropna(subset=[measure]).sort_values('horizon')
            if len(g):
                ax.plot(g.horizon, g[measure] * (100 if measure == 'wape' else 1), marker='o', label=name)
        ax.set(xlabel='Forecast horizon (calendar days)', ylabel=unit, title=f'{task.title()} baseline error — development folds only')
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
        save(fig, task + '_error_by_horizon')
        fig, ax = plt.subplots(figsize=(12, 6), layout='constrained')
        metrics[metrics.horizon == 7].dropna(subset=['bias']).set_index('baseline').bias.plot.barh(ax=ax)
        ax.set(xlabel='Bias: mean(prediction − actual)', title=f'{task.title()} bias at 7 days')
        save(fig, task + '_bias')
    for task, frame, h, ylabel in [('demand', demand, 7, 'Cumulative demand (units)'), ('price', price, 30, 'Purchase price (MYR)')]:
        selected = champions[(champions.task == task) & (champions.horizon == h)].baseline.iloc[0]
        if selected is None:
            continue
        group = frame[(frame.horizon == h) & (frame.baseline == selected)]
        pair = group[['store_id', 'product_id']].iloc[0]
        group = group[(group.store_id == pair.store_id) & (group.product_id == pair.product_id)].sort_values('date')
        fig, ax = plt.subplots(figsize=(12, 5), layout='constrained')
        ax.plot(group.date, group.actual, marker='o', label='Actual')
        ax.plot(group.date, group.prediction, marker='.', label=selected)
        ax.set(xlabel='Forecast origin', ylabel=ylabel, title=f'{h}-day {task}: {pair.store_id} / {pair.product_id}')
        ax.legend()
        save(fig, task + '_actual_vs_predicted')
    fig, axes = plt.subplots(3, 3, figsize=(12, 11), layout='constrained')
    methods = sorted(confusion.baseline.unique())
    labels = ['DOWN', 'FLAT', 'UP']
    for row, name in enumerate(methods):
        for col, h in enumerate([1, 7, 30]):
            ax = axes[row, col]
            g = confusion[(confusion.baseline == name) & (confusion.horizon == h)]
            matrix = g.pivot(index='actual', columns='predicted', values='count').reindex(index=labels, columns=labels).to_numpy()
            ax.imshow(matrix)
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, str(int(matrix[i, j])), ha='center', va='center',
                            bbox=dict(facecolor=plt.rcParams['figure.facecolor'], edgecolor='none', alpha=.85, pad=2))
            ax.set_xticks(range(3), labels)
            ax.set_yticks(range(3), labels)
            ax.set(title=f'{name}\n{h} days', xlabel='Predicted', ylabel='Actual')
    save(fig, 'price_direction_confusion')
    fig, ax = plt.subplots(figsize=(12, 6), layout='constrained')
    show = shocks[(shocks.horizon == 30) & (shocks.scenario == 'price_shock')]
    available = show[show['count'] > 0]
    if len(available):
        available.pivot(index='baseline', columns='scenario_active', values='mae').plot.barh(ax=ax)
        ax.legend(title='Absolute price change > threshold')
    else:
        ax.text(.5, .5, 'No eligible price-shock observations', ha='center', transform=ax.transAxes)
    ax.set(title='30-day purchase-price error during shocks', xlabel='MAE (MYR)')
    save(fig, 'price_shock_performance')
