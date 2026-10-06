import json
from typing import Optional
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay, mean_squared_error, r2_score, roc_auc_score, roc_curve

from typing import List, Optional, Dict, Tuple, Union
from pathlib import Path

from src.models import CARModel, DataSequence
from src.trainer import Trainer, SimpleTrainer, CrossValTrainer





def history(model: CARModel, metric: str, prefix='val', fig_size=(12, 10), legend='best'):
    plt.figure(figsize=fig_size)

    val_metric = f'{prefix}_{metric}'
    plt.plot(np.random.choice(model.history[metric],
                              size=len(model.history[val_metric]),
                              replace=False),
             label=metric)
    plt.plot(model.history[val_metric], label=val_metric)

    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.legend(loc=legend)
    plt.show()


def history_comparison(histories: dict, metric='val_auc', fig_size=(12, 10),
                       title: str = None, y_label: str = None, legend='best',
                       base_path='weights', highlight_best=False, verbose=True):
    """Histories can be: str (path to model's history), CARModel, or
        list[CARModel]"""
    def load_history(file: str) -> dict:
        # load history at the given path
        path = Path(base_path) / file

        if verbose:
            print(f'Loading model history "{path}"..')

        return json.load(fp=open(path, 'r'))

    plt.figure(figsize=fig_size)

    for k, h in histories.items():
        if isinstance(h, list):
            new_h = []
            for h_ in h:
                if isinstance(h_, CARModel):
                    new_h.append(h_.history)

                elif isinstance(h_, str):
                    new_h.append(load_history(file=h_))
                else:
                    new_h.append(h_)
            h = new_h

            values = [hist[metric] for hist in h]
            avg = np.mean(values, axis=0)

            line = plt.plot(avg, label=k, zorder=1)
            plt.fill_between(np.arange(len(avg)), np.min(values, axis=0),
                             np.max(values, axis=0), alpha=0.5,
                             color=line[-1].get_color())
            values = avg
        else:
            if isinstance(h, str):
                h = load_history(file=h)

            values = h[metric]
            line = plt.plot(values, label=k, zorder=2)

        if highlight_best:
            best_value = np.max(values)
            best_color = line[-1].get_color()

            plt.axhline(best_value, linestyle='dotted',
                        color=best_color, alpha=1, zorder=1)

            plt.plot(np.argmax(values), best_value, 'o',
                     color=best_color, zorder=3)

    if isinstance(title, str):
        plt.title(title)

    plt.xlabel('Epoch')
    plt.ylabel(y_label or metric)
    plt.legend(loc=legend)
    #plt.show()
    fig = plt.gcf()
    #plt.show()
    return fig


def confusion_matrix(models: List[CARModel], test_sequence: DataSequence, labels: Optional[List[str]] = None,
                     threshold=0.5, **display_kwargs):
    """Plots the binary or multi-class confusion matrix"""
    y_true, y_pred = _predict(models, test_sequence, should_squeeze=False)

    if y_true.ndim == 2 and y_true.shape[-1] > 1:
        # multiclass case
        y_true = y_true.argmax(-1)
        y_pred = y_pred.argmax(-1)
    else:
        # binary case
        y_pred = y_pred > threshold

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred,
                                            display_labels=labels, **display_kwargs)


def roc_comparison(models_and_sequences: Dict[str, Union[Trainer, Tuple[List[CARModel], List[DataSequence]]]],
                   fig_size=(12, 10), legend='lower right', **roc_kwargs):
    """Plots the ROC curve for the provided models"""
    plt.figure(figsize=fig_size)

    for k, trainer_or_models_and_sequences in models_and_sequences.items():
        if isinstance(trainer_or_models_and_sequences, Trainer):
            trainer = trainer_or_models_and_sequences

            if isinstance(trainer, SimpleTrainer):
                models = [trainer.model]
                sequences = [trainer.valid_seq]
            else:
                assert isinstance(trainer, CrossValTrainer)
                models = trainer.models
                sequences = [val_seq for _, val_seq in trainer.sequences]
        else:
            models, sequences = trainer_or_models_and_sequences

            if not isinstance(models, (list, tuple)):
                models = [models]

            if not isinstance(sequences, (list, tuple)):
                sequences = [sequences]

        assert len(models) == len(sequences), 'The num of provided models and sequences must be the same!'

        fpr_list = []
        tpr_list = []
        auc_list = []
        decision_point = []

        for model, seq in zip(models, sequences):
            y_true = np.concatenate([y for _, y in seq])
            y_pred = model.predict(seq, verbose=0)

            fpr, tpr, auc, fpr_at_t, tpr_at_t = roc_auc(y_true, y_pred, **roc_kwargs)

            fpr_list.append(fpr)
            tpr_list.append(tpr)
            auc_list.append(auc)
            decision_point.append((fpr_at_t, tpr_at_t))

        max_size = max([len(x) for x in fpr_list])
        fpr_common = np.linspace(0, 1, max_size)
        tpr_interp = [np.interp(fpr_common, fpr, tpr)
                      for fpr, tpr in zip(fpr_list, tpr_list)]

        tpr_arr = np.stack(tpr_interp)
        avg_tpr = np.mean(tpr_arr, axis=0)
        decision_point = np.array(decision_point)

        curve = plt.plot(fpr_common, avg_tpr,
                         label=fr'[{k}] AUC: {np.mean(auc_list):.2f} $\pm$ ({np.std(auc_list):.4f})')
        curve_color = curve[-1].get_color()

        plt.fill_between(fpr_common, tpr_arr.min(axis=0), tpr_arr.max(axis=0),
                         alpha=0.5, color=curve_color)

        t = roc_kwargs.get('threshold', 0.5)
        plt.scatter(decision_point[:, 0].mean(), decision_point[:, 1].mean(),
                    color=curve_color, marker='o',
                    label=f'[{k}] mTPR@{t}: {decision_point[:, 1].mean():.2f}')

    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC curve')

    plt.legend(loc=legend)
    plt.show()


def histogram(models: List[CARModel], test_sequence: DataSequence, bins: int, value_range: tuple = None,
              fig_size=(12, 10), legend='best', density=False, x_label='', hatch_pred='//',
              hatch_true=None):
    y_true, y_pred = _predict(models, test_sequence)

    plt.figure(figsize=fig_size)

    plt.hist(y_pred, range=value_range, bins=int(bins), label='pred', histtype='step',
             hatch=hatch_pred, color='tab:blue', density=bool(density))

    plt.hist(y_true, range=value_range, bins=int(bins), label='true', hatch=hatch_true,
             histtype='step', color='tab:orange', density=bool(density))

    plt.xlabel(x_label)
    plt.legend(loc=legend)
    plt.show()


def regression(models: List[CARModel], test_sequence: DataSequence, min_value: float = None, max_value: float = None,
               fig_size=(12, 10), legend='upper left'):
    """Plots the predicted vs expected target value, along with its linear fit"""
    y_true, y_pred = _predict(models, test_sequence)
    mask = _get_mask(y_pred, min_value, max_value)

    yt_mask = y_true[mask]
    yp_mask = y_pred[mask]

    plt.figure(figsize=fig_size)
    plt.scatter(yt_mask, yp_mask, marker='o', alpha=0.2)

    # line fitting
    z = np.polyfit(yt_mask, yp_mask, deg=1)
    line = np.poly1d(z)
    line_pred = np.array(line(yp_mask))

    r2 = r2_score(y_true, y_pred)
    plt.plot(yp_mask, line_pred, linestyle='dashed', color='black',
             label=f'R2 = {r2:.2f}')

    plt.title('Predicted vs Actual')
    plt.xlabel('Price')
    plt.ylabel('Predicted')

    plt.legend(loc=legend)
    plt.show()


def residual(models: List[CARModel], test_sequence: DataSequence, min_value: float = None, max_value: float = None,
             text_loc=(0, 0), text_size=10, fig_size=(12, 10)):
    """Plots the residual (true - predicted) value"""
    y_true, y_pred = _predict(models, test_sequence)
    y_residual = y_pred - y_true
    mask = _get_mask(y_residual, min_value, max_value)

    plt.figure(figsize=fig_size)

    plt.axhline(y=0, linestyle='dashed', color='black', label='zero residual')
    plt.scatter(y_pred[mask], y_residual[mask], alpha=0.2, label='true - pred')

    mse = mean_squared_error(y_true, y_pred)
    plt.text(*text_loc, f'MSE = {mse:.2f}', fontsize=int(text_size),
             bbox=dict(boxstyle='square', facecolor='whitesmoke', alpha=1, pad=0.5))

    plt.xlabel('Predicted')
    plt.ylabel('Residual')

    plt.legend()
    plt.show()


def roc_auc(true, pred, threshold=0.5, **roc_kwargs):
    fpr, tpr, t = roc_curve(true, pred)
    auc = roc_auc_score(true, pred, **roc_kwargs)

    idx = (np.abs(threshold - np.array(t))).argmin()

    return fpr, tpr, auc, fpr[idx], tpr[idx]


def _predict(models: List[CARModel], seq: DataSequence, should_squeeze=True) -> tuple:
    y_true = []
    y_pred = []

    for model in models:
        for x, y in seq:
            y_pred.append(model(x, training=False))
            y_true.append(y)

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    if should_squeeze:
        return y_true.squeeze(), y_pred.squeeze()

    return y_true, y_pred


def _get_mask(array, min_value, max_value, be_assertive=True) -> np.ndarray:
    mask = np.full(len(array), fill_value=True)

    if isinstance(min_value, (int, float)):
        mask = np.logical_and(mask, array > min_value)

    if isinstance(max_value, (int, float)):
        mask = np.logical_and(mask, array < max_value)

    if be_assertive:
        assert mask.any(), f'Interval ({min_value}, {max_value}) is too restrictive!'

    return mask


import os
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt


def plot_and_save_threshold(
    x: np.ndarray,
    discarded_norm: np.ndarray,
    coverage: np.ndarray,
    x_intersection: Optional[float],
    y_intersection: Optional[float],
    output_path: str,
    metric: str,
    class_label: Optional[str] = None,
    dataset: Optional[str] = None,
    topk_real_value: Optional[float] = None,
):
    
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    base, ext = os.path.splitext(output_path)
    if class_label is not None:
        filepath = f"{base}_{class_label}{ext}"
    else:
        filepath = f"{base}_dataset{ext}"
    title = (dataset[:1].upper() + dataset[1:]) if dataset else ""

    x_label = "Discarded fraction (top-K)" if metric == "topk" else f"{metric.capitalize()} threshold"

    COLOR_DISCARDED = "#4E7FA4"
    COLOR_COVERAGE = "#60A881"
    COLOR_POINT = "#EAA357"
    COLOR_GUIDE = "#8a8a8a"

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    ax.plot(x, discarded_norm, label="Fraction of rules discarded",
            linewidth=2.2, color=COLOR_DISCARDED, zorder=3)
    ax.plot(x, coverage, label="Fraction of dataset covered",
            linewidth=2.2, color=COLOR_COVERAGE, zorder=3)

    ax.fill_between(x, discarded_norm, coverage, color="gray", alpha=0.08, zorder=1, interpolate=True)

    if x_intersection is not None and y_intersection is not None:
      
        ax.plot([x_intersection, x_intersection], [0, y_intersection],
                linestyle=":", linewidth=1.1, color=COLOR_GUIDE, zorder=2)
        ax.plot([x.min(), x_intersection], [y_intersection, y_intersection],
                linestyle=":", linewidth=1.1, color=COLOR_GUIDE, zorder=2)

        point_label = "K selected rules" if metric == "topk" else "Selected Accuracy threshold "
        ax.scatter([x_intersection], [y_intersection], s=100, zorder=5,
                   color=COLOR_POINT, edgecolor="black", linewidth=0.5,
                   label=point_label)

        if metric == "topk":
            if topk_real_value is not None:
                x_label_value = f"{int(topk_real_value)}"
            else:
              
                x_label_value = f"{x_intersection:.2f}"
        else:
            x_label_value = f"{x_intersection:.2f}"

        ax.annotate(
            f"{x_label_value}",
            xy=(x_intersection, y_intersection),
            xytext=(13, 0), textcoords="offset points",
             va="center", ha="left",
            fontsize=9, color='black', fontweight="bold",
        )

    ax.set_title(title, fontsize=13)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("Normalized fraction", fontsize=11)
    ax.set_ylim(0, 1.05)
    
    ax.grid(axis="both", ls="--", lw=0.5, alpha=0.9)

    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(loc="lower right", fontsize=9, frameon=True, framealpha=0.6)

    fig.tight_layout()
    fig.savefig(filepath, dpi=300)
    fig.savefig(filepath.replace(ext, ".pdf")) 
    plt.close(fig)


