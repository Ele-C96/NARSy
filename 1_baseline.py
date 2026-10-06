import argparse
import json
import os
import platform

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf

from dataset_registry import DATASETS
from src import hyper, plot, utils
from src.experiment_utils import (
    create_run_structure,
    save_json,
    setup_logger,
    setup_tensorflow,
    start_timer,
    stop_timer,
)
from src.run_final_training import get_params
from src.models import Classifier
from src.trainer import CrossValTrainer

INVALID_KEYS = {"_score", "_metric"}

EXPERIMENT_NAME = "baseline"
DEFAULT_OUTPUT_ROOT = "results"
FINAL_EPOCHS = 200

SYSTEM_INFO = {
    "tensorflow": tf.__version__,
    "sklearn": sklearn.__version__,
    "python": platform.python_version(),
    "platform": platform.platform(),
}


def load_best_params(grid_file):
    with open(grid_file) as f:
        trials = json.load(f)
    best = max(trials, key=lambda x: x["_score"])
    return {k: v for k, v in best.items() if k not in INVALID_KEYS}


def build_paths(dataset):
    return {
        "data": f"data/{dataset}/{dataset}.csv",
        "data_pre": f"data/{dataset}/{dataset}_pre.csv",
    }


def run(dataset, output_root=None, epochs=FINAL_EPOCHS):
    setup_tensorflow()

    if dataset not in DATASETS:
        raise ValueError(f"Unregistered dataset: {dataset}")

    output_root = output_root or os.environ.get("NARSY_OUTPUT_ROOT") or DEFAULT_OUTPUT_ROOT

    run_info = create_run_structure(dataset=dataset, experiment_name="baseline")

    logger = setup_logger(os.path.join(run_info["logs_dir"], "execution.log"))
    logger.info(SYSTEM_INFO)
    logger.info(f"Results directory: {run_info['base_dir']}")

    info = DATASETS[dataset]
    target = info["target"]

    paths = build_paths(dataset)

    utils.set_random_seed(42)

    df_full = pd.read_csv(paths["data_pre"])

    grid_path = os.path.join(run_info["config_dir"], f"{dataset}_baseline_grid.json")

    best_params = get_params(
        dataset, EXPERIMENT_NAME, logger=logger, required=False, drop_keys=INVALID_KEYS
    )

    if best_params is None:
        logger.info("Starting grid search")
        grid_timer = start_timer()

        hyper.grid_search(
            Classifier,
            df_full,
            folds=3,
            training_epochs=100,
            batch_size=[64, 128, 256],
            lr=[1e-3, 3e-4],
            dropout=[0.0, 0.1],
            units=[
                [64] * 3,
                [128] * 3,
                [64, 128, 256],
                [256, 128, 64],
                [256] * 3,
            ],
            trainable_embeddings=[False, True],
            kernel_initializer=["he_uniform"],
            target=target,
            scaler=None,
            save=grid_path,
            processes=1,
            category_max_one_hot_size=[7],
            category_embedding_size=[4, 8],
        )

        logger.info("Grid search completed")
        grid_search_time = stop_timer(grid_timer)
        best_params = load_best_params(grid_path)
    else:
        grid_search_time = 0.0

    logger.info(f"Best params: {best_params}")
    logger.info("Starting final training")
    train_timer = start_timer()
    final_params = dict(best_params)
    batch_size = final_params.pop("batch_size")

    trainer = CrossValTrainer(
        df_full,
        folds=3,
        batch_sizes=(batch_size, 1024),
        target_column=target,
        stratify=True,
        seed=31,
    )

    trainer.train(num_epochs=epochs, model_class=Classifier, **final_params)

    trainer.save(name="classifier", base_dir=run_info["models_dir"], remove_old=True)

    logger.info("Training completed")
    final_training_time = stop_timer(train_timer)

    total_execution_time = grid_search_time + final_training_time

    metrics = {
        "cv_mean_score": float(np.mean(trainer.scores)),
        "cv_std_score": float(np.std(trainer.scores)),
        "scores_per_fold": [float(x) for x in trainer.scores],
        "grid_search_time_sec": grid_search_time,
        "final_training_time_sec": final_training_time,
        "total_execution_time_sec": total_execution_time,
    }

    save_json(metrics, os.path.join(run_info["metrics_dir"], "metrics.json"))

    experiment_config = {
        "run_id": run_info["run_id"],
        "dataset": dataset,
        "target": target,
        "best_params": best_params,
        "seed": 42,
        "model": "Classifier",
        "folds": 3,
        "num_samples": len(df_full),
        "num_features": len(df_full.columns),
        "train_epochs_grid": 200 if grid_search_time > 0.0 else None,
        "train_epochs_final": epochs,
        "params_reused": grid_search_time == 0.0,
        "system_info": SYSTEM_INFO,
    }

    save_json(
        experiment_config, os.path.join(run_info["config_dir"], "experiment_config.json")
    )

    fig = plot.history_comparison(
        {f"baseline dataset {dataset}": trainer.models}, highlight_best=True
    )

    fig.savefig(
        os.path.join(run_info["plots_dir"], f"history_auc_{dataset}_base.png"),
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    logger.info(f"Total execution time: {total_execution_time:.2f} seconds")
    logger.info(best_params)
    logger.info(metrics)

    print(f"\n{'=' * 62}")
    print(f"{dataset} / {EXPERIMENT_NAME}")
    print(f"  cv_mean_score : {metrics['cv_mean_score']:.4f} ± {metrics['cv_std_score']:.4f}")
    print(f"  per fold      : {[round(s, 4) for s in metrics['scores_per_fold']]}")
    print(f"  total time    : {total_execution_time / 60:.1f} min")
    print(f"  output        : {run_info['base_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-root", default=None, help=f"output root (default: {DEFAULT_OUTPUT_ROOT})")
    parser.add_argument("--epochs", type=int, default=FINAL_EPOCHS)
    args = parser.parse_args()

    run(args.dataset, output_root=args.output_root, epochs=args.epochs)