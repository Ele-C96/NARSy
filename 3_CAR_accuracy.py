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
from src.models import CARModel
from src.rules import AprioriRuleSet
from src.trainer import CrossValTrainer

INVALID_KEYS = {"_score", "_metric"}
DROP_KEYS = INVALID_KEYS | {"kernel_initializer"}

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
    return {k: v for k, v in best.items() if k not in DROP_KEYS}


def build_paths(dataset):
    return {
        "data": f"data/{dataset}/{dataset}.csv",
        "data_pre": f"data/{dataset}/{dataset}_pre.csv",
        "data_apriori": f"data/{dataset}/{dataset}_apriori.csv",
        "rules": f"rules/{dataset}/apriori_{dataset}.json",
    }


def run(dataset, accuracy):
    setup_tensorflow()

    if dataset not in DATASETS:
        raise ValueError(f"Unregistered dataset: {dataset}")

    run_info = create_run_structure(dataset=dataset, experiment_name="CAR_Accuracy")

    logger = setup_logger(os.path.join(run_info["logs_dir"], "execution.log"))
    logger.info(SYSTEM_INFO)
    logger.info(f"Results directory: {run_info['base_dir']}")

    info = DATASETS[dataset]
    target = info["target"]

    if accuracy is None:
        accuracy = info["car_accuracy"]

    logger.info(f"accuracy threshold = {accuracy}")

    paths = build_paths(dataset)

    utils.set_random_seed(42)

    logger.info("Loading datasets")
    df_full = pd.read_csv(paths["data_pre"])
    df_apriori = pd.read_csv(paths["data_apriori"])

    logger.info("Loading rules")
    rule_set = AprioriRuleSet.load(path=paths["rules"])
    logger.info(f"Initial rules: {len(rule_set.rules)}")

    scores = rule_set.classify(df_apriori, save_scores=False)
    rule_set.rules["accuracy"] = scores

    rs_filtered = rule_set.filter_by_metric(metric="accuracy", min_value=float(accuracy))
    logger.info(f"Rules after accuracy filtering: {len(rs_filtered.rules)}")

    rules_per_index = rs_filtered.get_per_index(df_apriori)

    grid_path = os.path.join(run_info["config_dir"], f"{dataset}_CAR_Accuracy_grid.json")

    best_params = get_params(
        dataset, "CAR_accuracy", df_apriori, logger=logger, required=False, drop_keys=DROP_KEYS
    )

    if best_params is None:
        logger.info("Starting grid search")
        grid_timer = start_timer()

        hyper.grid_search(
            CARModel,
            df_full,
            fixed_rules=rules_per_index,
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
            target=target,
            scaler=None,
            save=grid_path,
            embedding_size=[4, 8],
            num_embedding_columns=[len(df_apriori.columns)],
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
        rules=rules_per_index,
        target_column=target,
        stratify=True,
        seed=31,
    )

    trainer.train(num_epochs=200, model_class=CARModel, **final_params)

    trainer.save(name="CAR_Accuracy", base_dir=run_info["models_dir"], remove_old=True)

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
        "model": "CARModel",
        "folds": 3,
        "num_samples": len(df_full),
        "num_features": len(df_full.columns),
        "train_epochs_grid": 100 if grid_search_time > 0.0 else None,
        "train_epochs_final": 200,
        "params_reused": grid_search_time == 0.0,
        "accuracy_threshold": float(accuracy),
        "system_info": SYSTEM_INFO,
    }

    save_json(
        experiment_config, os.path.join(run_info["config_dir"], "experiment_config.json")
    )

    fig = plot.history_comparison(
        {f"CAR dataset {dataset} Accuracy {accuracy}": trainer.models}, highlight_best=True
    )

    fig.savefig(
        os.path.join(run_info["plots_dir"], f"history_auc_{dataset}_CAR_Accuracy.png"),
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    logger.info(f"Total execution time: {total_execution_time:.2f} seconds")
    logger.info(best_params)
    logger.info(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--accuracy", required=False, default=None)
    args = parser.parse_args()

    run(args.dataset, args.accuracy)