import json
import logging
import os
import time
from datetime import datetime
import tensorflow as tf


def setup_tensorflow():
    gpus = tf.config.experimental.list_physical_devices("GPU")

    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU detected: {len(gpus)}")
    else:
        print("No GPU detected")


def start_timer():
    return time.time()


def stop_timer(start_time: float):
    return round(time.time() - start_time, 4)


def create_run_structure(dataset: str, experiment_name: str):
    run_id = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    base_dir = f"results/{dataset}/{experiment_name}/run_{run_id}"

    metrics_dir = os.path.join(base_dir, "metrics")
    logs_dir = os.path.join(base_dir, "logs")
    models_dir = os.path.join(base_dir, "models")
    plots_dir = os.path.join(base_dir, "plots")
    config_dir = os.path.join(base_dir, "config")

    for d in [base_dir, metrics_dir, logs_dir, models_dir, plots_dir, config_dir]:
        os.makedirs(d, exist_ok=True)

    return {
        "run_id": run_id,
        "base_dir": base_dir,
        "metrics_dir": metrics_dir,
        "logs_dir": logs_dir,
        "models_dir": models_dir,
        "plots_dir": plots_dir,
        "config_dir": config_dir,
    }


def setup_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def save_json(data: dict, path: str):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)