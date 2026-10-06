import os
import pandas as pd
import numpy as np
from src.rules import AprioriRuleSet
from src.metrics_value import RuleSetMetricAnalyzer, per_class_thresholds, per_class_topk_thresholds
from src.plot import plot_and_save_threshold


DATASETS = {
    "NPHA": "Number_of_Doctors_Visited",
    "adult": "income",
    "loan": "loan_status",
    "diabetes012": "Diabetes_012",
    "student_performance":"GradeClass",
    "amazon":"target",
    "diabetes": "Outcome",
    "irrigation":"Irrigation_Need",
    "pollution": "Air Quality",
    "studentdespression": "Depression",

}

BASE_THRESHOLD_DIR = "thresholds"



def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_load_ruleset(path: str):
    if not os.path.exists(path):
        return None
    try:
        rs = AprioriRuleSet.load(path=path)
        return rs if rs and len(rs.rules) > 0 else None
    except Exception:
        return None
    
def infer_task_type(df: pd.DataFrame, target: str, max_classes=10):
    """
    return: 'classification' | 'regression'
    """
    y = df[target]

    if isinstance(y.dtype, pd.CategoricalDtype):
        return "classification"

    if pd.api.types.is_string_dtype(y) or y.dtype == object:
        return "classification"

    if pd.api.types.is_numeric_dtype(y):
        if y.nunique() <= max_classes:
            return "classification"
        else:
            return "regression"

    raise ValueError(f"target not supported: {y.dtype}")



for dataset, target_col in DATASETS.items():
    print(f"\nProcessing dataset: {dataset}")

    data_apriori_path = f"data/{dataset}/{dataset}_apriori.csv"
    print(f"Looking for dataset at: {data_apriori_path}")
    data_path = f"data/{dataset}/{dataset}_pre.csv"
    rules_path = f"rules/{dataset}/apriori_{dataset}.json"


    if not os.path.exists(data_apriori_path):
        print("Dataset not found, skipping.")
        continue

    rs = safe_load_ruleset(rules_path)

    if rs is None:
        print("Ruleset not found, skipping.")
        continue
    
    df = pd.read_csv(data_path)
    df_enc = pd.read_csv(data_apriori_path)
    task_type = infer_task_type(df, target_col)

    print(f"Detected task type: {task_type} of dataset {dataset}")

    analyzer = RuleSetMetricAnalyzer(rs, df_enc, df)


    dataset_out = os.path.join(BASE_THRESHOLD_DIR, dataset)
    ensure_dir(dataset_out)

    
    k_values = np.linspace(0, len(rs), 10).astype(int)
    print("\nLift calculation:\n")
    th_lift, lift_curve = analyzer.lift_threshold(return_curve=True)
    

    plot_and_save_threshold(
        **lift_curve,
        output_path=f"{dataset_out}/lift_{dataset}.png",
        metric="lift",
        dataset=dataset,
    )

    if task_type == "classification":
        print("\nAccuracy calculation:\n")
        th_acc, acc_curve = analyzer.accuracy_threshold(return_curve=True)
        print("\nTopK calculation:\n")
        th_topk, topk_curve = analyzer.topk_threshold(k_values, return_curve=True)
        
        plot_and_save_threshold(
            **acc_curve,
            output_path=f"{dataset_out}/accuracy_{dataset}.png",
            metric="accuracy",
            dataset=dataset,
        )

        plot_and_save_threshold(
            **topk_curve,
            output_path=f"{dataset_out}/topk_{dataset}.png",
            metric="topk",
            dataset=dataset,
            topk_real_value=th_topk
        )

    elif task_type == "regression":
        print("\nMSE calculation:\n")
        th_mse, mse_curve = analyzer.mse_threshold(target=target_col, return_curve=True)
        print("\nTopK calculation:\n")
        th_topk, topk_curve = analyzer.topk_threshold(
            k_values, return_curve=True, target=target_col, target_type="regression"
        )


        plot_and_save_threshold(
            **mse_curve,
            output_path=f"{dataset_out}/mse_{dataset}.png",
            metric="mse",
            dataset=dataset,
        )

        plot_and_save_threshold(
            **topk_curve,
            output_path=f"{dataset_out}/topk_{dataset}.png",
            metric="topk",
            dataset=dataset,
        )

    class_rule_sets = {}
    for col in df_enc.columns:
      
        if target_col in col:
            label = col.split("=")[-1]

            path = f"rules/{dataset}/apriori_{dataset}-{label}.json"
            rs_c = safe_load_ruleset(path)
            print(f"Loaded ruleset for class {label} of dataset {dataset}: {rs_c is not None}")
            if rs_c:
                class_rule_sets[label] = rs_c

    class_acc = {}
    class_lift = {}
    class_mse = {}
    class_topk = {}
    #k_values = np.linspace(len(rs_c), 0, 10).astype(int)
    k_values = np.linspace(0, len(rs_c), 10).astype(int)
    if class_rule_sets and task_type == "classification":
        for label, rs_c in class_rule_sets.items():

            print(f"\nProcessing class: {label}\n")

            single_class_dict = {label: rs_c}

            acc_value, acc_curve = per_class_thresholds(single_class_dict, df_enc, df, "accuracy", return_curve=True)
            class_acc[label] = acc_value[label]

            plot_and_save_threshold(
                **acc_curve[label],
                output_path=f"{dataset_out}/accuracy_{dataset}.png",
                metric="accuracy",
                class_label=label,
                dataset=dataset,
            )



            lift_value, lift_curve = per_class_thresholds(single_class_dict, df_enc, df, "lift", return_curve=True)
            class_lift[label] = lift_value[label]

            plot_and_save_threshold(
                **lift_curve[label],
                output_path=f"{dataset_out}/lift_{dataset}.png",
                metric="lift",
                class_label=label,
                dataset=dataset,
            )
            

            topk_value, topk_curve = per_class_topk_thresholds(single_class_dict, df_enc, df, k_values, return_curve=True, target=target_col, target_type=task_type)
            class_topk[label] = topk_value[label]
            plot_and_save_threshold(
                **topk_curve[label],
                output_path=f"{dataset_out}/topk_{dataset}.png",
                metric="topk",
                class_label=label,
                dataset=dataset,
            )
    elif class_rule_sets and task_type == "regression":

        for label, rs_c in class_rule_sets.items():
            print(f"Processing class: {label}")
            single_class_dict = {label: rs_c}

            mse_value, mse_curve = per_class_thresholds(single_class_dict, df_enc, df, "mean_squared_error", target_col, return_curve=True)
            class_mse[label] = mse_value[label]

            plot_and_save_threshold(
                **mse_curve[label],
                output_path=f"{dataset_out}/mse_{dataset}.png",
                metric="mse",
                class_label=label,
                dataset=dataset,
            )



            lift_value, lift_curve = per_class_thresholds(single_class_dict, df_enc, df, "lift", return_curve=True)
            class_lift[label] = lift_value[label]

            plot_and_save_threshold(
                **lift_curve[label],
                output_path=f"{dataset_out}/lift_{dataset}.png",
                metric="lift",
                class_label=label,
                dataset=dataset,
            )
          

            topk_value, topk_curve = per_class_topk_thresholds(single_class_dict, df_enc, df, k_values, return_curve=True, target=target_col, target_type=task_type)
            class_topk[label] = topk_value[label]

            plot_and_save_threshold(
                **topk_curve[label],
                output_path=f"{dataset_out}/topk_{dataset}.png",
                metric="topk",
                class_label=label,
                dataset=dataset,
            )


    def fmt(x):
        return f"{x:.2f}" if isinstance(x, (int, float)) else "N/A"


    with open(f"{dataset_out}/results.txt", "w") as f:
        f.write(f"Dataset: {dataset}\n")
        if task_type == "classification":
            f.write(f"Accuracy threshold: {fmt(th_acc)}\n")
        if task_type == "regression":
            f.write(f"MSE threshold: {fmt(th_mse)}\n")

        f.write(f"Lift threshold: {fmt(th_lift)}\n")
        f.write(f"Top-K threshold: {fmt(th_topk)}\n\n")



        if class_rule_sets:
            for label, rs_c in class_rule_sets.items():
                if label:
                    if task_type == "classification":
                        f.write(
                            f"Class {label}:\n"
                            f"  Accuracy: {fmt(class_acc.get(label))}\n"
                            f"  Lift: {fmt(class_lift.get(label))}\n"
                            f"  Top-K: {fmt(class_topk.get(label))}\n\n"
                       )
                    elif task_type == "regression":
                        f.write(
                            f"Class {label}:\n"
                            f"  MSE: {fmt(class_mse.get(label))}\n"
                            f"  Lift: {fmt(class_lift.get(label))}\n"
                            f"  Top-K: {fmt(class_topk.get(label))}\n\n"
                        )
                else:
                     f.write(f"Class label is empty, skipping results write.\n")

    print(f"\nCompleted dataset: {dataset}")

