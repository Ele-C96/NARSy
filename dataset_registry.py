
DATASETS = {

    "adult": {
        "file": "adult.csv",
        "target": "income",
        "classes": ["gt_50k_No", "gt_50k_Yes"],
        "accuracy": {"gt_50k_No": 0.70, "gt_50k_Yes": 0.73},
        "topk": {"gt_50k_No": 461, "gt_50k_Yes": 1492},
        "lift": {"gt_50k_No": 2.44, "gt_50k_Yes": 1.26},
        "car_accuracy": 0.83,
        "car_topk": 428,
        "extra_id_columns": [],
    },

    "loan": {
        "file": "loan.csv",
        "target": "loan_status",
        "classes": ["No", "Yes"],
        "accuracy": {"No": 0.59, "Yes": 0.60},
        "topk": {"No": 166, "Yes": 67},
        "lift": {"No": 1.07, "Yes": 1.24},
        "car_accuracy": 0.70,
        "car_topk": 36,
        "extra_id_columns": [],
    },

    "diabetes012": {
        "file": "diabetes012.csv",
        "target": "Diabetes_012",
        "classes": ["diabet", "no_diabet", "pre_diabet"],
        "accuracy": {"diabet": 0.55, "no_diabet": 0.80, "pre_diabet": 0.49},
        "topk": {"diabet": 882, "no_diabet": 392, "pre_diabet": 653},
        "lift": {"diabet": 1.04, "no_diabet": 1.06, "pre_diabet": 1.05},
        "car_accuracy": 0.90,
        "car_topk": 1368,
        "extra_id_columns": [],
    },

    "student_performance": {
        "file": "student_performance.csv",
        "target": "GradeClass",
        "classes": ["A", "B", "C", "D", "F"],
        "accuracy": {"A": 0.84, "B": 0.74, "C": 0.71, "D": 0.71, "F": 0.55},
        "topk": {"A": 50, "B": 154, "C": 268, "D": 190, "F": 43},
        "lift": {"A": 1.14, "B": 1.02, "C": 1.05, "D": 1.03, "F": 1.01},
        "car_accuracy": 0.57,
        "car_topk": 11,
        "extra_id_columns": [],
    },

    "amazon": {
        "file": "amazon.csv",
        "target": "target",
        "classes": ["0", "1"],
        "accuracy": {"0": 0.72, "1": 0.50},
        "topk": {"0": 3, "1": 3},
        "lift": {"0": 1.16, "1": 1.23},
        "car_accuracy": 0.72,
        "car_topk": 6,
        "extra_id_columns": [],
    },

    "diabetes": {
        "file": "diabetes.csv",
        "target": "Outcome",
        "classes": ["0", "1"],
        "accuracy": {"0": 0.58, "1": 0.62},
        "topk": {"0": 13, "1": 28},
        "lift": {"0": 1.06, "1": 1.05},
        "car_accuracy": 0.61,
        "car_topk": 9,
        "extra_id_columns": [],
    },

    "studentdepression": {
        "file": "studentdepression.csv",
        "target": "Depression",
        "classes": ["0", "1"],
        "accuracy": {"0": 0.62, "1": 0.56},
        "topk": {"0": 368, "1": 195},
        "lift": {"0": 1.00, "1": 1.00},
        "car_accuracy": 0.90,
        "car_topk": 195,
        "extra_id_columns": [],
    },

    "irrigation": {
        "file": "irrigation.csv",
        "target": "Irrigation_Need",
        "classes": ["High", "Low", "Medium"],
        "accuracy": {"High": 0.77, "Low": 0.46, "Medium": 0.56},
        "topk": {"High": 4, "Low": 9, "Medium": 17},
        "lift": {"High": 1.01, "Low": 1.01, "Medium": 1.01},
        "car_accuracy": 0.53,
        "car_topk": 1,
        "extra_id_columns": [],
    },

    "pollution": {
        "file": "pollution.csv",
        "target": "Air_Quality",
        "classes": ["Good", "Hazardous", "Moderate", "Poor"],
        "accuracy": {"Good": 0.69, "Hazardous": 0.85, "Moderate": 0.70, "Poor": 0.72},
        "topk": {"Good": 5, "Hazardous": 5, "Moderate": 5, "Poor": 36},
        "lift": {"Good": 1.08, "Hazardous": 1.00, "Moderate": 1.00, "Poor": 1.07},
        "car_accuracy": 0.89,
        "car_topk": 15,
        "extra_id_columns": [],
    },

    "NPHA": {
        "file": "NPHA.csv",
        "target": "Number_of_Doctors_Visited",
        "classes": ["1", "2", "3"],
        "accuracy": {"1": 0.71,"2": 0.53,"3": 0.62},
        "topk": {"1": 1028,"2": 991,"3": 973},
        "lift": {"1": 1.18,"2": 1.28,"3": 1.27},
        "car_accuracy": 0.82,   
        "car_topk": 1039,         
        "extra_id_columns": [],
    },

}

MIN_SUPPORT = 0.2
MIN_CONFIDENCE = 0.8
