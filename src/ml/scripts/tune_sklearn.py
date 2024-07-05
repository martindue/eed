# TODO: Make this take in an arbitrary data module as set by config file

import json
import os
import sys
import time
from pathlib import Path

import lightning.pytorch as lp
import numpy as np
import optuna
import pandas as pd
import skl2onnx.helpers.onnx_helper
import sklearn.model_selection
import xgboost as xgb
from jsonargparse import CLI, ActionConfigFile, ArgumentParser
from lightning.pytorch.cli import (LightningArgumentParser, LightningCLI,
                                   SaveConfigCallback)
from skl2onnx import convert_sklearn, to_onnx, update_registered_converter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)  # Assuming the project root is two directories above src
sys.path.append(project_root)

from funcy import omit
from onnxmltools.convert.xgboost.operator_converters.XGBoost import \
    convert_xgboost
from skl2onnx.common.shape_calculator import \
    calculate_linear_classifier_output_shapes

import eval.misc.utils as utils
from eval.misc import eval_utils, matching
# from ml.scripts.main_sklearn import main as main_sklearn
from ml.datasets.lookAtPointDatasetMiddleLabel.datamodule import \
    LookAtPointDataMiddleLabelModule
from ml.utils.classes import job, pp_args
from ml.utils.helpers import (impute_with_column_means,
                              linear_interpol_with_pandas)
from ml.utils.post_processing import post_process


def scale_data(X_train, X_val):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    return X_train, X_val

def calculate_metrics(data_gt, data_pr, job):
    matchers = job.matchers
    job_label = job.label
    multiclass_strategy = job.multiclass_strategy
    binary_strategy = job.binary_strategy
    event_map = job.event_map
    event_map = utils.keys2num(event_map)

    event_labels = event_map.values()
    if 0 not in event_map.keys():
        # add undef label
        event_labels = [0, *event_labels]

    event_labels = list(set(event_labels))
    event_matcher = matching.EventMatcher(gt=data_gt, pr=data_pr)

    result_accum = []
    # run eval
    for matcher, matching_kwargs in matchers.items():
        matcher_label = filter(None, (matcher, job_label))
        matcher_label = {"matcher": "-".join(matcher_label)}

        eval_result = eval_utils.calc_scores(
            event_matcher=event_matcher,
            matcher=matcher,
            matching_kwargs=matching_kwargs,
            labels=event_labels,
            multiclass_strategy=multiclass_strategy,
            binary_strategy=binary_strategy,
        )

        result_accum.extend(eval_result)
    return result_accum


def objective_function(trial: optuna.trial.Trial, parser: ArgumentParser, args) -> float:

    window_size = trial.suggest_int("window_size", 50, 300)
    window_size_vel = trial.suggest_int("window_size_vel", 48, 300)
    savgol_filter_width = trial.suggest_int("savgol_filter_width", 50, 200)

    # Selection of dataset datasets 

    #datasets = ["SE_recorded", "synthetic_data", "lookAtPoint"] # "hollywood2_em" #,"lund2013"
    #datasets = ["hollywood2_em"]
    #datasets = ["lund2013"]
    #datasets = ["lookAtPoint"]
    datasets = ["SE_recorded","synthetic_data"]

    #selected_indices = []
    #for i in range(len(datasets)):
    #    if trial.suggest_categorical('dataset_'+datasets[i], [True, False]):
    #        selected_indices.append(i)
#
    #if not selected_indices:
    #    return 0.0  # Return a baseline or zero score if no dataset is selected

    training_datasets = datasets
    #training_datasets = [datasets[i] for i in selected_indices]


    noise_levels = [0]
    if "lookAtPoint" in training_datasets:
        use_noise = trial.suggest_categorical("use_noise", [True, False])
        if use_noise:
            #noise_levels = args.data.noise_levels
            noise_levels = [1337]
        else:
            noise_levels = [0]


    data_module = LookAtPointDataMiddleLabelModule(
        data_dir=args.data.data_dir,
        sklearn=True,
        training_datasets=training_datasets,
        window_size=window_size, 
        savgol_filter_window=savgol_filter_width,
        window_size_vel=window_size_vel,
        noise_levels=noise_levels,
    )

    data_module.prepare_data()
    data_module.setup(stage="fit")
    train_data_loader = data_module.train_dataloader()
    val_data_loader = data_module.val_dataloader()

    for data, features, file_name_list in train_data_loader:
        X_train = features #data["features"]
        X_train = X_train.squeeze()
        #y_train_df = pd.DataFrame(omit(data, ["features","file_name"]))   
        #y_train_data = omit(data, ["features", "file_name"])
        for key, value in data.items():
            data[key] = value.reshape(-1)

        y_train_df = pd.DataFrame(data)#  pd.DataFrame({k: v.tolist() for k, v in y_train_data.items()})
        y_train_df["file_name"] = file_name_list
        y_train_df.rename(columns={"label": "evt"}, inplace=True)
        y_train_df.replace({"evt": args.jobs.event_map}, inplace=True)
        #y_train_df = y_train_df.explode(["evt","status"],ignore_index=True)

        y = data["label"]
        # Replace values in numpy array using dictionary
        for old_val, new_val in args.jobs.event_map.items():
            y[y == old_val] = new_val
        y.numpy()
        y=y.squeeze()

    for data, features, file_name_list in val_data_loader:
        X_val = features #data["features"]
        X_val = X_val.squeeze()

        #file_names = [item[0] for item in data['file_name']]
        #val_data_dict = omit(data, ["features","file_name"])
#
        #val_data_dict['file_name'] = file_names
        for key, value in data.items():
            data[key] = value.reshape(-1)
        val_data =  pd.DataFrame(data) #pd.DataFrame(val_data_dict)
        val_data["file_name"] = file_name_list
        y_val_df = val_data.filter(["t", "x", "y", "status", "label"])
        y_val_df.rename(columns={"label": "evt"}, inplace=True)
        y_val_df.replace({"evt": args.jobs.event_map}, inplace=True)
        #y_val_df = y_val_df.explode(["evt","status"],ignore_index=True)

    val_mask = y_val_df["status"]| (y_val_df["evt"] != 5)
    y_val_df = y_val_df[val_mask]
    X_val = X_val[val_mask] # this is slightly wrong, because the window can still contain samples with status False
    val_data = val_data[val_mask]   

    train_mask = y_train_df["status"]|(y_train_df["evt"] != 5)
    X_train = X_train[train_mask]
    y = y[train_mask]
    y_train_df = y_train_df[train_mask]

 
    y_val_df.reset_index(drop=True, inplace=True)
    val_data.reset_index(drop=True, inplace=True)

    print("Data loaded")
    X_train = X_train.numpy()
    y = y.numpy()

    X_train = linear_interpol_with_pandas(X_train)
    #X_train = impute_with_column_means(X_train)  # TODO: Change to a smarter approach

    X_val = X_val.numpy()
    X_val = linear_interpol_with_pandas(X_val)
    #X_val = impute_with_column_means(X_val)

    print("dimensions of X and y:", X_train.shape, y.shape)

    #classifier_name = trial.suggest_categorical("classifier", [ "RandomForest", "adaBoost", "knn","mlp"])#SVC,"mlp" , "logisticRegression"
    #classifier_name = "RandomForest"
    #classifier_name = "xgboost"
    #classifier_name = "adaBoost"
    #classifier_name = "mlp"
    classifier_name = "RandomForest"

    if classifier_name == "RandomForest":
        print("RandomForest classifier selected")
        classifier_obj = RandomForestClassifier()
        rf_config = {
            "n_estimators": trial.suggest_int("n_estimators_rf", 8, 48),
            "max_depth": trial.suggest_int("max_depth", 1, 32),
            "class_weight": "balanced_subsample", 
            "max_features": trial.suggest_int("max_features", 1, 6),
            "n_jobs": args.data.num_workers,
            "verbose": 3,
        }
        classifier_obj.set_params(**rf_config)
    elif classifier_name == "SVC":
        print("SVC classifier selected")
        svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
        classifier_obj = LinearSVC(C=svc_c)
        X_train, X_val = scale_data(X_train, X_val)
    elif classifier_name == "xgboost":
        print("XGBoost classifier selected")
        xgb_params = {
            "max_depth": trial.suggest_int("max_depth", 1, 32),
            "n_estimators": trial.suggest_int("n_estimators_xgboost", 8, 100),
            "learning_rate": trial.suggest_float("learning_rate_xgboost", 1e-3, 1e1, log=True),
            "gamma": trial.suggest_float("gamma", 1e-10, 1e10, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-10, 1e10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-10, 1e10, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 32),
            "subsample": trial.suggest_float("subsample", 0.1, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
            "n_jobs": args.data.num_workers,
            "verbosity": 3,
        }
        le = LabelEncoder()
        y = le.fit_transform(y)
        y_val_df["evt"] = le.transform(y_val_df["evt"])
        classifier_obj = xgb.XGBClassifier(**xgb_params)
        update_registered_converter(
                 xgb.XGBClassifier,
        "XGBoostXGBClassifier",
        calculate_linear_classifier_output_shapes,
        convert_xgboost,
        options={"nocl": [True, False], "zipmap": [True, False, "columns"]},)
    elif classifier_name == "adaBoost":
        print("AdaBoost classifier selected")
        classifier_obj = AdaBoostClassifier()
        ab_config = {
            "n_estimators": trial.suggest_int("n_estimators_adaboost", 8, 100),
            "learning_rate": trial.suggest_float("learning_rate_adaboost", 1e-3, 1e1, log=True),
        }
        classifier_obj.set_params(**ab_config)
    elif classifier_name == "logisticRegression":
        print("LogisticRegression classifier selected")
        classifier_obj = LogisticRegression()
        lr_config = {
            "C": trial.suggest_float("C", 1e-10, 1e10, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
            "solver": "liblinear",
            "n_jobs": args.data.num_workers,
        }
        classifier_obj.set_params(**lr_config)
        X_train, X_val = scale_data(X_train, X_val)
    elif classifier_name == "knn":
        print("KNN classifier selected")
        classifier_obj = KNeighborsClassifier()
        knn_config = {
            "n_neighbors": trial.suggest_int("n_neighbors", 1, 32),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan"]),
        }
        classifier_obj.set_params(**knn_config)
        X_train, X_val = scale_data(X_train, X_val)
    elif classifier_name == "mlp":
        print("MLP classifier selected")
        classifier_obj = MLPClassifier()
        mlp_config = {
            "hidden_layer_sizes": trial.suggest_int("hidden_layer_sizes", 1, 32),
            "activation": trial.suggest_categorical("activation", ["identity", "logistic", "tanh", "relu"]),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "sgd", "adam"]),
            "alpha": trial.suggest_float("alpha", 1e-10, 1e10, log=True),
            "learning_rate": trial.suggest_categorical("learning_rate", ["constant", "invscaling", "adaptive"]),
        }
        classifier_obj.set_params(**mlp_config)
        X_train, X_val = scale_data(X_train, X_val)
    
    else:
        raise ValueError(f"Unknown classifier {classifier_name}")

    print("Fitting classifier")
    classifier_obj.fit(X_train, y)
    print("Classifier fitted")

    start_time = time.time()
    _pred = classifier_obj.predict(X_val)
    pred_time = time.time() - start_time


    if classifier_name == "adaBoost":
        classifier_obj.classes_ = classifier_obj.classes_.astype(np.int32) #fix for uint08 not supported. 

    onx = to_onnx(classifier_obj, X_val,target_opset={"":15,'ai.onnx.ml': 3})
    bytes = skl2onnx.helpers.onnx_helper.save_onnx_model(onx)
    print(f"Size of onnx model: {len(bytes)} bytes")

    if classifier_name == "xgboost":
        _pred +=1 
        y_val_df["evt"] = le.inverse_transform(y_val_df["evt"])

    y_hat_df = val_data.filter(["t", "x", "y", "status"])
    y_hat_df["evt"] = _pred

    assert len(y_hat_df) == len(y_val_df), "Prediction and ground truth have different lengths"
    y_hat_df = post_process(y_hat_df, args.pp_args)
    assert len(y_hat_df)!= 0, "Post processing removed all samples"

    # Temporary removal of samples made false by post processing
    mask2 = y_hat_df["status"] == True
    y_val_df = y_val_df[mask2]
    y_hat_df = y_hat_df[mask2]

    # reset index
    y_val_df.reset_index(drop=True, inplace=True)
    y_hat_df.reset_index(drop=True, inplace=True)

    scores = calculate_metrics(y_val_df, y_hat_df, args.jobs)
    metrics = []
    for metric in args.objective_metrics:
        if metric == "IoU_mcc":
            metrics.append(scores[0]["mcc"])
        elif metric == "pred_time":
            metrics.append(pred_time)
        elif metric == "model_size":
            metrics.append(len(bytes))
        else:
            raise ValueError(f"Unknown metric {metric}")
    return tuple(metrics)


def get_study_directions(objective_metrics):
    study_directions = []
    for metric in objective_metrics:
        if metric in ["IoU_mcc", "IoU_f1", "IoU_precision", "IoU_recall"]:
            study_directions.append("maximize")
        elif metric in ["pred_time","model_size"]:
            study_directions.append("minimize")
        else:
            raise ValueError(f"Unknown metric {metric}")
    return study_directions


def main():
    parser = ArgumentParser()
    parser.add_class_arguments(LookAtPointDataMiddleLabelModule, "data")
    parser.add_class_arguments(pp_args, "pp_args")
    parser.add_class_arguments(job, "jobs")
    parser.add_argument(
        "log_dir", type=str, default="/home/martin/Documents/Exjobb/eed/.experiments"
    )
    parser.add_argument("-c", "--config", action=ActionConfigFile)
    parser.add_argument("-n", "--study_name", default="TestSweep")
    parser.add_argument("-t", "--n_trials", default=100)
    parser.add_argument("-m", "--objective_metrics", default=["IoU_mcc"])

    args = parser.parse_args()
    args.jobs.event_map = utils.keys2num(args.jobs.event_map)
    log_path = Path(args.log_dir) / "sklearn_optuna_logs"
    log_path.mkdir(parents=True, exist_ok=True)
    print(f"Logging to {log_path}")

    study_directions = get_study_directions(args.objective_metrics)

    # jpath = utils.path2abs(args.job, root_repo)  # TODO: take this from config file.
    # with open(jpath, "r") as f:
    #    jobs = json.load(f)

    # Wrap the objective inside a lambda and call objective inside it
    objective = lambda trial: objective_function(trial, parser, args)

    # study = optuna.create_study(directions=["maximize","minimize"]) # maximize mcc, minimize pred_time
    storage_name = f"sqlite:///{args.study_name}.db"
    storage_name = r"{}".format(storage_name)
    if Path(storage_name).exists():
        print(f"Load existing study {args.study_name}")
        study = optuna.study.load_study(study_name=args.study_name, storage=storage_name)
    else:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=storage_name,
            directions=study_directions,
            load_if_exists=True,
        )
    study.set_metric_names(args.objective_metrics)
    study.optimize(objective, n_trials=args.n_trials)

    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trials:")
    trial = study.best_trials

    if len(args.objective_metrics) == 1:
        print("  Value: {}".format(trial[0].value))
    else:
        print("  Values: {}".format(trial[0].values))

    print("  Params: ")
    for key, value in trial[0].params.items():
        print("    {}: {}".format(key, value))

    f1 = optuna.visualization.plot_parallel_coordinate(study)
    f2 = optuna.visualization.plot_contour(study)
    f3 = optuna.visualization.plot_param_importances(study)
    f1.write_html(log_path / "parallel_coordinate.html")
    f2.write_html(log_path / "contour.html")
    f3.write_html(log_path / "param_importances.html")


if __name__ == "__main__":
    main()
