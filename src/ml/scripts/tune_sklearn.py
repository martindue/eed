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
import sklearn.model_selection
from jsonargparse import CLI, ActionConfigFile, ArgumentParser
from lightning.pytorch.cli import (
    LightningArgumentParser,
    LightningCLI,
    SaveConfigCallback,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)  # Assuming the project root is two directories above src
sys.path.append(project_root)

import eval.misc.utils as utils
from eval.misc import eval_utils, matching
from funcy import omit

# from ml.scripts.main_sklearn import main as main_sklearn
from ml.datasets.lookAtPointDatasetMiddleLabel.datamodule import (
    LookAtPointDataMiddleLabelModule,
)
from ml.utils.post_processing import post_process
from ml.utils.classes import job, pp_args
from ml.utils.helpers import impute_with_column_means, linear_interpol_with_pandas


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
    data_module = LookAtPointDataMiddleLabelModule(
        data_dir=args.data.data_dir,
        sklearn=True,
        training_datasets=args.data.training_datasets,
    )
    data_module.prepare_data()
    data_module.setup(stage="fit")
    train_data_loader = data_module.train_dataloader()
    val_data_loader = data_module.val_dataloader()

    for data in train_data_loader:
        X_train = data["features"]
        X_train = X_train.squeeze()
        y_train_df = pd.DataFrame(omit(data, ["features"]))
        y_train_df.rename(columns={"label": "evt"}, inplace=True)
        y_train_df.replace({"evt": args.jobs.event_map}, inplace=True)
        y = data["label"]
        # Replace values in numpy array using dictionary
        for old_val, new_val in args.jobs.event_map.items():
            y[y == old_val] = new_val
        y.numpy()

    for data in val_data_loader:
        X_val = data["features"]
        X_val = X_val.squeeze()
        val_data = pd.DataFrame(omit(data, ["features"]))
        y_val_df = val_data.filter(["t", "x", "y", "status", "label"])
        y_val_df.rename(columns={"label": "evt"}, inplace=True)
        y_val_df.replace({"evt": args.jobs.event_map}, inplace=True)

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

    #classifier_name = trial.suggest_categorical("classifier", ["SVC", "RandomForest", "xgboost"])
    classifier_name = "RandomForest"
    #classifier_name = "xgboost"

    if classifier_name == "RandomForest":
        classifier_obj = RandomForestClassifier()
        rf_config = {
            "n_estimators": trial.suggest_int("n_estimators", 8, 48),
            "max_depth": trial.suggest_int("max_depth", 1, 32),
            "class_weight": trial.suggest_categorical(
                "class_weight", ["balanced", "balanced_subsample"]
            ),
            "max_features": trial.suggest_int("max_features", 1, 6),
            "n_jobs": args.data.num_workers,
            "verbose": 3,
        }
        classifier_obj.set_params(**rf_config)
    elif classifier_name == "SVC":
        svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
        classifier_obj = LinearSVC(C=svc_c)
    elif classifier_name == "xgboost":
        xgb_params = {
            "max_depth": trial.suggest_int("max_depth", 1, 32),
            "n_estimators": trial.suggest_int("n_estimators", 8, 48),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e1, log=True),
            "gamma": trial.suggest_float("gamma", 1e-10, 1e10, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-10, 1e10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-10, 1e10, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 32),
            "subsample": trial.suggest_float("subsample", 0.1, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
            "n_jobs": args.data.num_workers,
            "verbosity": 3,
        }
        #le = LabelEncoder()
        #y = le.fit_transform(y)
        #y_val_df["evt"] = le.transform(y_val_df["evt"])
        classifier_obj = xgb.XGBClassifier(**xgb_params)


    else:
        raise ValueError(f"Unknown classifier {classifier_name}")

    classifier_obj.fit(X_train, y)

    start_time = time.time()
    _pred = classifier_obj.predict(X_val)
    pred_time = time.time() - start_time

    y_hat_df = val_data.filter(["t", "x", "y", "status"])
    y_hat_df["evt"] = _pred

    y_hat_df = post_process(y_hat_df, args.pp_args)

    # Temporary removal of samples made false by post processing
    mask2 = y_hat_df["status"] == True
    y_val_df = y_val_df[mask2]
    y_hat_df = y_hat_df[mask2]

    # reset index
    y_val_df.reset_index(drop=True, inplace=True)
    y_hat_df.reset_index(drop=True, inplace=True)

    scores = calculate_metrics(y_val_df, y_hat_df, args.jobs)

    if "IoU_mcc" in args.objective_metrics and "pred_time" in args.objective_metrics:
        return scores[0]["mcc"], pred_time
    elif "IoU_mcc" in args.objective_metrics:
        return scores[0]["mcc"]
    elif "pred_time" in args.objective_metrics:
        return pred_time
    else:
        raise ValueError(f"Unknown metric {args.objective_metrics}")


def get_study_directions(objective_metrics):
    study_directions = []
    for metric in objective_metrics:
        if metric in ["IoU_mcc", "IoU_f1", "IoU_precision", "IoU_recall"]:
            study_directions.append("maximize")
        elif metric in ["pred_time"]:
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
    print("  Value: {}".format(trial[0].value))
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
