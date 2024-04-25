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
from lightning.pytorch.cli import (LightningArgumentParser, LightningCLI,
                                   SaveConfigCallback)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)  # Assuming the project root is two directories above src
sys.path.append(project_root)

import eval.misc.utils as utils
from eval.misc import eval_utils, matching
from funcy import omit

# from ml.scripts.main_sklearn import main as main_sklearn
from ml.datasets.lookAtPointDatasetMiddleLabel.datamodule import \
    LookAtPointDataMiddleLabelModule
from ml.utils.classes import job
from ml.utils.helpers import impute_with_column_means


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


def objective_function(
    trial: optuna.trial.Trial, parser: ArgumentParser, args
) -> float:
    data_module = LookAtPointDataMiddleLabelModule(
        data_dir=args.data.data_dir, sklearn=True
    )
    data_module.prepare_data()
    data_module.setup(stage="fit")
    train_data_loader = data_module.train_dataloader()
    val_data_loader = data_module.val_dataloader()

    for data in train_data_loader:
        X_train = data["features"]
        X_train = X_train.squeeze()
        y = data["label"]
    for data in val_data_loader:
        X_val = data["features"]
        X_val = X_val.squeeze()
        val_data = pd.DataFrame(omit(data, ["features"]))
        y_df = val_data.filter(["t", "x", "y", "status", "label"])
        y_df.rename(columns={"label": "evt"}, inplace=True)
    print("Data loaded")
    X_train = X_train.numpy()
    y = y.numpy()
    X_train = impute_with_column_means(X_train)

    X_val = X_val.numpy()
    X_val = impute_with_column_means(X_val)

    print("dimensions of X and y:", X_train.shape, y.shape)

    classifier_name = trial.suggest_categorical("classifier", ["SVC", "RandomForest"])
    # classifier_name = "RandomForest"
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

    else:
        raise ValueError(f"Unknown classifier {classifier_name}")

    classifier_obj.fit(X_train, y)

    start_time = time.time()
    _pred = classifier_obj.predict(X_val)
    pred_time = time.time() - start_time

    y_hat_df = val_data.filter(["t", "x", "y", "status"])
    y_hat_df["evt"] = _pred
    scores = calculate_metrics(y_df, y_hat_df, args.jobs)
    # score = sklearn.model_selection.cross_val_score(
    #    classifier_obj, X, y, n_jobs=-1, cv=3
    # )
    # score = score.mean()

    return scores[0]["mcc"], pred_time  # return mcc score for multiclass all events


def main():
    parser = ArgumentParser()
    parser.add_class_arguments(LookAtPointDataMiddleLabelModule, "data")
    parser.add_class_arguments(job, "jobs")
    parser.add_argument(
        "log_dir", type=str, default="/home/martin/Documents/Exjobb/eed/.experiments"
    )
    parser.add_argument("-c", "--config", action=ActionConfigFile)
    parser.add_argument("-n", "--study_name", default="TestSweep")
    parser.add_argument("-t", "--n_trials", default=100)

    args = parser.parse_args()

    log_path = Path(args.log_dir) / "sklearn_optuna_logs"
    log_path.mkdir(parents=True, exist_ok=True)
    print(f"Logging to {log_path}")

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
        study = optuna.study.load_study(
            study_name=args.study_name, storage=storage_name
        )
    else:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=storage_name,
            directions=["maximize", "minimize"],
            load_if_exists=True,
        )
    study.set_metric_names(["IoU_mcc", "pred_time"])
    study.optimize(objective, n_trials=args.n_trials)

    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trials:")
    trial = study.best_trials
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    f1 = optuna.visualization.plot_parallel_coordinate(study)
    f2 = optuna.visualization.plot_contour(study)
    f3 = optuna.visualization.plot_param_importances(study)
    f1.write_html(log_path / "parallel_coordinate.html")
    f2.write_html(log_path / "contour.html")
    f3.write_html(log_path / "param_importances.html")


if __name__ == "__main__":
    main()
