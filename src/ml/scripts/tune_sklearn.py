# TODO: Make this take in an arbitrary data module as set by config file

import optuna
import os, sys
import lightning.pytorch as lp
from lightning.pytorch.cli import (
    SaveConfigCallback,
    LightningArgumentParser,
    LightningCLI,
)
import sklearn.model_selection
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
from jsonargparse import ActionConfigFile, CLI, ArgumentParser
import numpy as np

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)  # Assuming the project root is two directories above src
sys.path.append(project_root)

from ml.scripts.main_sklearn import main as main_sklearn
from ml.datasets.lookAtPointDatasetNoWindow.datamodule import (
    LookAtPointDataModuleNoWindow,
)


def objective_function(trial: optuna.trial.Trial, parser: ArgumentParser, args) -> float:
    data_module = LookAtPointDataModuleNoWindow(data_dir=args.data.data_dir)
    data_module.prepare_data()
    data_module.setup(stage="fit")
    data_loader = data_module.train_dataloader()

    for data in data_loader:
        X = data["input"]
        X = X.squeeze()
        y = data["evt"]

    X = X.numpy()
    y = y.numpy()
    X = np.nan_to_num(X, nan=0)  # placeholder for missing values TODO: change to something more sensible
    print("dimensions of X and y:", X.shape, y.shape)

    # classifier_name = trial.suggest_categorical('classifier', ['SVC', 'RandomForest'])
    classifier_name = "RandomForest"
    if classifier_name == "RandomForest":
        classifier_obj = RandomForestClassifier()
        rf_config = {
            "n_estimators": trial.suggest_int("n_estimators", 8, 128),
            "max_depth": None,
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"]),
            "max_features": trial.suggest_int("max_features", 1, 5),
            "n_jobs": args.data.num_workers,
            "verbose": 3,
        }
        classifier_obj.set_params(**rf_config)
    else:
        pass

    score = sklearn.model_selection.cross_val_score(classifier_obj, X, y, n_jobs=-1, cv=3)
    accuracy = score.mean()

    return accuracy


def main():
    parser = ArgumentParser()
    parser.add_class_arguments(LookAtPointDataModuleNoWindow, "data")
    parser.add_argument("log_dir", type=str, default="/home/martin/Documents/Exjobb/eed/.experiments")
    parser.add_argument("-c", "--config", action=ActionConfigFile)
    parser.add_argument("-n", "--study_name", default="TestSweep")
    parser.add_argument("-t", "--n_trials", default=100)

    args = parser.parse_args()

    log_path = Path(args.log_dir) / "sklearn_optuna_logs"
    print(f"Logging to {log_path}")

    # Wrap the objective inside a lambda and call objective inside it
    objective = lambda trial: objective_function(trial, parser, args)

    study = optuna.create_study(direction="minimize")

    storage_name = f"sqlite:///{args.study_name}.db"
    storage_name = r"{}".format(storage_name)
    if Path(storage_name).exists():
        print(f"Load existing study {args.study_name}")
        study = optuna.study.load_study(study_name=args.study_name, storage=storage_name)
    else:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=storage_name,
            direction="minimize",
            load_if_exists=True,
        )

    study.optimize(objective, n_trials=args.n_trials)

    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial
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
