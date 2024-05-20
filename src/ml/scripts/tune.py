import os
import sys
from pathlib import Path

import lightning.pytorch as lp
import optuna
from jsonargparse import ActionConfigFile
from lightning.pytorch.cli import (
    LightningArgumentParser,
    LightningCLI,
    SaveConfigCallback,
)

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)  # Assuming the project root is two directories above src
sys.path.append(project_root)

from ml.datasets.lookAtPointDatasetMiddleLabel.datamodule import LookAtPointDataModule
from ml.engines.system import LitModule


def objective_function(
    trial: optuna.trial.Trial, parser: LightningArgumentParser, args
) -> float:
    hidden_size = trial.suggest_int("hidden_size", 64, 256, 64)
    args.model.model.init_args.hidden_size = hidden_size

    experiment = parser.instantiate_classes(args)

    experiment.trainer.callbacks.append(SaveConfigCallback(parser, args))

    hyperparameters = dict(hidden_size=hidden_size)

    experiment.trainer.logger.log_hyperparams(hyperparameters)
    experiment.trainer.fit(experiment.model, datamodule=experiment.data)

    return experiment.trainer.callback_metrics["train_loss"].item()


def main():
    parser = LightningArgumentParser()
    parser.add_class_arguments(LookAtPointDataModule, "data")
    parser.add_class_arguments(LitModule, "model")
    parser.add_class_arguments(lp.Trainer, "trainer")
    parser.add_optimizer_args()
    parser.add_lr_scheduler_args()
    parser.add_argument("-c", "--config", action=ActionConfigFile)
    parser.add_argument("-n", "--study_name", default="TestSweep")
    parser.add_argument("-t", "--n_trials", default=100)

    args = parser.parse_args()

    log_path = Path(args.trainer.logger.init_args.save_dir) / "optuna_logs"
    print(f"Logging to {log_path}")

    # Wrap the objective inside a lambda and call objective inside it
    objective = lambda trial: objective_function(trial, parser, args)

    study = optuna.create_study(direction="maximize")

    storage_name = f"sqlite:///{args.study_name}.db"
    storage_name = r"{}".format(storage_name)
    if Path(storage_name).exists():
        print(f"Load existing study {args.study_name}")
        study = optuna.study.load_study(study_name=args.study_name, storage=storage_name)
    else:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=storage_name,
            direction="maximize",
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
