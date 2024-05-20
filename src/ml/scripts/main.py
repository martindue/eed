import cProfile
import os
import sys

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)  # Assuming the project root is two directories above src
sys.path.append(project_root)


# main.py
from lightning.pytorch.cli import LightningCLI

from ml.datasets.lookAtPointDatasetMiddleLabel.datamodule import (
    LookAtPointDataMiddleLabelModule,
)
from ml.engines.system import LitModule
from ml.models.modelLSTMmiddleLabel import TorchModel

# Add the project root directory to the Python path


def main():
    class LightningLitModule(LightningCLI):
        pass

    cli = LightningLitModule(LitModule, LookAtPointDataMiddleLabelModule)


if __name__ == "__main__":
    main()
