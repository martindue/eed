import os, sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # Assuming the project root is two directories above src
sys.path.append(project_root)


# main.py
from lightning.pytorch.cli import LightningCLI
from ml.engines.system import LitModel
from ml.models.modelLSTMmiddleLabel import TorchModel
from ml.datasets.lookAtPointDatasetMiddleLabel.datamodule import LookAtPointDataModule
# Add the project root directory to the Python path

def main():
    class LightningLitModel(LightningCLI):
        pass
    cli = LightningLitModel(LitModel ,LookAtPointDataModule)

if __name__ == "__main__":
    main()
