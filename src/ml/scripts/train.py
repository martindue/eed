"""
Define your training process here.
Reference: https://lightning.ai/docs/pytorch/stable/common/trainer.html
Example:
from ml.utils.constants import LOGGING_DIR

model = MyLightningModule()
datamodule = MyLightningDataModule()
trainer = Trainer(logger = TensorBoardLogger(EXPERIMENTS_DIR))
trainer.fit(model, data_module=datamodule)
"""
import sys
import os
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # Assuming the project root is two directories above src
sys.path.append(project_root)


from ml.utils.constants import LOGGING_DIR, DATA_DIR, EXPERIMENTS_DIR
from ml.engines.system import LitModel
from ml.datasets.lookAtPointDatasetMiddleLabel.dataset import LookAtPointDatasetMiddleLabel
from ml.datasets.lookAtPointDatasetMiddleLabel.datamodule import LookAtPointDataModule
from ml.models.model1 import TorchModel
from ml.models.modelLSTMmiddleLabel import TorchModel

model = LitModel(TorchModel(input_size=2, hidden_size=64, num_classes=6))
datamodule = LookAtPointDataModule(DATA_DIR)
trainer = Trainer(logger = TensorBoardLogger(EXPERIMENTS_DIR))
trainer.fit(model, datamodule)