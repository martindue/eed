"""
Define your predict process here.
Reference: https://lightning.ai/docs/pytorch/stable/common/trainer.html
Example:

model = MyLightningModule()
datamodule = MyLightningDataModule()
trainer = Trainer(logger = TensorBoardLogger(EXPERIMENTS_DIR))
trainer.predict(model, data_module=datamodule)
"""
import os, sys
import torch
from lightning.pytorch.cli import LightningCLI

from jsonargparse import CLI



project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)  # Assuming the project root is two directories above src
sys.path.append(project_root)


from ml.engines.system import LitModule
from ml.datasets.lookAtPointDatasetMiddleLabel.datamodule import LookAtPointDataMiddleLabelModule


def predict():
    #chkpt = torch.load("/home/martin/Documents/Exjobb/eed/.experiments/logs2/EED/version_112/checkpoints/epoch=0-step=2013.ckpt")
    #print(chkpt.keys())

    litModule = LitModule()
    model = LitModule.load_from_checkpoint("/home/martin/Documents/Exjobb/eed/.experiments/logs2/EED/version_112/checkpoints/epoch=0-step=2013.ckpt")
    model.eval()

    data_module = LookAtPointDataMiddleLabelModule()
    data_module.setup("test")

    test_data_loader = data_module.test_dataloader()

    for batch in test_data_loader:
        x, y = batch
        y_hat = model(x)
        print(y_hat)
        print(y)
        break

if __name__ == "__main__":
    CLI(predict)