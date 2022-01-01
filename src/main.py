import hydra
import pytorch_lightning as pl
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything

from config import BirdConfig
from modules import BirdSpeciesDataModule, BirdSpeciesModule

cs = ConfigStore.instance()
cs.store(name="bird_config", node=BirdConfig)


@hydra.main(config_path="conf", config_name="config")
def main(config: BirdConfig) -> None:
    print(OmegaConf.to_yaml(config))

    seed_everything(config.params.seed)

    datamodule = BirdSpeciesDataModule(config)
    model = BirdSpeciesModule(config)

    es_callback = EarlyStopping(monitor="val_loss")
    lr_callback = callbacks.LearningRateMonitor()
    checkpoint_callback = callbacks.ModelCheckpoint(
        filename="best_loss",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        save_last=True,
    )
    logger = TensorBoardLogger(config.params.model.name)
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=config.params.epochs,
        auto_lr_find=True,
        callbacks=[es_callback, lr_callback, checkpoint_callback],
        **config.params.trainer
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()