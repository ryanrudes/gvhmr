from pytorch_lightning.callbacks import LearningRateMonitor

from gvhmr.configs import MainStore, builds

MainStore.store(name="pl", node=builds(LearningRateMonitor), group="callbacks/lr_monitor")
