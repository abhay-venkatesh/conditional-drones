from lib.utils.logger import Logger
from pathlib import Path
import os
import torch


class Trainer:
    def __init__(self, experiment):
        self.experiment = experiment
        self.device = torch.device('cuda' if torch.cuda.
                                   is_available() else 'cpu')
        self.logger = Logger(self.experiment.stats_folder)

    def train(self):
        raise NotImplementedError

    def _load_checkpoint(self, model):
        start_epochs = 0
        if self.experiment.config["checkpoint path"]:
            start_epochs = int(
                Path(self.experiment.config["checkpoint path"]).stem)
            model.load_state_dict(
                torch.load(Path(self.experiment.config["checkpoint path"])))
        return start_epochs

    def _save_checkpoint(self, epoch, model, retain=False):
        checkpoint_filename = str(epoch + 1) + ".ckpt"
        checkpoint_path = Path(self.experiment.checkpoints_folder,
                               checkpoint_filename)

        if not retain:
            prev_checkpoint_filename = str(epoch) + ".ckpt"
            prev_checkpoint_path = Path(self.experiment.checkpoints_folder,
                                        prev_checkpoint_filename)
            if os.path.exists(prev_checkpoint_path):
                os.remove(prev_checkpoint_path)

        torch.save(model.state_dict(), checkpoint_path)
