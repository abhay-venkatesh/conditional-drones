from lib.metrics import get_iou
from pathlib import Path
from statistics import mean
from tqdm import tqdm
import csv
import importlib
import lib.functional
import torch


class Trainer:
    def __init__(self, config):
        model_module = "lib.models." + config["model"]
        self.experiment_name = config["name"]

        # Data parameters
        self.batch_size = config["batch size"]

        loaders = get_icg_loaders(config["dataset"], self.batch_size)
        self.train_loader, self.val_loader, self.n_classes = loaders

        # Model parameters
        self.model = importlib.import_module(model_module).get_model(
            self.n_classes)
        self.device = torch.device('cuda' if torch.cuda.
                                   is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.loss_fn = getattr(lib.functional, config["loss function"])

        # Training hyperparameters
        self.num_epochs = config["num epochs"]
        self.lr = config["learning rate"]
        self.checkpoint_path = config["checkpoint path"]

        self.start_epochs = 0
        if self.checkpoint_path:
            self.start_epochs = int(Path(self.checkpoint_path).stem)
            self.model.load_state_dict(torch.load(self.checkpoint_path))

        self.stats_folder = config["stats folder"]
        self.checkpoints_folder = config["checkpoints folder"]

    def _log_row(self, filename, row):
        filepath = Path(self.stats_folder, filename)
        with open(filepath, mode='a', newline='') as logfile:
            writer = csv.writer(
                logfile,
                delimiter=',',
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL)
            writer.writerow(row)

    def _log(self, filename, rows):
        filepath = Path(self.stats_folder, filename)
        with open(filepath, mode='a', newline='') as logfile:
            writer = csv.writer(
                logfile,
                delimiter=',',
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL)
            for row in rows:
                writer.writerow(row)

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        total_step = len(self.train_loader)
        for epoch in tqdm(range(self.start_epochs, self.num_epochs)):
            self.model.train()
            stats = []
            for step, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                target = labels[0].long()
                target = target.to(self.device)

                output = self.model(images)
                loss = self.loss_fn(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if False:
                    print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(
                        epoch + 1, self.num_epochs, step + 1, total_step,
                        loss.item()))
                stats.append([epoch, step, loss.item()])

            mean_iou = self.validate(loss)
            self._log_row("mean_ious.csv", [epoch, mean_iou])
            self._log("losses.csv", stats)
            checkpoint_filename = str(epoch + 1) + ".ckpt"
            checkpoint_path = Path(self.checkpoints_folder,
                                   checkpoint_filename)
            torch.save(self.model.state_dict(), checkpoint_path)

    def validate(self, loss):
        self.model.eval()
        ious = []
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels[0].long()
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                iou = get_iou(predicted, labels)
                ious.append(iou.item())

        mean_iou = mean(ious)
        print("")
        print('Loss: {:.4f}, Mean IoU: {}'.format(loss.item(), mean_iou))
        return mean_iou