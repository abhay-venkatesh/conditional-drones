from lib.datasets.icg_stuff import ICGStuff, ICGStuffBuilder
from lib.models.segnet import get_model
from lib.trainers.functional import cross_entropy2d, get_iou
from lib.trainers.trainer import Trainer
from pathlib import Path
from statistics import mean
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch


class ICGStuffTrainer(Trainer):
    def train(self):
        trainset_folder, valset_folder = ICGStuffBuilder().build(
            self.experiment.config["dataset path"])

        trainset = ICGStuff(trainset_folder)
        train_loader = DataLoader(
            dataset=trainset,
            batch_size=self.experiment.config["batch size"],
            shuffle=True)

        valset = ICGStuff(valset_folder)
        val_loader = DataLoader(
            dataset=valset, batch_size=self.experiment.config["batch size"])

        model = get_model(n_classes=trainset.N_CLASSES).to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.experiment.config["learning rate"])

        start_epochs = 0
        if self.experiment.config["checkpoint path"]:
            start_epochs = int(
                Path(self.experiment.config["checkpoint path"]).stem)
            model.load_state_dict(
                torch.load(Path(self.experiment.config["checkpoint path"])))

        for epoch in tqdm(
                range(start_epochs, self.experiment.config["epochs"])):

            model.train()
            total_loss = 0
            for X, Y in tqdm(train_loader):
                X, Y = X.to(self.device), Y.long().to(self.device)
                Y_ = model(X)
                loss = cross_entropy2d(Y_, Y)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            avg_loss = total_loss / len(train_loader)
            self.logger.log("epoch", epoch, "avg_loss", avg_loss)

            model.eval()
            ious = []
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels[0].long()
                    labels = labels.to(self.device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    iou = get_iou(predicted, labels)
                    ious.append(iou.item())

            mean_iou = mean(ious)
            self.logger.log("epoch", epoch, "mean_iou", mean_iou)

            self.logger.graph()

            checkpoint_filename = str(epoch + 1) + ".ckpt"
            checkpoint_path = Path(self.experiment.checkpoints_folder,
                                   checkpoint_filename)
            torch.save(model.state_dict(), checkpoint_path)
