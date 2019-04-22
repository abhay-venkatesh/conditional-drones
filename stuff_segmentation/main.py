from lib.icg import ICG
from lib.segnet import get_model
from pathlib import Path
from statistics import mean
from torch.utils.data import DataLoader
import csv
import torch
import torch.functional as F

BATCH_SIZE = 5
LR = 0.001
N_CLASSES = 13
NUM_EPOCHS = 80


def cross_entropy2d(output, target, weight=None):
    n, c, h, w = output.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between output and target
    if h != ht and w != wt:  # upsample labels
        output = F.interpolate(
            output, size=(ht, wt), mode="bilinear", align_corners=True)

    output = output.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(output, target, weight=weight)

    return loss


def get_iou(outputs, labels):
    SMOOTH = 1e-6
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most
    # probably be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    intersection = (outputs & labels).float().sum(
        (1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1,
                                            2))  # Will be zzero if both are 0
    iou = (intersection + SMOOTH) / (union + SMOOTH
                                     )  # We smooth our devision to avoid 0/0
    thresholded = torch.clamp(
        20 * (iou - 0.5), 0,
        10).ceil() / 10  # This is equal to comparing with thresolds
    return thresholded.mean()


def train():
    trainset = ICG(Path("D:/code/data/icg/training_set/train"))
    train_loader = DataLoader(
        dataset=trainset, batch_size=BATCH_SIZE, shuffle=True)

    valset = ICG(Path("D:/code/data/icg/training_set/test"))
    val_loader = DataLoader(dataset=valset, batch_size=BATCH_SIZE)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(n_classes=N_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(NUM_EPOCHS):

        model.train()
        total_loss = 0
        for X, Y in train_loader:
            Y_ = model(X)
            loss = cross_entropy2d(Y_, Y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        avg_loss = total_loss / len(train_loader)

        model.eval()
        ious = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels[0].long()
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                iou = get_iou(predicted, labels)
                ious.append(iou.item())

        mean_iou = mean(ious)
        with open("log.csv", mode='a', newline='') as logfile:
            writer = csv.writer(
                logfile,
                delimiter=',',
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL)
            writer.writerow([epoch, avg_loss, mean_iou])


if __name__ == "__main__":
    train()