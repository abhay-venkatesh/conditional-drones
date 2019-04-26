from lib.datasets.icg_stuff import ICGStuff
from lib.datasets.translated_unreal import TranslatedUnreal
from lib.models.segnet import get_model
from pathlib import Path
from torch.utils.data import DataLoader
import torch


class TranslatedEvaluator:
    DATASET_PATH = Path("D:/code/data/unreal_translated")
    CHECKPOINT_PATH = Path(
        "D:/code/src/conditional-drones/stuff_segmentation/experiments/" +
        "icg_stuff_tesla/checkpoints/163.ckpt")
    N_CLASSES = 9
    OUTPUTS_FOLDER = Path("D:/code/src/conditional-drones/stuff_segmentation" +
                          "/experiments/translated")
    BATCH_SIZE = 4

    def evaluate(self):
        dataset = TranslatedUnreal(self.DATASET_PATH)
        val_loader = DataLoader(dataset=dataset, batch_size=self.BATCH_SIZE)
        model = get_model(n_classes=self.N_CLASSES).cuda()
        model.load_state_dict(torch.load(self.CHECKPOINT_PATH))
        model.eval()
        with torch.no_grad():
            for i, images in enumerate(val_loader):
                images = images.cuda()
                output = model(images)
                _, predicted = torch.max(output.data, 1)
                ICGStuff.visualize_prediction(predicted, i,
                                              self.OUTPUTS_FOLDER)
