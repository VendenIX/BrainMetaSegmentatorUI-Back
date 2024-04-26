import os
import shutil
import tempfile
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import json

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    list_data_collate,
)

import torch
import pytorch_lightning
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from unetr.utilsUnetr.transforms import ResizeOrDoNothingd

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print_config()


directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = './'
print(root_dir)

base_dir = "dataset/"
img_paths = []
mask_paths = []

for patient_folder in os.listdir(base_dir):
    patient_path = os.path.join(base_dir, patient_folder)
    
    img_path = os.path.join(patient_path, "image.nii.gz")
    mask_path = os.path.join(patient_path, "mask_meta.nii.gz")
    
    if os.path.exists(img_path) and os.path.exists(mask_path):
        img_paths.append(img_path)
        mask_paths.append(mask_path)    
    
    
train_img_paths, temp_img_paths, train_mask_paths, temp_mask_paths = train_test_split(img_paths, mask_paths, test_size=0.2, random_state=42)

# Redivision de la validation et du test en deux parties

val_img_paths, test_img_paths, val_mask_paths, test_mask_paths = train_test_split(temp_img_paths, temp_mask_paths, test_size=0.5, random_state=42)



data_split = {
    "training" : {"image" : train_img_paths, "label" : train_mask_paths},
    "validation" : {"image" : val_img_paths, "label" : val_mask_paths},
    "test" : {"image" : test_img_paths, "label" : test_mask_paths}
}


class Net(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()

        self._model = UNETR(
            in_channels=1,
            out_channels=6,
            img_size=(96, 96, 96),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            conv_block=True,
            dropout_rate=0.0,
        ).to(device)

        self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        #self.post_pred = AsDiscrete(argmax=True, to_onehot=14)
        #self.post_label = AsDiscrete(to_onehot=14)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=True, num_classes=6)
        self.post_label = AsDiscrete(to_onehot=True, num_classes=6)

        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.max_epochs =1
        self.check_val = 30
        self.warmup_epochs = 20
        self.metric_values = []
        self.epoch_loss_values = []
        self.validation_step_outputs = []

    def forward(self, x):
        return self._model(x)

    def prepare_data(self):

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                ScaleIntensityRanged(
                    keys=["image"],
                    # a_min=-175,
                    a_min = -200,
                    a_max=300,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[0],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[1],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[2],
                    prob=0.10,
                ),
                RandRotate90d(
                    keys=["image", "label"],
                    prob=0.10,
                    max_k=3,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-175,
                    a_max=250,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                ResizeOrDoNothingd(keys=["image", "label"], max_spatial_size=(96, 96, 96)), # ramène l'image à la taille maximale acceptée par le modèle
            ]
        )

        self.train_ds = CacheDataset(
            data= [{'image' : img, "label": lbl} for img, lbl in zip(data_split['training']['image'], data_split['training']['label'])],
            transform=train_transforms,
            cache_num=24,
            cache_rate=1.0,
            num_workers=8,
        )
        self.val_ds = CacheDataset(
            data= [{'image' : img, "label": lbl} for img, lbl in zip(data_split['validation']['image'], data_split['validation']['label'])],
            transform=val_transforms,
            cache_num=6,
            cache_rate=1.0,
            num_workers=8,
        )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_ds,
            batch_size=1,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            collate_fn=list_data_collate,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        return val_loader

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=1e-4, weight_decay=1e-5)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = (batch["image"], batch["label"])
        output = self.forward(images)
        loss = self.loss_function(output, labels)
        tensorboard_logs = {"train_loss": loss.item()}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.epoch_loss_values.append(avg_loss.detach().cpu().numpy())

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        roi_size = (96, 96, 96)
        sw_batch_size = 4
        outputs = sliding_window_inference(images, roi_size, sw_batch_size, self.forward)
        loss = self.loss_function(outputs, labels)
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs, y=labels)
        d = {"val_loss": loss, "val_number": len(outputs)}
        self.validation_step_outputs.append(d)
        return d

    def on_validation_epoch_end(self):
        val_loss, num_items = 0, 0
        for output in self.validation_step_outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]
        mean_val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        mean_val_loss = torch.tensor(val_loss / num_items)
        
        # log val loss for modelcheckpoint to monitor
        self.log('val_loss', mean_val_loss, on_epoch=True, prog_bar=True, logger=True)
        
        tensorboard_logs = {
            "val_dice": mean_val_dice,
            "val_loss": mean_val_loss,
        }
        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch
        print(
            f"current epoch: {self.current_epoch} "
            f"current mean dice: {mean_val_dice:.4f}"
            f"\nbest mean dice: {self.best_val_dice:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
    
        self.metric_values.append(mean_val_dice)
        self.validation_step_outputs.clear()  # free memory
        return {"log": tensorboard_logs}


if __name__ == "__main__":
    net = Net()

    slice_map = {
            # "dataset/201709127/image.nii.gz" : 28.899,
            # "dataset/201704321/image.nii.gz" : 31.706,
            # "dataset/201709668/image.nii.gz" : 41.868,
            # "dataset/201705956/image.nii.gz" : 54.147,
            "image.nii.gz" : 20,
            # "dataset/201708552/image.nii.gz" : 47.003,
            # "dataset/201707259/image.nii.gz" : 56.608,
            # "dataset/201709600/image.nii.gz" : 26.502,
            # "dataset/201705438/image.nii.gz" : 9.831,
            # "dataset/201709795/image.nii.gz" : 29.178,
    }
    case_num = 0
    net.load_from_checkpoint(os.path.join(root_dir, "best_checkpoint.ckpt"))
    net.eval()
    net.to(device)
    net.prepare_data()

    with torch.no_grad():
            img_name = os.path.split(net.val_ds[case_num]['image_meta_dict']['filename_or_obj'])[-1]
            img = net.val_ds[case_num]["image"]
            label = net.val_ds[case_num]["label"]
            val_inputs = torch.unsqueeze(torch.from_numpy(img), 1).to(device)
            print(val_inputs.shape)
            val_labels = torch.unsqueeze(torch.from_numpy(img), 1).to(device)
            print(val_labels.shape)
            
            plt.figure("check", (18, 6))
            plt.subplot(1, 3, 1)
            plt.title("image")
            plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
            plt.subplot(1, 3, 2)
            plt.title("label")
            plt.imshow(val_labels.cpu().numpy()[0, 0, :, :, slice_map[img_name]])
            plt.subplot(1, 3, 3)
            val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, net, overlap=0.8)
            print(val_outputs.shape)
            plt.title("output")
            plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice_map[img_name]])
            plt.show()
