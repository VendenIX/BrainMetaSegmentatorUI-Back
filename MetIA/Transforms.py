import torch
import monai.transforms as transforms
from unetr.utilsUnetr.transforms import CropBedd


"""
Crée une série de transformations pour le traitement d'images.

Returns:
    torch.nn.Sequential: Composition de transformations pour le traitement d'images DICOM.
"""

class TransformationFactory:
    @staticmethod
    def create_transform():
        dtype = torch.float32
        voxel_space = (1.5, 1.5, 2.0)
        a_min = -200.0
        a_max = 300
        b_min = 0.0
        b_max = 1.0
        clip = True
        crop_bed_max_number_of_rows_to_remove = 0
        crop_bed_max_number_of_cols_to_remove = 0
        crop_bed_min_spatial_size = (300, -1, -1)
        enable_fgbg2indices_feature = False
        pos = 1.0
        neg = 1.0
        num_samples = 1
        roi_size = (96, 96, 96)
        random_flip_prob = 0.2
        random_90_deg_rotation_prob = 0.2
        random_intensity_scale_prob = 0.1
        random_intensity_shift_prob = 0.1
        val_resize = (-1, -1, 250)

        spacing = transforms.Identity()
        if all([space > 0.0 for space in voxel_space]):
            spacing = transforms.Spacingd(
                keys=["image", "label"], pixdim=voxel_space, mode=("bilinear", "nearest")
            )

        transform = transforms.Compose(
            [
                transforms.Orientationd(keys=["image", "label"], axcodes="LAS", allow_missing_keys=True),
                spacing,
                transforms.ScaleIntensityRanged(
                    keys=["image"], a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=clip, allow_missing_keys=True
                ),
                CropBedd(
                    keys=["image", "label"], image_key="image",
                    max_number_of_rows_to_remove=crop_bed_max_number_of_rows_to_remove,
                    max_number_of_cols_to_remove=crop_bed_max_number_of_cols_to_remove,
                    min_spatial_size=crop_bed_min_spatial_size,
                    axcodes_orientation="LAS",
                ),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image", allow_missing_keys=True),
                transforms.RandFlipd(keys=["image", "label"], prob=random_flip_prob, spatial_axis=0, allow_missing_keys=True),
                transforms.RandFlipd(keys=["image", "label"], prob=random_flip_prob, spatial_axis=1, allow_missing_keys=True),
                transforms.RandFlipd(keys=["image", "label"], prob=random_flip_prob, spatial_axis=2, allow_missing_keys=True),
                transforms.RandRotate90d(keys=["image", "label"], prob=random_90_deg_rotation_prob, max_k=3, allow_missing_keys=True),
                transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=random_intensity_scale_prob),
                transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=random_intensity_shift_prob),
                transforms.ToTensord(keys=["image", "label"], dtype=dtype),
            ]
        )
        return transform