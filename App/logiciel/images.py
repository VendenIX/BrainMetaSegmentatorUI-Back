from unetr.utilsUnetr.transforms import CropBedd, RandCropByPosNegLabeld, ResizeOrDoNothingd
from functools import partial
from monai.inferers import sliding_window_inference
from unetr.networks.unetr import UNETR
import os
from unetr.model_module import SegmentationTask
from monai.transforms import LoadImage
import numpy as np
import matplotlib.pyplot as plt
import time 
import torch
from torchvision.utils import save_image
import cv2 as cv
import monai.transforms as transforms
import scipy
from PIL import Image
import rt_utils

class Images():
	def __init__(self, image):
		self.image, self.masks, self.excel = self.getLabelOfIRM(image)

	def transformsImages(self):
		dtype= torch.float32
		voxel_space =(1.5, 1.5, 2.0)
		a_min=-200.0
		a_max=300
		b_min=0.0
		b_max=1.0
		clip=True
		crop_bed_max_number_of_rows_to_remove=0
		crop_bed_max_number_of_cols_to_remove=0
		crop_bed_min_spatial_size=(300, -1, -1)
		enable_fgbg2indices_feature=False
		pos=1.0
		neg=1.0
		num_samples=1
		roi_size=(96, 96, 96)
		random_flip_prob=0.2
		random_90_deg_rotation_prob=0.2
		random_intensity_scale_prob=0.1
		random_intensity_shift_prob=0.1
		val_resize=(-1, -1, 250)

		spacing = transforms.Identity()
		if all([space > 0.0 for space in voxel_space]):
			spacing = transforms.Spacingd(keys=["image", "label"], pixdim=voxel_space, mode=("bilinear", "nearest")) # to change the dimension of the voxel to have less data to compute

			posneg_label_croper_kwargs = {"keys": ["image", "label"],
					"label_key": "label",
					"spatial_size": roi_size,
					"pos": pos,
					"neg": neg,
					"num_samples": num_samples,
					"image_key": "image",
					"allow_smaller": True,}

			fgbg2indices = transforms.Identity()
			if enable_fgbg2indices_feature:
				fgbg2indices = transforms.FgBgToIndicesd(keys=["image", "label"], image_key="label", image_threshold=0.0) # to crop samples close to the label mask
				posneg_label_croper_kwargs["fg_indices_key"] = "image_fg_indices"
				posneg_label_croper_kwargs["bg_indices_key"] = "image_bg_indices"
			else:
				posneg_label_croper_kwargs["image_threshold"] = 0.0

		transform = transforms.Compose(
                [
                    transforms.Orientationd(keys=["image", "label"], axcodes="LAS", allow_missing_keys=True), # to have the same orientation
                    spacing,
                    transforms.ScaleIntensityRanged(
                        keys=["image"], a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=clip, allow_missing_keys=True
                    ), # scales image from a values to b values
                    CropBedd(
                        keys=["image", "label"], image_key="image",
                        max_number_of_rows_to_remove=crop_bed_max_number_of_rows_to_remove,
                        max_number_of_cols_to_remove=crop_bed_max_number_of_cols_to_remove,
                        min_spatial_size=crop_bed_min_spatial_size,
                        axcodes_orientation="LAS",
                    ), # crop the bed from the image (useless data)
                    transforms.CropForegroundd(keys=["image", "label"], source_key="image", allow_missing_keys=True), # remove useless background image part
                    fgbg2indices,
                    transforms.RandFlipd(keys=["image", "label"], prob=random_flip_prob, spatial_axis=0, allow_missing_keys=True), # random flip on the X axis
                    transforms.RandFlipd(keys=["image", "label"], prob=random_flip_prob, spatial_axis=1, allow_missing_keys=True), # random flip on the Y axis
                    transforms.RandFlipd(keys=["image", "label"], prob=random_flip_prob, spatial_axis=2, allow_missing_keys=True), # random flip on the Z axis
                    transforms.RandRotate90d(keys=["image", "label"], prob=random_90_deg_rotation_prob, max_k=3, allow_missing_keys=True), # random 90 degree rotation
                    transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=random_intensity_scale_prob), # random intensity scale
                    transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=random_intensity_shift_prob), # random intensity shifting
                    transforms.ToTensord(keys=["image", "label"], dtype=dtype), # to have a PyTorch tensor as output
                ]
            )
		return transform

	def loadModel(self):
		model= SegmentationTask.load_from_checkpoint("logiciel/trainedModel/checkpoint-epoch=1599-val_loss=0.225.ckpt")
		model.eval()
		return model

	def applyTransforms(self, transform, image):
		image={"image":image, "label":torch.zeros_like(image),"patient_id":'201905984', "has_meta":True}
		image=transform(image)
		return image

	def applyUNETR(self, dicoImage, model):
		label =sliding_window_inference(inputs=dicoImage["image"][None], 
										roi_size=(96, 96, 96), 
										sw_batch_size=4,
										predictor=model,
										overlap=0.5)
		print(label.shape)
		label = torch.argmax(label, dim=1, keepdim=True)
		print(label.shape)
		size=label.shape
		dicoImage["label"]=label.reshape((size[1], size[2], size[3], size[4]))
		return dicoImage

	def disapplyTransforms(self, transform, dicoImage):
		dicoImage = transform.inverse(dicoImage)
		return dicoImage["label"]

	def labelButtun(self, nbMask, color, slicesLabels, volumePixel, MaskButton, MaskButtonTxt):
		MaskButtonTxt.append(tk.StringVar())
		MaskButton.append(tk.Button(root, textvariable= MaskButtonTxt[nbMask], font=fontStyle, bg=color, fg=white, height=2, width=30))
		MaskButtonTxt[nbMask].set("GTV"+str(nbMask+1)+"_Volume:"+str(volumePixel)+"_Slices:"+str(min(slicesLables)) +"-"+ str(max(slicesLabels)))
		MaskButton[nbMask].grid(columns=1, row=2, sticky="W", padx=widthScreen/20)
		#ici créer les boutons de labels str(min(slicesLables)) +"-"+ str(max(slicesLabels))
		return "No Buttun now"
		
	def create_masked_image(self, img, mask, alpha, color):
		color = np.asarray(color).reshape(1,1,3)
		#colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
		#img = np.expand_dims(img, 0).repeat(3, axis=0)
		image_overlay = np.ma.MaskedArray(img, mask=mask, fill_value=color)
		image_overlay = image_overlay.filled()
		image_combined = cv.addWeighted(img, 1-0.5, image_overlay, alpha, 0)
		return image_combined.astype("uint8"), np.sum(mask)

	def creaImage(self, image, masks, alpha=0.5):
		image_combined = np.expand_dims(image[0,:,:,:], 0).repeat(3, axis=0).transpose([1,2,0,3])

		excel=[]
		for nbMask in range (1, np.max(masks)+1):
			slicesLabels=[]
			volumePixel=0
			color = np.array([255,0,0])
			colorName="rouge"
			if nbMask%3==0:
				color = np.array([0,255,0])
				colorName="vert"
			elif nbMask%2==0:
				color = np.array([0,0,255])
				colorName="bleu"
			mask = np.expand_dims(np.where(masks==nbMask, masks, 0)[0,:,:,:], 0).repeat(3, axis=0).transpose([1,2,0,3])
			for slice_ in range (0,image_combined.shape[3]):
				image_combined[:,:,:,slice_], labelOrNot = self.create_masked_image(image_combined[:,:,:,slice_], mask[:,:,:,slice_], alpha, color)
				if labelOrNot!=0:
					slicesLabels.append(slice_)
					volumePixel+=labelOrNot
			#self.labelButtun(nbMask, color, slicesLabels, volumePixel, MaskButton, MaskButtonTxt)
			excel.append([nbMask, volumePixel*0.5*0.5, np.min(slicesLabels), np.max(slicesLabels),colorName])
		return ((image_combined/np.max(image_combined))*255), excel

	def getLabelOfIRM(self, image):
		image=torch.tensor(image)[None]
		image=np.array((image/torch.max(image))*255)
		image=torch.tensor(np.flip(image.copy(),1).copy())
		transform = self.transformsImages()
		dicoImage = self.applyTransforms(transform, image)
		model = self.loadModel()
		dicoImage = self.applyUNETR(dicoImage, model)
		#rt_struct=rt_utils.RTStructBuilder.create_from("C:/Users/ralph/Documents/Model__UNETR__/utilisationMETIA/patient1/IRM", "C:/Users/ralph/Documents/Model__UNETR__/utilisationMETIA/patient1/RTSTRUCT/RS1.2.752.243.1.1.20230320160739553.2000.66260.dcm")
		#masks=[]
		#for i in rt_struct.get_roi_names():
		#	if "GTV" in i or "FRONTAL" in i:
		#		masks.append(np.rot90(np.flip((rt_struct.get_roi_mask_by_name(i).astype(np.float32)>0).tolist(), 0),3))
		label= self.disapplyTransforms(transform, dicoImage)
		label = scipy.ndimage.label(label)[0]
		image, excel = self.creaImage(np.array(image), label)
		return image, label, excel

	def getValues(self):
		return self.image, self.masks, self.excel
		
