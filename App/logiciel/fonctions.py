import tkinter as tk
from tkinter import filedialog
from tkinter import font
from PIL import ImageTk, Image
import pyautogui
import os
import glob
import monai
from monai.transforms import LoadImage
import torch
import numpy as np
from logiciel.images import Images

class Fonctions():
	"""docstring for ClassName"""
	def __init__(self, slice_, root, fontStyle, blueDark, white):
		self.slice_ = slice_
		self.root = root
		self.fontStyle=fontStyle
		self.blueDark=blueDark
		self.white=white
		self.image3D=torch.empty(512,512,208)
		self.IRM3DMiddle = tk.Label(self.root)
		self.NumberSliceTxt = tk.StringVar()
		
	def showImageSlice(self):
		image3D = Image.fromarray(np.array(self.image3D[:,:,:,0]).astype(np.uint8), "RGB").rotate(270)
		IRM3D = ImageTk.PhotoImage(image=image3D.resize((int(image3D.width*1.5), int(image3D.height*1.5))))
		self.IRM3DMiddle = tk.Label(self.root, image = IRM3D)
		self.IRM3DMiddle.image = IRM3D
		self.IRM3DMiddle.grid(columns=3, row=3, sticky="NS")
		    
	def replaceImageSlice(self, side):
		if int(self.NumberSliceTxt.get())!=self.slice_ and int(self.NumberSliceTxt.get())<self.image3D.shape[3] and int(self.NumberSliceTxt.get())>=0:
			self.slice_ = int(self.NumberSliceTxt.get())
		else:
			if side=="R" and self.slice_<self.image3D.shape[3]-1:
				self.slice_ +=1
			if side=="L" and self.slice_>0:
				self.slice_ -=1

		self.NumberSliceTxt.set(self.slice_)
		image3D = Image.fromarray(np.array(self.image3D[:,:,:,self.slice_]).astype(np.uint8), "RGB").rotate(270)
		IRM3D = ImageTk.PhotoImage(image=image3D.resize((int(image3D.width*1.5), int(image3D.height*1.5))))
		self.IRM3DMiddle.configure(image=IRM3D)
		self.IRM3DMiddle.image = IRM3D

	def createDir(self, path_dir, name_dir, name):
	    path = os.path.join(path_dir, name_dir)
	    if not os.path.exists(path):
	        os.mkdir(path)
	    for mr in glob.glob(os.path.join(path_dir, name)):
	        path_file, name_file = os.path.split(mr)
	        os.rename(mr, str(path)+str(name_file))
	    return path

	def foundMetastases(self,path_dir, SelectDirButtonTxt):
		path_MR = self.createDir(path_dir,"IRM/", "MR*")
		#path_RS = self.createDir(path_dir,"RTSTRUCT/", "RS*")
		self.image3D=torch.tensor(LoadImage(image_only=True)(path_MR))
		self.image3D = ((self.image3D/np.max(np.array(self.image3D)))*255)

		#ici appliquer UNETR
		self.image3D, masks, excel= Images(self.image3D).getValues()
		self.showImageSlice()
		
		widthScreen, heightScreen= pyautogui.size ()
	
		NumberSlice = tk.Entry(self.root, textvariable=self.NumberSliceTxt, font=font.Font(family="Raleway", size=15))
		self.NumberSliceTxt.set(self.slice_)
		NumberSlice.grid(columns=4, row=3, sticky="S")

		RigthButtonTxt = tk.StringVar()
		RigthButton = tk.Button(self.root, textvariable= RigthButtonTxt, command=lambda:self.replaceImageSlice("R"), font=self.fontStyle, bg=self.blueDark, fg=self.white, height=2, width=4)
		RigthButtonTxt.set(">")
		RigthButton.grid(columns=2, row=3, sticky="SE", padx=widthScreen/20)
		
		LeftButtonTxt = tk.StringVar()
		LeftButton = tk.Button(self.root, textvariable= LeftButtonTxt, command=lambda:self.replaceImageSlice("L"), font=self.fontStyle, bg=self.blueDark, fg=self.white, height=2, width=4)
		LeftButtonTxt.set("<")
		LeftButton.grid(columns=5, row=3, sticky="SW", padx=widthScreen/4)
		return excel, masks

		
