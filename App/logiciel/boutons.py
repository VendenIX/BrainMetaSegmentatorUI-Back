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
import pandas as pd
from rt_utils import RTStructBuilder
import pydicom

class Bouton():
	"""docstring for Bouton"""
	def __init__(self):
		self.masks=[]
		self.excel=[]
		self.path_dir = ""

	def selectIRM(self, SelectDirButtonTxt, fonctionClass):
	    SelectDirButtonTxt.set("calculs en cours ...")
	    self.path_dir = filedialog.askdirectory(title="Choisissez un dossier")
	    self.excel, self.masks = fonctionClass.foundMetastases(self.path_dir, SelectDirButtonTxt)
	    self.masks=self.masks
	    #self.masks=np.rot90(self.masks,3)
	    SelectDirButtonTxt.set("Selection du dossier patient")
	 
	def parcoursValue(self,valeur, listeValeurInterdite):
		print(str(valeur not in listeValeurInterdite), valeur, listeValeurInterdite)
		while str(valeur not in listeValeurInterdite)!=str(True):
			print(valeur)
			valeur+=1
		listeValeurInterdite.append(valeur)
		return valeur, listeValeurInterdite
	def changeValueDicom(self,ds, un, deux, trois, quatre):
		listeValeurInterdite=[]
		valeur=0
		for i in range (0,len(ds[un ,deux].value)):
			valeur+=1
			try:
				print(ds[un, deux].value[i][un, quatre].value[:9] !="GTV_MetIA", ds[un, deux].value[i][un, quatre].value[:9])
				if ds[un, deux].value[i][un, quatre].value[:9] !="GTV_MetIA":
					listeValeurInterdite.append(ds[un, deux].value[i][un, trois].value)
				else:
					valeur, listeValeurInterdite=self.parcoursValue(valeur, listeValeurInterdite)
					ds[un, deux].value[i][un, trois].value=valeur
			except:
				valeur, listeValeurInterdite=self.parcoursValue(valeur, listeValeurInterdite)
				ds[un, deux].value[i][un, trois].value=valeur
			print(ds[un, deux].value[i])
		return ds

	def saveRTSTRUCT_Excel(self, SaveButtonTxt):
		SaveButtonTxt.set("sauvegarde en cours...")
		path_dir = filedialog.askdirectory(title="Choisissez un dossier avec ou sans RTSTRUCT")
		if glob.glob(path_dir+"/RS*")!=[]:
			rtstruct = RTStructBuilder.create_from(dicom_series_path=self.path_dir, rt_struct_path=glob.glob(path_dir+"/RS*")[0])
			for i in range (1, np.max(self.masks)+1):
				rtstruct.add_roi(mask=np.rot90(np.where(self.masks==i, True, False)[0,:,:,:],3),color=[255, 0, 0], name="GTV_MetIA_"+str(i))
			rtstruct.save(path_dir +"/RS_updated.dcm")
			ds = pydicom.dcmread(glob.glob(path_dir +"/RS_updated.dcm")[0])
			#ds = self.changeValueDicom(ds, 0x3006, 0x0080, 0x0084, 0x0085)
			ds = self.changeValueDicom(ds, 0x3006, 0x0020, 0x0022, 0x0026)
			pydicom.filewriter.dcmwrite(path_dir +"/RS_updated.dcm",ds,write_like_original=True)
			
		else:
			rtstruct = RTStructBuilder.create_new(dicom_series_path=self.path_dir)
			for i in range (1, np.max(self.masks)+1):
				rtstruct.add_roi(mask=np.rot90(np.where(self.masks==i, True, False)[0,:,:,:],3),color=[255, 0, 0], name="GTV_MetIA_"+str(i))
			rtstruct.save(path_dir +"/RS_MetIA.dcm")
		nomMeta, volume, sliceD, sliceF, couleur=[],[],[],[],[]
		if self.excel!=[]:
			for meta in self.excel:
				nomMeta.append("GTV_"+str(meta[0]))
				volume.append(str(meta[1]))
				sliceD.append(str(meta[2]))
				sliceF.append(str(meta[3]))
				couleur.append(str(meta[4]))
		df=pd.DataFrame(list(zip(nomMeta,volume,sliceD, sliceF,couleur)), columns=["Métastase", "volume (mm3)", "slice debut", "slice fin", "couleur"])
		df.to_excel(path_dir +'/metastases.xlsx')
		SaveButtonTxt.set("Sauvegarde effectuée")
	
	def getValues(self):
		return self.masks, self.excel
