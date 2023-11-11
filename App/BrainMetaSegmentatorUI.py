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
from logiciel.boutons import Bouton
from logiciel.fonctions import Fonctions

def main():
	#récuperer la taille de l'écran
	widthScreen, heightScreen= pyautogui.size ()

	#définir une variable pour tkinter
	root = tk.Tk()
	#définir une écriture
	fontStyle = font.Font(family="Raleway", size=20)
	#définir les couleurs
	blueDark='#006988'
	blueLigth='#81E4DF'
	white='#F0FFFE'

	#donner des indications sur la fenêtre
	    #définir l'emplacement
	root.geometry('+%d+%d'%(0,0))
	    #définir l'entête
	header = tk.Frame(root, width=widthScreen, height=heightScreen/100, bg=white)
	header.grid(columnspan=5, rowspan=2, row=0)
	    #définir le corps
	main_content = tk.Frame(root, width=widthScreen, height=heightScreen/1.2, bg=white)
	main_content.grid(columnspan=3, rowspan=2, row=3)

	fonctionClass = Fonctions(0, root, fontStyle, blueDark, white)
	boutonClass = Bouton()

	#créer le bouton de parcours des dossier
	SelectDirButtonTxt = tk.StringVar()
	SelectDirButton = tk.Button(root, textvariable= SelectDirButtonTxt, command=lambda:boutonClass.selectIRM(SelectDirButtonTxt, fonctionClass), font=fontStyle, bg=blueDark, fg=white, height=2, width=30)
	SelectDirButtonTxt.set("Selection du dossier patient")
	SelectDirButton.grid(columns=1, row=2, sticky="W", padx=widthScreen/20)

	#ajout du logo baclesse
	logoBaclesse = ImageTk.PhotoImage(Image.open("logiciel/images/baclesse.png").resize((int(widthScreen/20), int(widthScreen/20))))
	logoBaclesseTitle = tk.Label(root, image = logoBaclesse)
	logoBaclesseTitle.grid(columns=2, row=2, sticky="S", padx=widthScreen/20)

	#ajout du Titre
	logoMetIA = ImageTk.PhotoImage(Image.open("logiciel/images/MetIA.png").resize((int(widthScreen/20)*3, int(widthScreen/20))))
	logoMetIATitle = tk.Label(root, image = logoMetIA)
	logoMetIATitle.grid(columns=3, row=2, sticky="N", padx=widthScreen/20)

	#créer le bouton de sauvegarde
	SaveButtonTxt = tk.StringVar()
	SaveButton = tk.Button(root, textvariable= SaveButtonTxt, command=lambda:boutonClass.saveRTSTRUCT_Excel(SaveButtonTxt), font=fontStyle, bg=blueDark, fg=white, height=2, width=30)
	SaveButtonTxt.set("Sauvegarder le RTSTRUCT et excel")
	SaveButton.grid(columns=5, row=2, sticky="E", padx=widthScreen/20)

	def callback(event):
		fonctionClass.replaceImageSlice("M")

	root.bind('<Return>', callback)

	root.mainloop()

if __name__ == "__main__":
	main()