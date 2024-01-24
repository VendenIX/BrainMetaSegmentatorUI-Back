import pydicom
import json
from glob import glob
import matplotlib.pylab as plt

pathRTS = "./data/1-1.dcm"
dataRTS = pydicom.dcmread(pathRTS)

# Exclure les éléments non sérialisables en JSON du Dataset
dicoRTS = {}
for elem in dataRTS:
    try:
        # Tentative de sérialisation en JSON, exclut les éléments non sérialisables
        json.dumps({elem.tag: elem.value})
        dicoRTS[elem.tag] = elem.value
    except TypeError:
        pass

# Ouverture et écriture du .dcm en .json avec les données sérialisables
jsonRTS = json.dumps(dicoRTS, indent=2)
jsonPath = "./json/resultat.json"

with open(jsonPath, "w") as json_file :
    json_file.write(jsonRTS)

# Ouverture et écriture du .dcm entier
ds = pydicom.filereader.dcmread('./data/1-1.dcm')

with open("test.txt", "w") as text_file :
    text_file.write(str(ds))

print("Fini")
