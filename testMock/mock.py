import json

tab = [["GTV_0", 1515.75, 120, 129, "vert"],
       ["GTV_1", 239.25, 123, 128, "rouge"],
       ["GTV_2", 157.5, 62, 66, "bleu"],
       ["GTV_3", 225.0, 112, 116, "vert"],
       ["GTV_4", 68.25, 153, 155, "bleu"],
       ["GTV_5", 560.25, 143, 151, "rouge"],
       ["GTV_6", 501.75, 73, 80, "vert"]
       ]

jsonMock = json.dumps(tab, indent=2)
jsonPath = "./json/mock.json"

with open(jsonPath, "w") as json_file :
    json_file.write(jsonMock)