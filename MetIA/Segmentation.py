from monai.inferers import sliding_window_inference
import torch


"""
    Effectue une inférence de segmentation sur les images contenues dans le dictionnaire d'images en utilisant un modèle UNETR.

    Args:
        dicoImage (dict): Dictionnaire contenant les images sur lesquelles effectuer l'inférence.
                          Doit contenir au moins une clé "image" correspondant aux images à segmenter.
        model (torch.nn.Module): Modèle UNETR à utiliser pour l'inférence de segmentation.

    Returns:
        dict: Dictionnaire mis à jour avec les prédictions de segmentation ajoutées sous la clé "label".
              Les prédictions sont remaniées pour correspondre à la taille des images d'origine.
"""

class Segmentation:
    def __init__(self, model):
        self.model = model

    def apply_unetr(self, image_dict):
        label = sliding_window_inference(inputs=image_dict["image"][None],
                                          roi_size=(96, 96, 96),
                                          sw_batch_size=4,
                                          predictor=self.model,
                                          overlap=0.5)
        label = torch.argmax(label, dim=1, keepdim=True)
        size = label.shape
        image_dict["label"] = label.reshape((size[1], size[2], size[3], size[4]))
        return image_dict