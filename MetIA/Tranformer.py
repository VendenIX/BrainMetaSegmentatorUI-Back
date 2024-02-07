import torch



class Transformer:
    def __init__(self, transform):
        self.transform = transform


    """
        Applique une série de transformations à une image.

        Args:
            transform (torchvision.transforms.Compose): Composition de transformations à appliquer.
            image (np.ndarray): Image sur laquelle appliquer les transformations.

        Returns:
            dict: Dictionnaire contenant l'image transformée et d'autres métadonnées.
    """

    def apply_transforms(self, image):
        image = {"image": image, "label": torch.zeros_like(image), "patient_id": '201905984', "has_meta": True}
        image = self.transform(image)
        return image
    
    """
        Annule les transformations appliquées à une image.

        Args:
            transform (torchvision.transforms.Compose): Composition de transformations inverses.
            dicoImage (dict): Dictionnaire contenant l'image transformée et ses métadonnées.

        Returns:
            tuple: Tuple contenant l'étiquette désappliquée et l'image désappliquée.
    """

    def disapply_transforms(self, image_dict):
        image_dict = self.transform.inverse(image_dict)
        return image_dict["label"], image_dict["image"]