from executionSansTkinter.Back.MetIA.Load import DicomImageLoader, ModelLoader
from executionSansTkinter.Back.MetIA.Segmentation import Segmentation
from executionSansTkinter.Back.MetIA.Tranformer import Transformer


class ImageLabeler:
    def __init__(self, path_slices_irm, path_model_file, transform):
        self.path_slices_irm = path_slices_irm
        self.path_model_file = path_model_file
        self.transform = transform


    """
        Charge une image IRM à partir du dossier spécifié, applique une inférence de segmentation à l'aide du modèle spécifié
        et retourne l'image originale, l'étiquette de segmentation et l'image transformée.

        Args:
            path_slices_irm (str): Chemin vers le dossier contenant les images IRM.
            path_model_file (str): Chemin vers le fichier contenant le modèle de segmentation.
            transform (torchvision.transforms.Compose): Composition de transformations à appliquer à l'image IRM.

        Returns:
            tuple: Tuple contenant l'image IRM, l'étiquette de segmentation et l'image transformée.
    """

    def get_label_of_irm(self):
        image = DicomImageLoader(self.path_slices_irm).load_dicom_image()
        image_dict = Transformer(self.transform).apply_transforms(image)
        model = ModelLoader(self.path_model_file).load_model()
        image_dict = Segmentation(model).apply_unetr(image_dict)
        label, image_transformed = Transformer(self.transform).disapply_transforms(image_dict)
        return image / 255, label, image_transformed