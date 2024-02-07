import argparse
from executionSansTkinter.Back.MetIA.ImageLabel import ImageLabeler

from executionSansTkinter.Back.MetIA.Load import DicomImageLoader, ModelLoader
from executionSansTkinter.Back.MetIA.Transforms import TransformationFactory

# from executionSansTkinter.Back.MetIA import ModelLoader, DicomImageLoader, TransformationFactory, ImageLabeler

def main():
    parser = argparse.ArgumentParser(description='Process DICOM images and perform segmentation.')
    parser.add_argument('--slices-folder', type=str, help='Path to the folder containing DICOM slices.')
    parser.add_argument('--model-file', type=str, help='Path to the segmentation model file.')
    args = parser.parse_args()

    # Load DICOM images
    dicom_loader = DicomImageLoader(args.slices_folder)
    image = dicom_loader.load_dicom_image()

    # Load segmentation model
    model_loader = ModelLoader(args.model_file)
    model = model_loader.load_model()

    # Create transformation
    transform = TransformationFactory.create_transform()

    # Apply segmentation
    image_labeler = ImageLabeler(args.slices_folder, args.model_file, transform)
    _, label, _ = image_labeler.get_label_of_irm()

    # Process segmentation results as needed
    print("Segmentation completed.")

if __name__ == "__main__":
    main()
