Knee Bone Segmentation (Femur) using U-Net
--A Machine Vision and CNN Course Project

Overview

    This project implements a complete pipeline for bone structure segmentation (femur) from X-ray images using a U-Net based convolutional neural network.
    The pipeline includes dataset creation, preprocessing, training with BCE + Dice loss, and final evaluation on a 50-image test set.

The project was implemented following the mandatory modular structure:

    knee_project/
    │
    ├── args.py
    ├── dataset.py
    ├── model.py
    ├── trainer.py
    ├── utils.py
    ├── evaluate.py
    ├── main.py
    │
    └── data/
        ├── xrays/
        ├── masks/
        └── CSVs/


Dataset

    150 X-ray images
	•	100 manually annotated (training set)
	•	50 test images without masks
	
    CSV files were automatically generated using create_csvs.py:
	•	dataset.csv
	•	train.csv (80 images)
	•	val.csv (20 images)
	•	test.csv (50 unlabeled images)

Model

The neural network is a configurable U-Net architecture (UNetLext) featuring:

	•	Multi-level encoder/decoder
	•	Skip connections
	•	Flexible convolution blocks
	•	Final 1×1 convolution for binary mask output

Loss function:

	•	BCEWithLogitsLoss
	•	Dice Loss

Total loss = BCE + Dice

Training:
    
    Run training: python3 main.py
    The best model (highest validation Dice) is automatically saved as: session/best_model.pth

Evaluation:

    The evaluation script loads the best model and visualizes predictions on the 50-image test set: python3 evaluate.py
    Example visualization:
	•	Left: Original X-ray
	•	Right: Predicted mask

Installation:
    
    Create a virtual environment:
        python3 -m venv venv
        source venv/bin/activate
    Install dependencies:
        pip3 install -r requirements.txt