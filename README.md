🩺 Lesion-Based Diabetic Retinopathy Detection Using Hybrid Deep Learning
📌 Project Overview

A Lesion-Based Diabetic Retinopathy Detection Through Hybrid Deep Learning Model is a deep learning-based application designed to analyze retinal fundus images and identify diabetic retinopathy-related lesions.

The system processes retinal images and classifies them into different categories, helping support automated retinal image analysis and diabetic retinopathy screening.

🎯 Objectives
Detect diabetic retinopathy-related abnormalities from retinal images.
Identify important retinal lesion categories.
Apply deep learning techniques for automated image classification.
Provide a user-friendly interface for testing retinal images.
Reduce the need for completely manual retinal image screening.
✨ Key Features
🧠 Hybrid deep learning-based retinal image analysis
👁️ Lesion-based classification
🖼️ Retinal fundus image processing
🔍 Automated disease prediction
💻 User-friendly application interface
📊 Multiple retinal condition categories
⚡ Automated prediction from test images
🧪 Disease / Lesion Classes

The project works with the following categories:

Class	Description
Normal	Retinal image without the targeted abnormality
DR	Diabetic Retinopathy
MH	Macular Hole
ODC	Optic Disc Cupping
🛠️ Technologies Used
Python
Deep Learning
TensorFlow / Keras
NumPy
OpenCV
Machine Learning
Image Processing
Tkinter / GUI
📁 Project Structure
DR_Project/
│
├── Dataset/
│   ├── RFMiD_Training_Labels.csv
│   └── images/
│
├── SelectedImages/
│   ├── Normal/
│   ├── DR/
│   ├── MH/
│   └── ODC/
│
├── model/
│   ├── *.hdf5
│   ├── *.npy
│   └── *.pckl
│
├── Main.py
├── test.py
├── test1.py
├── run.bat
├── requirements.txt
├── .gitignore
└── README.md

Note: Large dataset images, generated model files, and processed image folders are excluded from GitHub using .gitignore to keep the repository lightweight.

⚙️ Installation
1. Clone the Repository
git clone https://github.com/kondalwadijeevan/Diabetic-Retinopathy-detection.git
2. Navigate to the Project Directory
cd Diabetic-Retinopathy-detection
3. Create a Virtual Environment
python -m venv venv
4. Activate the Virtual Environment

For Windows:

venv\Scripts\activate
5. Install Dependencies
pip install -r requirements.txt
📂 Dataset Setup

The project uses retinal fundus images and corresponding labels.

Place the required dataset files in the following structure:

Dataset/
│
├── images/
│   └── retinal images
│
└── RFMiD_Training_Labels.csv

The dataset images should be placed inside:

Dataset/images/

The training labels should be placed at:

Dataset/RFMiD_Training_Labels.csv
▶️ How to Run
Step 1 — Prepare the Images

Run:

python test1.py

This prepares/selects the required retinal images for the project.

Step 2 — Run the Application

Run:

python Main.py

Alternatively, on Windows you can use:

run.bat

The application can then be used to process retinal images and perform the prediction.

🔬 Methodology

The overall workflow of the project can be represented as:

Retinal Fundus Image
        ↓
Image Preprocessing
        ↓
Lesion / Feature Analysis
        ↓
Hybrid Deep Learning Model
        ↓
Feature Learning
        ↓
Classification
        ↓
Predicted Retinal Condition

The system uses deep learning to learn relevant visual patterns from retinal images and classify them into the supported categories.

📊 Results

The project provides automated classification of retinal images into the supported categories.

The prediction output is generated through the trained deep learning model and can be used as an automated aid for retinal image analysis.

Add your actual accuracy, precision, recall, F1-score, confusion matrix, or other evaluation metrics here if they are available from your experiments.
