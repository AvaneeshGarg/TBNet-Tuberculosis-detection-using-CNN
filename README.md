# Tuberculosis Detection Using CNN

This project focuses on building a Convolutional Neural Network (CNN) to detect Tuberculosis (TB) from chest X-ray images. The model is trained on a dataset of chest X-rays labeled as "Normal" or "Tuberculosis." The project also explores image preprocessing, augmentation, and evaluation using various metrics.


## Project Overview
The goal of this project is to develop a robust CNN model for the accurate detection of Tuberculosis from chest X-ray images. The project includes:
- Data preprocessing and augmentation.
- Building and training a CNN model.
- Evaluating the model using metrics like accuracy, precision, recall, and F1-score.
- Comparing the performance of the custom CNN with pre-trained models like VGG19 and ResNet50.

---

## Dataset
The dataset used in this project is the **TB Chest Radiography Database**, which contains chest X-ray images labeled as:
- **Normal**: Images of healthy patients.
- **Tuberculosis**: Images of patients diagnosed with TB.

The dataset is preprocessed and augmented to improve model performance.

---

## Installation
To run this project, you need the following dependencies:

1. **Python 3.x**
2. **Libraries**:
   - TensorFlow
   - NumPy
   - Pandas
   - Matplotlib
   - Seaborn
   - Scikit-learn
   - OpenCV
   - Scikit-image

You can install the required libraries using the following command:

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn opencv-python scikit-image
