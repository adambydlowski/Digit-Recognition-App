#  Handwritten Digit Recognition App

Author: Adam Bydłowski
---

An interactive desktop application that allows you to draw a handwritten digit (0–9) and receive real-time predictions from a trained **Convolutional Neural Network (CNN)**.

This project combines modern deep learning with a user-friendly GUI to demonstrate digit classification using the MNIST datasets.

---

##  Features

- **Draw digits** directly in the application window
-  **CNN-based classifier** built with PyTorch
-  Displays **prediction probabilities** for all 10 digits
-  Load and use a **pre-trained model** (`model.pth`)
-  **Polished GUI** using `Tkinter` and `ttkbootstrap`

---

##  Technologies Used

- **Python 3.11.9**
- [PyTorch] — neural network training & inference
- [ttkbootstrap] — modern GUI styling
- `Tkinter` — GUI framework
- `Pillow` — image preprocessing
- `UMAP`, `DBSCAN`, `matplotlib`, `sklearn`, `numpy` — tools for clustering & data analysis

---

##  Preview

![App Demo](media/App-usage.gif)

---

##  Model Overview

###  Architecture:

- 3 convolutional layers with batch normalization and ReLU activation

- Max pooling after the first two convolutional layers

- Fully connected classifier with dropout and softmax output

####  Training techniques:

- Data augmentation: rotation, noise, scaling

- Trained with PyTorch, achieves ~98% accuracy on standard digit datasets

Training of CNN is saved in `model/model_training.ipynb`

---
