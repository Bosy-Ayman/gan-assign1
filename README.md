# Medical MNIST – Autoencoder & Variational Autoencoder (AE & VAE)

## 📌 Overview

This project implements **Autoencoders (AE)** and **Variational Autoencoders (VAE)** for representation learning using the **Medical MNIST dataset**. The goal is to learn compressed representations of medical images, reconstruct them, and explore generative capabilities.

---

##  Dataset

* Dataset: Medical MNIST
* Source: Kaggle
* Images resized to **64x64 RGB**
* Loaded using TensorFlow `image_dataset_from_directory`

---

## ⚙️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* Scikit-learn

---

## 🔧 Implementation Details

### 1. Autoencoder (AE)

* Encoder:

  * Conv2D (32 filters)
  * Conv2D (64 filters)
  * Dense (128 latent vector)

* Decoder:

  * Dense + Reshape
  * Conv2DTranspose layers
  * Output layer with sigmoid activation

* Loss: **Mean Squared Error (MSE)**

---

### 2. Variational Autoencoder (VAE)

* Learns a **probabilistic latent space**

* Encoder outputs:

  * Mean (z_mean)
  * Log variance (z_log_var)

* Uses **reparameterization trick**:
  z = z_mean + exp(0.5 * z_log_var) * epsilon

* Loss:

  * Reconstruction loss (MSE)
  * KL divergence

---

## 📊 Features Implemented

###  Image Reconstruction

* AE reconstructs images with high accuracy
* VAE reconstructs with slight blur due to regularization

###  Latent Space Visualization

* PCA used to reduce latent vectors to 2D
* Shows clustering of similar images

###  Image Generation (VAE)

* Random latent vectors sampled
* New images generated from decoder

### Denoising Autoencoder

* Noise added to images
* Model trained to reconstruct clean images

---

## 📈 Results Summary

| Model | Strength              | Weakness        |
| ----- | --------------------- | --------------- |
| AE    | Sharp reconstruction  | No generation   |
| VAE   | Generates new samples | Slightly blurry |

---

## ▶️ How to Run

### 1. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Install dependencies

```bash
pip install tensorflow matplotlib numpy scikit-learn kagglehub
```

### 3. Run training

Execute all cells in the notebook:

* Train AE
* Train VAE
* Visualize results

---

## 🎯 Key Learnings

* Difference between deterministic and probabilistic models
* Importance of latent space structure
* Trade-off between reconstruction quality and generalization

---

## 📌 Conclusion

* **AE** is better for reconstruction tasks
* **VAE** is better for generative modeling
* Latent space visualization provides insight into learned representations

---

---

## 📎 Notes

* Dataset stored in Google Drive for persistence
* Training performed using Google Colab
* GPU recommended for faster training
