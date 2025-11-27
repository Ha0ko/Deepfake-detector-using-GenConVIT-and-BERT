#  Deepfake Detector (GenConViT + BERT)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A hybrid deep learning framework for Deepfake detection utilizing **GenConViT** (Visual Transformer) for visual artifacts and **BERT** for textual/audio-transcript consistency. This project detects manipulated media by analyzing both spatial-temporal features and semantic inconsistencies.

---

##  Features
- **Multi-Modal Detection:** Combines visual analysis (GenConViT) with text analysis (BERT).
- **VAE Support:** Includes Variational Autoencoder components for feature extraction.
- **Web Interface:** Includes `app.py` for a user-friendly GUI.
- **API Server:** Includes `server.py` for backend deployment.
- **Docker Support:** Ready for containerized deployment.

---

## ⚠️ Important: Download Model Weights
Due to GitHub file size limits, the pre-trained model weights are hosted externally. You **must** download them for the detector to work.

1. **Download the weights** from **(https://drive.google.com/drive/folders/16MchwqPLGpUqGgleFnkYv3a70aQX9_K8?usp=sharing)**.
2. Create a folder named `weight` in the root directory.
3. Place the following files inside the `weight/` folder:
   - `genconvit_ed_inference.pth`
   - `genconvit_vae_inference.pth`

*(If you have a sample dataset, place it in the `sample_prediction_data/` folder.)*

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Ha0ko/Deepfake-detector-using-GenConVIT-and-BERT.git
cd Deepfake-detector-using-GenConVIT-and-BERT
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

##  Usage

### **Run the Web Interface**
```bash
streamlit run app.py
```

---

## 📂 Project Structure
```
├── dataset/                # Data loaders and processing scripts
├── model/                  # GenConViT, VAE, and Transformer architectures
├── weight/                 # PLACE DOWNLOADED WEIGHTS HERE (.pth files)
├── sample_prediction_data/ # Sample videos for testing
├── app.py                  # Frontend application
├── server.py               # Backend API server
├── predict.py              # CLI inference script
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container configuration
└── README.md               # Documentation
```

---

## 📄 License
This project is licensed under the **MIT License** – see the LICENSE file for details.

---

##  Credits
Based on the architecture of **GenConViT (Generative ConvViT)** and **BERT transformers**.

