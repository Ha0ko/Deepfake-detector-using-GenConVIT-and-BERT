🕵️‍♂️ Deepfake Detector (GenConViT + BERT)
![alt text](https://img.shields.io/badge/Python-3.8%2B-blue)

![alt text](https://img.shields.io/badge/PyTorch-2.0%2B-orange)

![alt text](https://img.shields.io/badge/License-MIT-green)
A hybrid deep learning framework for Deepfake detection utilizing GenConViT (Visual Transformer) for visual artifacts and BERT for textual/audio-transcript consistency. This project detects manipulated media by analyzing both spatial-temporal features and semantic inconsistencies.
⚠️ Important: Download Model Weights
Due to GitHub file size limits, the pre-trained model weights are hosted externally. You must download them for the detector to work.
Download the weights from: [INSERT YOUR DOWNLOAD LINK HERE]
Create a folder named weight in the root directory.
Place the following files inside the weight/ folder:
genconvit_ed_inference.pth
genconvit_vae_inference.pth
(If you have a sample dataset, place it in the sample_prediction_data/ folder).
🛠️ Installation
1. Clone the Repository
code
Bash
git clone https://github.com/Ha0ko/Deepfake-detector-using-GenConVIT-and-BERT.git
cd Deepfake-detector-using-GenConVIT-and-BERT
2. Install Dependencies
Ensure you have Python installed, then run:
code
Bash
pip install -r requirements.txt
💻 Usage
Option 1: Run the Web Interface
To start the graphical interface (Streamlit/Gradio):
code
Bash
python app.py
Option 2: Run via Command Line
To predict a specific video file:
code
Bash
python predict.py --video_path sample_prediction_data/sample_1.mp4
Option 3: Run the Backend Server
To start the API server:
code
Bash
python server.py
Option 4: Docker
Build and run the container:
code
Bash
docker build -t deepfake-detector .
docker run -p 5000:5000 deepfake-detector
📂 Project Structure
code
Text
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
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
🤝 Credits
Based on the architecture of GenConViT (Generative ConvViT) and BERT transformers.
