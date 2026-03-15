# FaceForge: Deepfake Detection Lab 🛡️

A high-performance deep learning solution for detecting AI-generated facial manipulations. Built for forensic analysis with explainable AI (XAI) features.

### 🚀 Key Features
* **Core Model:** EfficientNet-B4 CNN architecture (Achieved 99.2% Accuracy).
* **Explainable AI:** Integrated **Grad-CAM** heatmaps to visualize which facial regions (eyes, skin, mouth) the AI is flagging as "Fake."
* **Forensic Tools:** Frequency domain analysis (FFT) to identify GAN-generated artifacts.
* **Full Stack:** Fast-API backend for high-speed inference and a modern React frontend.

### 📁 Project Structure
* `deepfake-backend/`: PyTorch model logic, Grad-CAM generation, and API.
* `deepfake-frontend/`: Interactive React dashboard for uploading and analyzing media.

### 🛠️ Setup
1. Clone the repo.
2. Install backend requirements: `pip install -r deepfake-backend/requirements.txt`
3. Download the model weights (`best_model.pth`) from [INSERT YOUR GOOGLE DRIVE LINK HERE] and place in `deepfake-backend/app/checkpoints/`.
