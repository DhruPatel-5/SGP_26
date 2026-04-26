# 🧠 Deepfake Detection System

A state-of-the-art deepfake detection system built with PyTorch and EfficientNet-B0, featuring a user-friendly web interface for real-time image and video analysis.

## ⚙️ Created By

- 👨‍💻 [T RAHUL SINGH](https://github.com/TRahulsingh)
- 🧑‍💻 [Mallikarjun Macherla](https://github.com/Mallikarjun-Macherla)
- 🧑‍💻 [Sainath](https://github.com/sainathch45/)

---

## 🌟 Features

- **Deep Learning Model**: EfficientNet-B0 architecture fine-tuned for deepfake detection
- **Multi-format Support**: Analyze both images (.jpg, .jpeg, .png) and videos (.mp4, .mov)
- **Web Interface**: Interactive Gradio-based web application for easy testing
- **Real-time Analysis**: Process first frame of videos for quick deepfake detection
- **Training Pipeline**: Complete PyTorch Lightning training infrastructure
- **Model Export**: Support for PyTorch (.pt) and ONNX format exports

## 📐 System Architecture

For detailed system architecture diagrams, data flow, and component interactions, see [ARCHITECTURE.md](ARCHITECTURE.md).

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, but recommended for training)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/TRahulsingh/DeepfakeDetector.git
   cd DeepfakeDetector
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download a pre-trained model** (or train your own):
   - Place your model file as `models/best_model-v3.pt`

### Usage

#### 🖥️ Web Application

Launch the interactive web interface:

```bash
python web-app.py
```

The web app will open in your browser where you can:
- Drag and drop images or videos
- View real-time predictions with confidence scores
- See preview of analyzed content

#### 🔍 Command Line Classification

Classify individual images:

```bash
python classify.py path/to/your/image.jpg
```

#### 🎥 Video Analysis

Process videos from a folder:

```bash
# Place videos in 'videos_to_predict' folder, then run:
python inference/video_inference.py
```

## 📂 Supported Datasets

This deepfake detection system supports various popular deepfake datasets. Below are the recommended datasets for training and evaluation:

### 🎬 Video-based Datasets

#### **FaceForensics++**
- **Description**: One of the most comprehensive deepfake datasets with 4 manipulation methods
- **Size**: ~1,000 original videos, ~4,000 manipulated videos
- **Manipulations**: Deepfakes, Face2Face, FaceSwap, NeuralTextures
- **Quality**: Raw, c23 (light compression), c40 (heavy compression)
- **Download**: [GitHub Repository](https://github.com/ondyari/FaceForensics)
- **Usage**: Excellent for training robust models across different manipulation types

#### **Celeb-DF (v2)**
- **Description**: High-quality celebrity deepfake dataset
- **Size**: 590 real videos, 5,639 deepfake videos
- **Quality**: High-resolution with improved visual quality
- **Download**: [Official Website](https://github.com/yuezunli/celeb-deepfakeforensics)
- **Usage**: Great for testing model performance on high-quality deepfakes

#### **DFDC (Deepfake Detection Challenge)**
- **Description**: Facebook's large-scale deepfake detection dataset
- **Size**: ~100,000 videos (real and fake)
- **Diversity**: Multiple actors, ethnicities, and ages
- **Download**: [Kaggle Competition](https://www.kaggle.com/c/deepfake-detection-challenge)
- **Usage**: Large-scale training and benchmarking

#### **DFD (Google's Deepfake Detection Dataset)**
- **Description**: Google/Jigsaw deepfake dataset
- **Size**: ~3,000 deepfake videos
- **Quality**: High-quality with various compression levels
- **Download**: [FaceForensics++ repository](https://github.com/ondyari/FaceForensics)
- **Usage**: Additional training data for model robustness

### 🖼️ Image-based Datasets

#### **140k Real and Fake Faces**
- **Description**: Large collection of real and AI-generated face images
- **Size**: ~140,000 images
- **Source**: StyleGAN-generated faces vs real faces
- **Download**: [Kaggle Dataset](https://www.kaggle.com/xhlulu/140k-real-and-fake-faces)
- **Usage**: Perfect for image-based deepfake detection training

#### **CelebA-HQ**
- **Description**: High-quality celebrity face dataset
- **Size**: 30,000 high-resolution images
- **Quality**: 1024×1024 resolution
- **Download**: [GitHub Repository](https://github.com/tkarras/progressive_growing_of_gans)
- **Usage**: Real face examples for training

### 🔧 Dataset Preparation

#### Option 1: Download Pre-processed Datasets
1. Download your chosen dataset from the links above
2. Extract to the `data/` folder
3. Organize as shown in the training section below

#### Option 2: Use Dataset Preparation Tools
Use our built-in tools to prepare datasets. Edit the source/destination paths inside each script before running:

```bash
# Extract frames from videos (every 15th frame) and split into train/val
# Edit source & dest paths in the script, then run:
python tools/split_video_dataset.py

# Split an existing image dataset into 80/20 train/validation
# Edit source_dataset & destination paths in the script, then run:
python tools/split_train_val.py

# Extract frames from a single video directory
# Edit video_dir & output_dir in the script, then run:
python tools/split_dataset.py
```

### 📋 Dataset Recommendations

- **For Beginners**: Start with **140k Real and Fake Faces** (image-based, easy to work with)
- **For Research**: Use **FaceForensics++** (comprehensive, multiple manipulation types)
- **For Production**: Combine **DFDC** + **Celeb-DF** (large scale, diverse)
- **For High-Quality Testing**: Use **Celeb-DF v2** (challenging, high-quality deepfakes)

### ⚠️ Dataset Usage Notes

- **Ethical Use**: These datasets are for research purposes only
- **Legal Compliance**: Ensure compliance with dataset licenses and terms of use
- **Privacy**: Respect privacy rights of individuals in the datasets
- **Citation**: Properly cite the original dataset papers when publishing research

## 🏋️ Training

### Dataset Structure

Organize your training data in the `data` folder as follows:
```
data/
├── train/
│   ├── real/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── fake/
│       ├── fake1.jpg
│       └── fake2.jpg
└── validation/
    ├── real/
    └── fake/
```

### Configuration

Update `config.yaml` with your dataset paths:

```yaml
train_paths:
  - data/train

val_paths:
  - data/validation

lr: 0.0001
batch_size: 4
num_epochs: 10
```

### Start Training

```bash
python main_trainer.py
```

The training will:
- Use PyTorch Lightning for efficient training
- Save best model based on validation loss
- Log metrics to TensorBoard
- Apply early stopping to prevent overfitting

### Monitor Training

View training progress with TensorBoard:

```bash
tensorboard --logdir lightning_logs
```

## 📁 Project Structure

```
├── web-app.py                    # Main web application
├── main_trainer.py               # Primary training script
├── classify.py                   # Image classification utility
├── realeval.py                   # Real-world evaluation script
├── config.yaml                   # Training configuration
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── ARCHITECTURE.md               # System architecture & design
├── LICENSE                       # MIT License
├── .gitignore                    # Git ignore rules
├── data/                         # Dataset storage (not tracked by git)
│   ├── train/                    # Training data
│   └── validation/               # Validation data
├── datasets/
│   └── hybrid_loader.py          # Custom dataset loader
├── lightning_modules/
│   └── detector.py               # PyTorch Lightning module
├── models/
│   └── best_model-v3.pt          # Trained model weights
├── tools/                        # Dataset preparation utilities
│   ├── export_to_pt.py           # .ckpt → .pt model converter
│   ├── split_dataset.py          # Video frame extractor
│   ├── split_train_val.py        # 80/20 train/val splitter
│   └── split_video_dataset.py    # Video-aware dataset splitter
└── inference/
    ├── export_onnx.py            # ONNX export
    └── video_inference.py        # Multi-frame video inference
```

## 🛠️ Model Architecture

- **Backbone**: EfficientNet-B0 (pre-trained on ImageNet)
- **Classifier**: Custom 2-class classifier with dropout (0.4)
- **Input Size**: 224x224 RGB images
- **Output**: Binary classification (Real/Fake) with confidence scores

## 📊 Performance

- **Inference Speed**: Real-time on GPU, ~200ms per image on CPU
- **Input Support**: Images (.jpg, .png) and videos (.mp4, .mov)
- **Video Analysis**: 10-frame uniform sampling with probability averaging
- **Robustness**: Tested with Gaussian blur and JPEG compression noise simulation (`realeval.py`)

> **Note**: Accuracy metrics depend on your training dataset. Monitor `val_loss` and `val_acc` via TensorBoard during training.

## 🔧 Advanced Usage

### Export to ONNX

Convert PyTorch model to ONNX format:

```bash
python inference/export_onnx.py
```

### Batch Evaluation

Evaluate a folder of real-world samples with optional noise simulation:

```bash
# Place test images/videos in realworld_samples/ folder, then run:
python realeval.py
```

### Export Checkpoint to PyTorch

Convert a Lightning `.ckpt` to a standalone `.pt` file:

```bash
# Edit ckpt_path and pt_output in the script, then run:
python tools/export_to_pt.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- EfficientNet architecture by Google Research
- PyTorch Lightning for training infrastructure
- Gradio for web interface framework
- The research community for deepfake detection advances

---

## 📄 License

This project is licensed under the **MIT License**.

---

⭐ **Star this repository if you found it helpful!**
