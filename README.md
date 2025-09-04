```markdown
# 🌊 Multispectral Water Segmentation

**Deep learning-based water body segmentation from 12-band satellite imagery using U-Net.**

This project aims to accurately segment water bodies from multispectral and optical satellite data for applications in environmental monitoring, flood detection, and water resource management. We use a **U-Net architecture** trained on 12-band imagery (128×128 pixels) to perform pixel-level classification into water and non-water classes.

🎯 **Goal**: High-precision, robust water segmentation using deep learning.

---

## 📸 Project Overview

| Feature | Description |
|--------|-------------|
| **Input** | 12-band multispectral & optical satellite images (`.tif`) |
| **Output** | Binary segmentation mask (water = 1, non-water = 0) in `.png` |
| **Resolution** | 128 × 128 pixels |
| **Model** | U-Net with 12-channel input |
| **Framework** | PyTorch |
| **Loss** | Combined BCE + Dice Loss |
| **Metrics** | IoU, Precision, Recall, F1-score (water class) |

![Example Output](results/sample_1.png)

> *Sample: Input (left), Ground Truth (center), Prediction (right)*

---

## 📁 Dataset Structure

The dataset should be organized as:

```

data/
├── images/
│   ├── 0.tif
│   ├── 1.tif
│   └── ...
└── labels/
├── 0.png
├── 1.png
└── ...

````

- Each `.tif` file contains **12 spectral bands**.
- Each `.png` is a **binary mask** (grayscale: 0 = land, 255 = water → converted to 0/1).

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/multispectral-water-segmentation.git
cd multispectral-water-segmentation
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note**: Make sure `rasterio` is installed via:
>
> ```bash
> pip install rasterio opencv-python matplotlib numpy scikit-learn torch torchvision
> ```

### 3. Prepare Your Data

Place your satellite images and masks in:

* `data/images/` (`.tif` files)
* `data/labels/` (`.png` masks)

Ensure filenames match: `N.tif` ↔ `N.png`

---

## 🏗️ Training the Model

Run the main script:

```bash
python unet_water_seg.py
```

This will:

* Load and preprocess the data
* Train the U-Net model
* Save the best model to `models/unet_water_seg.pth`
* Generate visualizations and metrics in the `results/` folder

---

## 📊 Evaluation Metrics

After training, the model is evaluated using:

| Metric                            | Target   | Achieved |
| --------------------------------- | -------- | -------- |
| **IoU (Intersection over Union)** | ≥ 0.85   | TBD      |
| **Precision**                     | High     | TBD      |
| **Recall**                        | High     | TBD      |
| **F1-Score**                      | Balanced | TBD      |

Results are printed during training and saved in `results/training_curve.png`.

---

## 🖼️ Visualization

The script automatically generates:

* **Band visualization** (`results/band_visualization.png`)
* **Training curves** (loss & IoU)
* **Prediction samples** (input, ground truth, prediction)

Example:
![Band Visualization](results/band_visualization.png)

---

## 🧠 Model Architecture

We use a **U-Net** with:

* Encoder-decoder structure
* Skip connections
* 12-channel input adaptation
* Sigmoid output for binary segmentation

![U-Net Diagram](assets/unet-schematic.png)
*(Optional: Add a diagram later)*

---

## 📦 Repository Structure

```
multispectral-water-segmentation/
├── data/
│   ├── images/           # .tif files (12 bands)
│   └── labels/           # .png binary masks
├── models/               # Saved model weights
├── results/              # Outputs: plots, predictions
├── unet_water_seg.py     # Main training & inference script
├── requirements.txt      # Dependencies
└── README.md
```

---

## 📝 Requirements

```txt
torch
torchvision
rasterio
opencv-python
matplotlib
numpy
scikit-learn
tqdm
```
