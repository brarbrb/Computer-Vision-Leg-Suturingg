# Computer-Vision-Leg-Suturingg
This repository provides a pipeline for training and evaluating YOLO models to detect surgical tools (e.g., tweezers, needle driver) in images and videos. It includes data augmentation configurations, model comparisons, prediction scripts, and exploratory data analysis (EDA).

## 📁 Project Structure
├── video.py # Predicts bounding boxes on a video and saves the annotated output
├── predict.py # Predicts bounding boxes on a single image and visualizes the output
├── comparing_models.ipynb # Notebook comparing performance of different models
├── EDA_training_models.ipynb # Exploratory Data Analysis and training steps
├── custom_aug.yaml # Custom data augmentation configuration for yolov9
├── pseudo.yaml # Pseudo-labeling dataset config for yolov9 (on in distribution videos)
├── pseudo_ood.yaml # Pseudo-labeling for out-of-distribution data
├── surgical.yaml # Main training configuration for surgical dataset for base model with no modifications


## 🛠️ Requirements

- Python 3.8+
- [Ultralytics YOLOv9](https://docs.ultralytics.com/)
- OpenCV
- Matplotlib

For efficient training use GPU: We had Tesla T100

Install dependencies:
```bash
pip install -r requirements.txt
```

Models
The YOLOv8 model is used for surgical tool detection with the following classes:
- 0: Empty

- 1: Tweezers

- 2: Needle_driver

##🎥 Run Inference on Videos
```bash 
python video.py path/to/input.mp4 path/to/output.mp4 --model path/to/your_model.pt
```
Default model path: runs/detect/base_model2/weights/best.pt


## 🖼️ Run Inference on Images
```bash
python predict.py path/to/input.jpg path/to/output.jpg --model path/to/your_model.pt
```
Default model path: trained_models/base_model2.pt

📊 Notebooks
comparing_models.ipynb — Compare loss and mAp of different models.

EDA_training_models.ipynb — Analyzing the dataset, training the model, Pipeline for 
