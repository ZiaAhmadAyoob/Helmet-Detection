# 🪖 Helmet Detection using YOLOv11

A deep learning–based object detection application that identifies whether a person is **wearing a helmet or not** from images. The system uses **YOLOv11** for detection and is deployed as a **Streamlit web application** for easy access and real-time inference.

---

## 🔍 Problem Statement
Lack of helmet usage is a major cause of severe injuries in road accidents. Manual monitoring is inefficient and not scalable. This project automates helmet compliance detection using computer vision.

---

## 🎯 Purpose
- Detect helmet and no-helmet cases automatically
- Support traffic safety and rule enforcement
- Provide a simple web-based interface for inference

---

## 🧠 Model & Dataset
- **Model:** YOLOv11 (Ultralytics)
- **Task:** Object Detection
- **Classes:** Helmet, No Helmet
- **Dataset:** Custom / public helmet detection dataset
- **Annotation Format:** YOLO

---

## ⚙️ Tech Stack
- Python
- YOLOv11
- OpenCV
- NumPy
- Streamlit

---

## 🏗️ Workflow
1. Dataset preparation and annotation
2. Model training using YOLOv11
3. Evaluation using Precision, Recall, and mAP
4. Deployment using Streamlit Cloud

---

## 🖥️ Web Application
The Streamlit app allows users to:
- Upload images
- Detect helmet compliance
- Visualize bounding boxes and labels

🔗 **Live App:** https://helmet-detection-appp.streamlit.app/
