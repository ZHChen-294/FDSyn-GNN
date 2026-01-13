# FDSyn-GNN

**Paper:** *An Interpretable Functional-Dynamic Synaptic Graph Neural Network for Major Depressive Disorder Diagnosis from rs-fMRI*  

<div align="center">
  <img src="https://github.com/ZHChen-294/FDSyn-GNN/blob/main/Img/FDSyn-GNN.jpg">
</div>

<p align="center">
Created by <a href="https://github.com/ZHChen-294">Zhihong Chen</a><sup>a</sup>, Jiayi Peng<sup>b</sup>, Xiaorui Han<sup>c</sup>, Mengfan Wang<sup>c</sup>, Jiang Wu<sup>a</sup>, Xinhua Wei<sup>c</sup>, Zhengze Gong<sup>de</sup>, <a href="https://scholar.google.com.hk/citations?user=ClUoWqsAAAAJ&hl=zh-CN&oi=ao">Dezhong Yao</a><sup>a</sup>*, Li Pu<sup>a</sup>* and <a href="https://scholar.google.com.hk/citations?user=KJqKYq4AAAAJ&hl=zh-CN">Hongmei Yan</a><sup>a</sup>*
</p>

_<sup>a</sup> The Clinical Hospital of Chengdu Brain Science Institute, Sichuan Institute for Brain Science and Brain-Inspired Intelligence, School of Life Science and Technology, University of Electronic Science and Technology of China, Chengdu, 610054, Sichuan, China_

_<sup>b</sup> Department of Radiology, Sichuan Provincial People’s Hospital, University of Electronic Science and Technology of China, Chengdu, 610072, Sichuan, China_

_<sup>c</sup> Department of Radiology, Guangzhou First People's Hospital, School of Medicine, South China University of Technology, Guangzhou, 510180, Sichuan, China_

_<sup>d</sup> Information and Data Centre, Guangzhou First People's Hospital, School of Medicine, South China University of Technology, Guangzhou, 510180, Sichuan, China_

_<sup>e</sup> Information and Data Centre, The Second Affiliated Hospital, School of Medicine, South China University of Technology, Guangzhou, 510180, Sichuan, China_

---

# 🧠 Project Overview

This repository contains **PyTorch** implementation for **FDSyn-GNN**.

This project is implemented in **Python (≥3.8)** and conducted with **stratified five-fold cross-validation**.

Code: https://github.com/ZHChen-294/FDSyn-GNN

---

## ⚙️ Dependencies & Environment

The project depends on the following core libraries.  
All required packages are listed in `requirements.txt`.

**Main dependencies include:**
- **Deep learning & GNNs:** `torch`, `torchvision`, `torch-geometric`, `einops`
- **Scientific computing:** `numpy`, `scipy`, `pandas`
- **Machine learning:** `scikit-learn`
- **Utilities:** `tqdm`, `tensorboard`

> **Note:**  
> `torch-geometric` should be installed using the wheel that matches your **PyTorch** and **CUDA** versions.  
> Please refer to: https://pytorch-geometric.readthedocs.io

---

## 💻 Hardware Requirements (Recommended)

- **GPU:** NVIDIA RTX 3060 (≥12 GB VRAM) or higher  
- **CPU:** ≥ 12 cores  
- **RAM:** ≥ 32 GB  
- **Storage:** ≥ 100 GB free disk space (for multi-fold caching)  
- **OS:** Ubuntu 20.04+ or Windows 10+  
- **CUDA Toolkit:** 11.8 (must match the PyTorch build)  
- **Python:** 3.8 or 3.9 recommended  

---

## 📂 Data Availability

Experiments are conducted on the **DIRECT (REST-meta-MDD)** dataset, a large-scale resting-state fMRI dataset for **Major Depressive Disorder (MDD)**.  
The dataset is publicly released by the REST-meta-MDD consortium and can be accessed according to the consortium’s data-sharing policy.

---

## 🚀 Installation

Clone the repository and install all dependencies:

```bash
git clone https://github.com/ZHChen-294/MAF-GNN.git
cd MAF-GNN
pip install -r requirements.txt

