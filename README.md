<div align="center">
<h1 align="center">🚦 Composite Model for Traffic Signal Control</h1>

<h3>Optimal Traffic Signal Control for an Atypical Isolated Intersection
Using Composite AI Model</h3>

[Seunghyeok Lee](https://orcid.org/0009-0009-4709-7931)<sup>a</sup>, [Bumku Choi](https://orcid.org/0009-0004-1728-7420)<sup>b</sup>, [Weeyoung Kwon](https://orcid.org/0009-0004-4781-1385)<sup>c</sup>, [Jungeun Cha](https://orcid.org/0009-0000-0579-6405)<sup>d</sup>, [Jihyo Jung](https://orcid.org/0009-0008-3803-8625)<sup>d</sup>, [Hoe Kyoung Kim](https://orcid.org/0000-0001-5057-2558)<sup>e</sup> 

<sup>a</sup> Department of Electronic Engineering, Dong-A University, Busan, Korea

<sup>b</sup> Department of Forecast, Geosystem Research Corporation, Busan, Korea

<sup>c</sup> Graduate School of Advanced Imaging Science, Multimedia, and Film (GSAIM), Chung-Ang University, Seoul, Korea

<sup>d</sup> Department of Urban Planning and Landscape Architecture, Dong-A University, Busan, Korea

<sup>e</sup> Department of Urban Planning and Engineering, Dong-A University, Busan, Korea

</div>

## 📝 Project Overview

This project proposes and evaluates a novel AI-based composite framework for traffic signal control optimization, which integrates a Long Short-Term Memory (LSTM) model for traffic flow prediction, a Deep Neural Network (DNN) for vehicle speed estimation, and a Genetic Algorithm (GA) for signal phase optimization. It has been validated through high-fidelity microscopic simulations (i.e., VISSIM), calibrated with field data at the atypical six-leg intersection in Busan, Korea.

## 🔑 Key Features

## 🧩 Requirements
- Python 3.8+
- Tensorflow
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

### 🍳 Installation
Clone the repository:

```bash
git clone [https://github.com/seunghyeokleeme/Composite-Model-Traffic-Control.git](https://github.com/seunghyeokleeme/Composite-Model-Traffic-Control.git)

cd Composite-Model-Traffic-Control
```

Create and activate a virtual environment (recommended):

### Using python's built-in venv

```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
.\venv\Scripts\activate   # On Windows
```

Install the required packages:

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib ...
```

## 💿 Dataset
This project utilizes real-world data from the 2023 Yeonsan Intersection Smart City Project. The data is divided into two main categories: traffic volume and traffic speed.

- Traffic Volume Data: `data/traffic_volume/`

    `traffic_volume.csv`: Directional traffic volume data.

    `traffic volume timestemp.csv`: Timestamped version of the directional traffic volumes.

- Traffic Speed Data: `data/traffic_speed/`

    `dnn_train.csv`: Training data for the DNN model, containing features like predicted traffic volume, signal timings, and corresponding vehicle speeds.

The expected directory structure is:

```
├── data/
│   ├── traffic_speed/
│   │   └── dnn_train.csv
│   └── traffic_volume/
│       ├── traffic_volume.csv
│       └── traffic_volume_timestemp.csv
├── traffic_speed/
│   ├── eval.py
│   ├── make_data.py
│   ├── train.py
│   ├── speed_dataset.py
│   └── predict.py
├── traffic_volume/
│   ├── make_data.py
│   ├── eval.py
│   ├── predict.py
│   ├── train.py
│   └── eval.py
└── traffic_signal_optimizer.py
```

## 🚀 How to Run
The project is executed in stages, following the hierarchical model structure.

1. Prepare Traffic Volume Data
This script processes the raw traffic volume CSV files into a format suitable for training the LSTM model.

```bash
python ./traffic_volume/make_data.py
```

2. Train the Traffic Volume Prediction Model (LSTM)
```bash
python ./traffic_volume/train.py \
    --lr 1e-4 --seed 42 \
    --epochs 100
```

3. Train per-approach average Speed Prediction Model (DNN)

4. Evaluate Models


### Evaluate LSTM model

### Evaluate DNN model

## 🧪 System Architecture & Methodology

## Problem Formulation

## 📈 Results

### Traffic Volume Prediction Results

### Traffic Speed Prediction Results

### Traffic Phase Optimization Results


📜 Reference
This project is based on the methodology outlined in the following documents:
