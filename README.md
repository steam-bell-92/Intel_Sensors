# 🧠 Sensor-Based Occupancy Prediction
This project investigates how well we can ***detect room occupancy*** from wireless sensor network data, specifically under noisy real-world conditions.<br>
The dataset comes from the Intel Berkeley Research Lab, featuring readings of temperature, humidity, light, and voltage from 54 deployed sensors.

> The Datset is available on Kaggle as well as UCI ML Datasets:<a href="https://www.kaggle.com/datasets/divyansh22/intel-berkeley-research-lab-sensor-data">Kaggle Dataset (divyansh22)</a>

---

## 💡 Core Concept
The core idea is to evaluate model robustness and sensitivity to noise in sensor data. Two parallel tracks were used:

- `Random Forest Classifier` (Supervised Learning)<br>
Trained on a version of the data where 10% bit-flip noise was added to a key feature. This tests whether the model can still generalize well, instead of memorizing patterns.

- `K-Means Clustering` (Unsupervised Learning)<br>
Trained on clean, original features to assess how well natural groupings align with actual occupancy. This helps understand the structure of the data without labels.
>This contrast shows how `supervised` and `unsupervised` learning deal differently with signal and noise.

---

## ⚖️ Tradeoff Design Philosophy: Prioritizing Safety Over Raw Accuracy while training K-Means Clustering model

- This project was developed with a deliberate and thoughtful trade-off strategy: to prioritize detecting real occupancy at all costs — even if that means tolerating false alarms.
- This is especially relevant for applications where a missed detection (False Negative) could have severe consequences, such as:
  - 🔥 Fire or hazard detection systems
  - 👴 Elderly care and fall monitoring
  - 🏢 Unauthorized access or motion sensing in secure areas

>It's better to be alerted unnecessarily than to ignore a real threat.

### 🎯 Why Accuracy Was Not the Primary Goal
- The original dataset is heavily imbalanced toward the "not occupied" class.
- Had the model simply tried to maximize accuracy, it would have:
- Predicted “not occupied” almost everywhere

    - Achieved  around 0.80 accuracy, but completely failed to detect real occupancy (high FN)
    - Yielded a low ROC-AUC (~0.43) due to poor class separation

This intentional bias toward recall and responsiveness produced:

|Metric	             | Value / Behavior |	Interpretation                                             |
|--------------------|----------------  |--------------------------------------------------------------|
|✅ Recall        	 | High	            |Catches nearly all occupancy cases                            |
|⚠️ Precision       |	Moderate to Low |	More false alarms, acceptable in safety context            |
|❌ Accuracy        | ~0.36	          | Drops due to many FPs, but that’s expected and acceptable  |
|✅ ROC-AUC         |	~0.58	          |Better than random (~0.50), shows improved class separation |
|✅ False Negatives |	Very Low	      |Critical success — system rarely misses true occupancy      |

The higher ROC-AUC (~0.58) is not despite the trade-off — it's because of it.
The model actually learned to distinguish between classes better by focusing on what matters.

### 🧠 My Intution
This trade-off reflects how machine learning should behave in real deployment, especially in high-stakes domains:
  - Accuracy and ROC-AUC are context-sensitive — they shouldn't be blindly optimized.
  -  Safety-critical ML must be designed to handle uncertainty, imbalance, and consequence.
  - False Positives waste time, but False Negatives cost lives.
  - This project demonstrates not just technical modeling, but responsible ML system design — with awareness of domain priorities, risks, and ethical consequences.

---

## 📊 Results

### 🔁 Random Forest Classifier (Supervised with 10% noise)    

| Metric              | Value     |                                
|---------------------|-----------|                                
| Accuracy            | `~0.79`     |                              
| ROC-AUC             | `~0.78`     |                              
                                                                   
### 🔘 K-Means Clustering (Unsupervised)

| Metric              | Value     |
|---------------------|-----------|
| Accuracy (Post-labeling) | `~0.36` |
| ROC-AUC             | `~0.58`     |
| Silhouette Score    | `~0.85`     |

---

## 📁 File Structure

```
Intel_Sensors/
│
├── Intel_Sensors.ipynb    🔹 Jupyter notebook containing entire ML Workflow
├── intel_sensors.py       🔹 Python File
└── README.md              🔹 This file !!
```

## 👤 Author
Anuj Kulkarni - aka - steam-bell-92

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
