# 🧠 Sensor-Based Occupancy Prediction
This project investigates how well we can detect room occupancy from wireless sensor network data, specifically under noisy real-world conditions.<br>
The dataset comes from the Intel Berkeley Research Lab, featuring readings of temperature, humidity, light, and voltage from 54 deployed sensors.

---

## 💡 Core Concept
The core idea is to evaluate model robustness and sensitivity to noise in sensor data. Two parallel tracks were used:

- `Random Forest Classifier` (Supervised Learning)<br>
Trained on a version of the data where 10% bit-flip noise was added to a key feature. This tests whether the model can still generalize well, instead of memorizing patterns.

- `K-Means Clustering` (Unsupervised Learning)<br>
Trained on clean, original features to assess how well natural groupings align with actual occupancy. This helps understand the structure of the data without labels.
>This contrast shows how `supervised` and `unsupervised` learning deal differently with signal and noise.

---

.

## 📁 File Structure

Intel_Sensors/
│
├── Intel_Sensors.ipynb    # Jupyter notebook containing entire ML Workflow
│
└── README.md              # This one !!
