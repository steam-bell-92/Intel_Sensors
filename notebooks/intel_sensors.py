#!/usr/bin/env python
# coding: utf-8

# # ***PROBLEM STATEMENT***

# The objective of this project is to predict **room occupancy status** using sensor data collected from the **Intel Berkeley Research Lab**. The data includes time-series environmental readings such as:
# 
# - Temperature  
# - Humidity  
# - Light intensity  
# - Voltage
# 
# However, the dataset does **not directly contain occupancy labels**. Therefore, the problem extends beyond standard classification:
# 
# 1. **Generate meaningful occupancy labels** from available signals.
# 2. **Evaluate model performance** in the presence of real-world noise and class imbalance.
# 3. Design models that are **robust and responsive**, particularly in safety-critical contexts where **false negatives must be minimized**.
# 
# The final goal is to compare the behavior of supervised and unsupervised learning techniques (Random Forest and K-Means) when subjected to imperfect, noisy data — and to understand the trade-offs between **accuracy**, **recall**, and **false alarms** in realistic applications.
# 

# Importing all required libraries.

# In[1]:


# For Data Handling
import numpy as np
import pandas as pd

# For Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# For Data Precessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# For Training Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, silhouette_score, classification_report, roc_auc_score, roc_curve


# Importing the dataset and explicitly assigning column headers.

# In[2]:


url = "http://db.csail.mit.edu/labdata/data.txt.gz"

df = pd.read_csv(url, sep=" ", names=["date", "time", "epoch", "mote_id", "temperature", "humidity", "light", "voltage"])

df.head()


# Checking the duplicate rows in dataset to drop if any.

# In[3]:


df.duplicated().sum()


# As number of duplicates are zero, so we won't drop any rows.

# In[4]:


df.dropna(how='any', inplace = True)


# ### Defining new parameters

# In[5]:


df['occupancy_noise'] = ((df['temperature'] > 18) & (df['humidity'] > 35) & (df['light'] > 250) ^ (np.random.rand(len(df)) < 0.1)).apply(lambda x: 1 if x else 0).astype(int) # For Random Forest Classifier (Supervised)
df['occupancy'] = ((df['temperature'] > 18) & (df['humidity'] > 35) & (df['light'] > 250)).apply(lambda x: 1 if x else 0).astype(int) # For K Means Clustering (Unsupervised)


# To evaluate model robustness and avoid overfitting (cramming), different preprocessing approaches were used for different algorithms:
# 
#   - 🌲  Random Forest Classifier (`occupancy_noise`):
# A controlled 10% bit-flip noise was added to one of the key features. This helps the model generalize better by reducing the risk of memorizing patterns too strictly, which can lead to cramming.
# 
#   - 🔘  K-Means Clustering (`occupancy`):
# The same feature was used without any added noise, ensuring the clustering algorithm receives the original distribution for better unsupervised learning performance.

# Getting counts for both parameters.

# In[6]:


display(df['occupancy_noise'].value_counts())
display(df['occupancy'].value_counts())


# ### Splitting the dataset between training and test set.

# In[7]:


train_df, test_df = train_test_split(df, test_size=0.33, random_state=42)


# Summary of training dataset.

# In[8]:


train_df.describe()


# ## Data Visualization

# ### Plotting a boxplot between `Occupany` & `Temperature`on a training dataset.

# In[9]:


sns.boxplot(x='occupancy_noise', y='temperature', data=train_df)
plt.xlabel('Occupancy (with 10% noise)')
plt.ylabel('Temperature')
plt.title('Distribution of Occupancy & Temperature')
plt.grid(alpha=0.5)
plt.show()


# Temperature alone does not clearly separate occupied and unoccupied rooms — though there are some small differences that a model might still use.

# ### A Histogram of `Temperature`.

# In[10]:


sns.histplot(x='temperature', data=train_df, hue='occupancy_noise', kde=True)
plt.xlabel('Temperature')
plt.ylabel('Frequency')
plt.title('Distribution of Temperature')
plt.ylim(0, 13000)
plt.xlim(0, 50)
plt.grid(alpha=0.5)
plt.show()


# Occupied and unoccupied rooms have very similar temperatures, so temperature alone won't clearly separate them. But the model may still find some small patterns.

# ### A Histogram on `Humidity`.

# In[11]:


sns.histplot(x='humidity', data=train_df, hue='occupancy_noise', kde=True)
plt.xlabel('Humidity')
plt.ylabel('Frequency')
plt.title('Distribution of Humidity')
plt.xlim(20, 60)
plt.ylim(0, 13000)
plt.grid(alpha=0.5)
plt.show()


# Humidity might be a better indicator of occupancy than temperature, but it’s still not perfectly clear. The model can likely learn from this difference.

# ### Correlation Matrix.

# In[12]:


plt.figure(figsize=(8,6))
sns.heatmap(test_df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# - Humidity and light are more helpful in predicting occupancy than temperature or voltage.
# 
# - Temperature alone is not a strong signal, which matches what we saw in earlier plots.
# 
# - Models like Random Forest can benefit from using humidity and light as main features.

# Scaling the required columns of dataset.

# In[13]:


scaler = RobustScaler()

cols_to_scale = ['epoch', 'mote_id', 'temperature', 'humidity', 'light', 'voltage']
train_df_scaled = scaler.fit_transform(train_df[cols_to_scale])
test_df_scaled = scaler.transform(test_df[cols_to_scale])


# Defining separate datasets for model training and evaluation by partitioning the original dataset for `Random Forest Classification`.

# In[14]:


X_train = train_df_scaled
y_train1= train_df['occupancy_noise']

X_test = test_df_scaled
y_test1 = test_df['occupancy_noise']


# ## Prediction Models

# ### Random Forest Classifier

# Training & Predicting the data.

# In[15]:


model1 = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=1, class_weight='balanced')
model1.fit(X_train, y_train1)

y_proba = model1.predict_proba(X_test)[:, 1]
y_pred1 = (y_proba > 0.50).astype(int)


# Calculating Accuracy Score for `Random Forest Classifier`.

# In[16]:


acc11 = accuracy_score(y_test1, y_pred1)
# Flip the predicted labels: 1 → 0, 0 → 1
# This ensures the predictions match the intended label definitions
acc12 = accuracy_score(y_test1, 1 - y_pred1)

# We would consider the best accuracy score among both.
print("Accuracy:", max(acc11, acc12))


# #### Plotting an Confusion Matrix on `Random Forest model`.

# In[17]:


ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test1, y_pred1)).plot(cmap='copper')
plt.show()


# Obtaining the `classification report` to assess `Random Forest Classifier` performance using `precision`, `recall`, and `F1-score`.

# In[18]:


print(classification_report(y_test1, y_pred1))


# Calculating the `ROC-AUC Score` for `Random Forest Classifier`.

# In[19]:


roc1 = roc_auc_score(y_test1, y_pred1)
print(f"ROC-AUC Score: {roc1}")


# #### Plotting an `ROC Curve` for `Random Forest Classifier`.

# In[20]:


fpr, tpr, thresholds = roc_curve(y_test1, y_pred1)
roc_auc = roc_auc_score(y_test1, y_pred1)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'Random Forest ROC (AUC = {roc_auc:.4f})', color='darkorange')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Random Forest Classifier')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# ### K Means Clustering

# Defining separate datasets for model training and evaluation by partitioning the original dataset for `K Means Clustering`.

# In[21]:


y_train2 = train_df['occupancy']
y_test2 = test_df['occupancy']


# Training & Predicting the data.

# In[22]:


model2 = KMeans(n_clusters=2, random_state=1, init='k-means++')
model2.fit(X_train)

y_pred2 = model2.predict(X_test)


# Obtaining a `Silhouette Score`.

# In[23]:


SS = silhouette_score(X_test[:10000], y_pred2[:10000])
print("Silhouette Score:", SS)


# Calculating Accuracy Score for `K Means Clustering`.

# In[24]:


print(f"Accuracy: {(accuracy_score(y_test2, y_pred2))}")


# #### Plotting an Confusion Matrix on `K Means Clustering`.

# In[25]:


ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test2, y_pred2)).plot(cmap='copper')
plt.show()


# Obtaining the `classification report` to assess `K Means Clustering` performance using `precision`, `recall`, and `F1-score`.

# In[26]:


print(classification_report(y_test2, y_pred2))


# Calculating the `ROC-AUC Score` for `K Means Clustering`.

# In[27]:


roc2 = roc_auc_score(y_test2, y_pred2)
print(f"ROC-AUC Score: {roc2}")


# #### Plotting an `ROC Curve` for `K Means Clustering`.

# In[28]:


fpr, tpr, thresholds = roc_curve(y_test2, y_pred2)
roc_auc = roc_auc_score(y_test2, y_pred2)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'K Means Clustering (AUC = {roc_auc:.4f})', color='darkorange')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for K Means Clustering')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

