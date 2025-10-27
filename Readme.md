# Task 5 - Decision Trees & Random Forests  
**AI & ML Internship | By Mrutyunjay Joshi**

---

## Objective
Learn and implement **tree-based models** for classification and regression using the **Heart Disease Dataset**.  
Understand model interpretability through **Decision Trees**, **Random Forests**, and **Feature Importance**.

---

## Tech Stack
-> Python 
-> Pandas  
-> NumPy  
-> Scikit-learn  
-> Matplotlib  
-> Graphviz  

## 🚀 Project Flow

### Step 1 : Load Dataset
Used **Heart Disease Dataset** (Kaggle).  
Explored shape, missing values, and data distribution.

### Step 2 : Split the Data
Divided into:
- 75% Training Data  
- 25% Testing Data  

### Step 3 : Train Decision Tree Classifier
Built a **DecisionTreeClassifier** and checked accuracy using the test set.

### Step 4 : Visualize the Tree
Plotted the trained tree using `plot_tree()` to understand splitting logic and decision hierarchy.

### Step 5 : Control Depth – Avoid Overfitting
Limited depth using `max_depth=4` to balance bias–variance trade-off.

### Step 6 : Train Random Forest Classifier
Trained a **RandomForestClassifier (n_estimators=150)** for better generalization through ensemble learning.

### Step 7 : Interpret Feature Importance
Plotted which features had the highest importance in classification.

### Step 8 : Cross-Validation
Performed **5-Fold Cross Validation** for robust evaluation and stability check.

---

## Model Comparison

| Model | Accuracy |
|:------|:----------:|
| Decision Tree | ~0.82 |
| Depth Controlled Tree | ~0.84 |
| Random Forest | ~0.88 |

**Random Forest** gave the highest and most stable accuracy.  
Controlled-depth trees reduced overfitting effectively.

## Key Concepts Learned

### Decision Tree
A tree structure that splits data using **Entropy / Gini Index** to maximize **Information Gain**.

### Random Forest
An **ensemble** of multiple decision trees built using **bagging** (bootstrap aggregation).  
Averaging results improves stability and reduces overfitting.

### Overfitting Control
Handled by limiting tree depth and number of features considered at each split.

### Feature Importance
Helps interpret which features contribute most to predictions.

### Cross-Validation
Ensures that model performance is consistent across multiple folds.

---

## Libraries Used
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
