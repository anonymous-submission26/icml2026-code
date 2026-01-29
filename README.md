# Anonymous ICML Submission

This repository contains an anonymized implementation corresponding to an ICML submission.

---

## Overview

This code implements the method described in the accompanying paper.
The repository supports loading a pretrained checkpoint and running inference through Orion-MSP.

---

## Dependencies
The package will automatically install required dependencies:
- `torch>=2.2,<3`
- `scikit-learn>=1.7,<2.0`
- `numpy`, `scipy`, `joblib`
- `xgboost`
- `transformers`
- `einops>=0.7`
- `huggingface-hub`
- `wandb` (for training)

---

## Pretrained Checkpoint

A pretrained checkpoint is provided via **an anonymous Google Drive link**:

> Checkpoint: https://drive.google.com/drive/folders/1K5Z1KFpE9dbsflg77C9Gw1O4ZvAIWtvR?usp=sharing

Download the checkpoint locally before running inference.

---

## Basic Usage

Orion-MSP provides a scikit-learn compatible interface for easy integration:

```python
from orionmsp_v15.sklearn import OrionMSPv15Classifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Load your data
data = pd.read_csv('your_data.csv')
X = data.drop('target', axis=1)
y = data['target']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the classifier
clf = OrionMSPv15Classifier()

# Fit the model (prepares data transformations)
clf.fit(X_train, y_train)

# Make predictions
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)
```

---

## Setting the Checkpoint Path

To use the pretrained model:

1. Download the checkpoint from the anonymous link.
2. Open `classifier.py`
3. Set the checkpoint path by editing:

```python
model_path = "PATH/TO/DOWNLOADED/CHECKPOINT"
```

---

## Notes

- Training code is included for completeness but is not required to reproduce the main results.
- This repository will be fully de-anonymized and released upon acceptance.
