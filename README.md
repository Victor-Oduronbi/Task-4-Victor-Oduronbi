# Facial Image Recognition Using Eigenfaces (PCA) and Support Vector Machines (SVM)

An end-to-end computer vision and high-dimensional classification pipeline developed during my **Data Science Internship** at **DecodeLabs**. This project leverages Principal Component Analysis (PCA) for dimensional footprint reduction and Support Vector Machines (SVM) to execute facial identity recognition.

## 📋 Task Overview
The goal of this project is to apply unsupervised dimensionality reduction techniques alongside maximum-margin classifiers to process raw facial photograph pixels. Key milestones include:
* **Eigenface Feature Extraction:** Utilizing randomized Singular Value Decomposition (SVD) to capture the top 150 orthogonal spatial structures.
* **Hyperparameter Optimization:** Implementing a Grid Search Cross-Validation framework to evaluate optimal soft-margin penalties (\(C\)) and radial basis function (RBF) kernel widths (\(\gamma\)).
* **Class Imbalance Mitigation:** Incorporating balanced class weights to prevent majority-class bias toward highly sampled individuals.

---

## 🗂️ Dataset Glossary & Features
The project utilizes the **Labeled Faces in the Wild (LFW) People Database** sourced from [Kaggle](https://www.kaggle.com/datasets/atulanandjha/lfwpeople).
* **Dimensionality:** 337 evaluation samples across 8 distinct global political figures.
* **Input Features:** Unprocessed pixel arrays flattened into 1-dimensional vectors from their raw \(62 \times 47\) image dimensions.
* **Target Classes:** `Ariel Sharon`, `Colin Powell`, `Donald Rumsfeld`, `George W. Bush`, `Gerhard Schroeder`, `Hugo Chavez`, `Junichiro Koizumi`, `Tony Blair`.

---

## ⚙️ Data Preprocessing & Methodology

### 1. Unsupervised Dimensionality Reduction (PCA)
Directly processing thousands of unscaled pixels introduces an immense computational burden and severe multi-collinearity. To fix this, a randomized PCA pipeline extracted the top **150 principal components (Eigenfaces)**. Incorporating `whiten=True` scales each component to unit variance, satisfying the structural inputs needed for an RBF kernel.

```python
pca = PCA(n_components=150, whiten=True, svd_solver='randomized', random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca, svc)
```

### 2. Hyperparameter Grid Search Optimization
To maximize generalization on unseen testing faces, hyperparameter spaces were swept across an exhaustive parameter matrix:

```python
param_grid = {
    'svc__C': [1, 5, 10, 50], 
    'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]
}
grid = GridSearchCV(model, param_grid)
```

---

## 📈 Prediction Verification Grid Analysis

The model performance is evaluated visually by rendering a predictive grid array where correct identifications map to black text and classification failures are clearly isolated in red:

<p align="center">
  <img width="525" height="437" alt="image" src="https://github.com/user-attachments/assets/1b0782de-1399-42eb-b573-055be105681f" />
</p>

---

## 📊 Model Evaluation Results

The baseline PCA-SVM pipeline achieved an overall classification accuracy of **85.00%** on unseen test images:

### 1. Classification Performance Breakdown



| Target Identity | Precision | Recall | F1-Score | Support (Samples) |
| :--- | :---: | :---: | :---: | :---: |
| **Ariel Sharon** | 0.65 | 0.87 | 0.74 | 15 |
| **Colin Powell** | 0.83 | 0.88 | 0.86 | 68 |
| **Donald Rumsfeld** | 0.70 | 0.84 | 0.76 | 31 |
| **George W. Bush** | 0.97 | 0.80 | 0.88 | 126 |
| **Gerhard Schroeder** | 0.76 | 0.83 | 0.79 | 23 |
| **Hugo Chavez** | 0.93 | 0.70 | 0.80 | 20 |
| **Junichiro Koizumi** | 0.86 | 1.00 | 0.92 | 12 |
| **Tony Blair** | 0.82 | 0.98 | 0.89 | 42 |
| **Overall Accuracy** | | | **0.85** | **337** |
| *Macro Average* | 0.82 | 0.86 | 0.83 | 337 |
| *Weighted Average* | 0.86 | 0.85 | 0.85 | 337 |

### 2. Statistical Metrics & Confusions Breakdown
* **Majority Sample Stability:** Despite having the highest representation in the dataset, `George W. Bush` achieved an exceptional **0.97 precision**. This confirms that incorporating `class_weight='balanced'` effectively stopped the SVM hyperplanes from defaulting and over-predicting his class.
* **Targeted Miss Analysis:** As visualized in the validation test grid, the model flags prediction errors in red text. For example, it misclassified a distinct image of George W. Bush as **Rumsfeld** (Row 3, Column 2) and an image of Donald Rumsfeld as **Blair** (Row 4, Column 3). This highlights localized facial lighting or expression variances that structural PCA components can occasionally smooth over.
<p align="center">
  <img width="533" height="533" alt="image" src="https://github.com/user-attachments/assets/469d16ff-aa6f-4b1b-a3e2-46cb6aba2cd1" />
</p>
---

## 🚀 Future Enhancements
Upcoming iterations of this repository will focus on:
1. **Non-Linear Dimensionality Reduction:** Swapping standard PCA for Kernel PCA (KPCA) or t-SNE to capture complex facial surface topography.
2. **Deep Spatial Feature Learning:** Transitioning away from Scikit-Learn pipelines into Convolutional Neural Networks (CNNs) using PyTorch or TensorFlow to extract local spatial transformations natively.

---

## 🤝 Acknowledgments
* **DecodeLabs** for providing the advanced computer vision internship assignment track.
* **Kaggle & LFW Curators** for maintaining the open-source face dataset.
