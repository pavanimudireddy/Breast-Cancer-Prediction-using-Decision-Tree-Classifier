# Breast-Cancer-Prediction-using-Decision-Tree-Classifier
This repository contains a **machine learning project** for predicting breast cancer diagnoses using the **Breast Cancer Wisconsin (WBC) dataset**. The model uses a **Decision Tree Classifier** to classify tumors as **malignant (M)** or **benign (B)** based on features computed from digitized images of fine needle aspirate (FNA) of breast mass.

## 📝 Table of Contents
- [Dataset](#dataset)
- [Project Overview](#project-overview)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Decision Boundary Visualization](#decision-boundary-visualization)
- [Feature Importance](#feature-importance)
- [Decision Tree Visualization](#decision-tree-visualization)
- [Tools and Libraries Used](#Tools-and-Libraries-Used)
- [Project Structure](#Project-Structure)
- [Key Results](Key-Results)
- [Installation](#installation)
- [Usage](#usage)
- [References](#references)

## Dataset 

The dataset used in this project is the **Breast Cancer Wisconsin (Diagnostic) Dataset**

### Dataset Overview
- **Number of Instances:** 569  
- **Number of Features:** 30 numeric features + 1 ID column + 1 target column  
- **Problem Type:** Binary Classification (Malignant vs Benign)

### Target Variable
- **Column Name:** `diagnosis`  
- **Description:** Indicates whether the tumor is malignant or benign  
- **Values:**
  | Class | Label |
  |-------|-------|
  | Malignant | M → 1 |
  | Benign    | B → 0 |

> Note: The target variable is encoded as `1` for malignant and `0` for benign for machine learning purposes.

### Feature Description
The dataset contains **30 real-valued features** computed from digitized images of fine needle aspirate (FNA) of breast mass. Each feature is computed for three categories: **mean**, **standard error (SE)**, and **worst (largest)** value.

| Feature Category        | Example Features                  | Description |
|-------------------------|----------------------------------|-------------|
| Radius                  | `radius_mean`, `radius_se`, `radius_worst` | Size of the tumor |
| Texture                 | `texture_mean`, `texture_se`, `texture_worst` | Variation in gray-scale |
| Perimeter               | `perimeter_mean`, `perimeter_se`, `perimeter_worst` | Length around the tumor |
| Area                    | `area_mean`, `area_se`, `area_worst` | Tumor area |
| Smoothness              | `smoothness_mean`, `smoothness_se`, `smoothness_worst` | Local variation in radius lengths |
| Compactness             | `compactness_mean`, `compactness_se`, `compactness_worst` | Perimeter^2 / area - 1 |
| Concavity               | `concavity_mean`, `concavity_se`, `concavity_worst` | Severity of concave portions of the contour |
| Concave Points          | `concave points_mean`, `concave points_se`, `concave points_worst` | Number of concave portions of the contour |
| Symmetry                | `symmetry_mean`, `symmetry_se`, `symmetry_worst` | Symmetry of the tumor |
| Fractal Dimension       | `fractal_dimension_mean`, `fractal_dimension_se`, `fractal_dimension_worst` | “Coastline approximation” dimension of the tumor |

### Target Distribution
- Benign (B): 357 instances (~62.7%)  
- Malignant (M): 212 instances (~37.3%)  

### Dataset Characteristics
- No missing values in the numeric features after removing unnecessary columns (`id`, `Unnamed: 32`)  
- Features have varying scales and ranges  
- Some features are highly correlated (e.g., `radius_mean` and `perimeter_mean`)  
- Suitable for machine learning models such as **Decision Tree, Random Forest, Logistic Regression, SVM, and KNN**

## Project Overview
The goal of this project is to:
1. Clean and preprocess the data.
2. Explore relationships between features.
3. Train a **Decision Tree Classifier** to predict whether a tumor is malignant or benign.
4. Optimize the model to reduce overfitting.
5. Visualize the results including feature importance and decision boundaries.

## Data Preprocessing

Data preprocessing is an essential step to prepare the dataset for modeling. The following steps were performed:

1. **Remove irrelevant columns**  
   - Dropped the `id` column as it does not contribute to prediction.  
   - Dropped the `Unnamed: 32` column since it contained only null values.

2. **Handle duplicates**  
   - Checked for duplicate rows and removed them to avoid bias in the model.

3. **Encode target variable**  
   - Converted the `diagnosis` column from categorical to numerical:  
     - `M` → `1` (Malignant)  
     - `B` → `0` (Benign)

4. **Handle missing values**  
   - Verified that no missing values remain after column removal.

5. **Train-test split**  
   - Split the dataset into **training** and **testing** sets.  
   - Ratio: **70% training, 30% testing**  
   - Random state set for reproducibility.

## Exploratory Data Analysis (EDA)
 
- Outliers detected using IQR method but retained to preserve medical relevance.  
- Correlation analysis via heatmap shows some highly correlated features.  
- Target distribution: majority Benign, minority Malignant; visualized with bar plots.  
- Scatter plots reveal key feature pairs (e.g., `radius_mean` vs `texture_mean`) that separate classes, indicating suitability for Decision Tree modeling.

## Modeling

- **Model Used:** Decision Tree Classifier for breast cancer prediction.  
- **Reason for Choice:** Handles numeric features well, interpretable, and does not require feature scaling.  
- **Training:** Model trained on 70% of the dataset (`X_train`, `y_train`).  
- **Hyperparameter Tuning:** Used `max_depth` parameter to reduce overfitting; best depth selected via cross-validation.  
- **Comparison:** Decision Tree compared with Logistic Regression for decision boundary visualization using key features.

## Model Evaluation

- **Predictions:** Model tested on 30% of the dataset (`X_test`, `y_test`).  
- **Metrics Used:** Accuracy, Precision, Recall, F1-Score, AUROC, and Confusion Matrix.  
- **Decision Tree Performance:**
  - High accuracy and F1-score indicate effective classification.  
  - AUROC score confirms strong ability to distinguish Malignant vs. Benign.  
- **Confusion Matrix:** Visualized to show correct vs. incorrect predictions.  
- **Classification Report:** Provides detailed evaluation of model performance for both classes.

## Decision Boundary Visualization

- Used only two features (radius_mean, texture_mean) for 2D plotting of decision regions.
- Compared Decision Tree and Logistic Regression boundaries to analyze classification strategies.
- Decision Tree captures non-linear patterns, whereas Logistic Regression shows linear separation.
- Provides visual insight into how the model makes decisions.
- Helps in understanding feature influence in separating classes.

## Feature Importance

- Identify which features contribute most to the Decision Tree model.  
- `DecisionTreeClassifier.feature_importances_` used to compute importance scores.  
- Features like `radius_mean`, `perimeter_mean`, and `concavity_mean` have the highest importance.  
- Bar chart plotted to show relative importance of all features.  
- Helps in understanding which features drive model predictions and can guide feature selection.  

## Decision Tree Visualization

- Tree structure plotted using sklearn.tree.plot_tree to illustrate decision-making.
- Nodes display feature splits, thresholds, class counts, and impurity measures.
- Color-filled visualization improves readability of the tree.
- Enables interpretation of how the model classifies tumors.
- Helps communicate model logic to non-technical audiences.

## Tools and Libraries Used

- **Python** – Programming language used for the project  
- **Pandas** – Data manipulation and analysis  
- **NumPy** – Numerical computations  
- **Matplotlib & Seaborn** – Data visualization (plots, heatmaps, charts)  
- **Scikit-learn** – Machine learning models and evaluation metrics (Decision Tree, Logistic Regression, confusion matrix, classification report)

## 📂 Project Structure
Breast_Cancer_WBC_Prediction/
│
├── data/
│ └── wbc.csv # Dataset containing white blood cell features
│
├── notebooks/
│ └── breast_cancer_analysis.ipynb # Jupyter notebook with data preprocessing, model building, and evaluation
│
├── images/
│ ├── decision_tree.png # Visualization of the decision tree
│ ├── feature_importance.png # Feature importance bar chart
│ └── confusion_matrix.png # Confusion matrix visualization
│
├── requirements.txt # Python dependencies for the project
├── README.md # Project documentation

## 📊 Key Results
- Accuracy : 0.9298245614035088
- precision : 0.8805970149253731
- recall : 0.9365079365079365
- f1_score : 0.9076923076923077


