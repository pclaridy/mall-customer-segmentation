# Advanced Segmentation Analysis: Unlocking Mall Customer Insights

## Table of Contents
1. [Problem Statement](#1-problem-statement)  
2. [Data Source](#2-data-source)  
3. [Data Cleaning & Preprocessing](#3-data-cleaning--preprocessing)  
4. [Exploratory Data Analysis (EDA)](#4-exploratory-data-analysis-eda)  
5. [Modeling Approach](#5-modeling-approach)  
6. [Evaluation Metrics](#6-evaluation-metrics)  
7. [Outcome](#7-outcome)  
8. [Tools Used](#8-tools-used)  
9. [Business Impact / Use Case](#9-business-impact--use-case)  

---

## 1. Problem Statement

Effectively segmenting customers is a key component of personalized marketing and customer relationship management in the retail sector. However, identifying meaningful customer groups can be challenging without labeled data or a clear understanding of behavioral patterns. This project aims to apply unsupervised learning techniques to segment mall customers based on demographic and spending attributes such as age, income, and spending score. By comparing clustering algorithms including K-Means, DBSCAN, Spectral Clustering, Agglomerative Clustering, and Gaussian Mixture Models, the goal is to uncover distinct customer profiles that can inform targeted marketing strategies, enhance customer experiences, and optimize resource allocation in retail environments.

---

## 2. Data Source

The dataset [**mall_customers.csv**](http://github.com/pclaridy/mall-customer-segmentation/blob/main/data/mall_customers.csv) contains demographic and behavioral data on mall customers. Key variables include:

- **Target**: Unlabeled (unsupervised) – the goal is to identify clusters based on shared characteristics  
- **Predictors**: Gender, Age, Annual Income (in thousands), and Spending Score (1–100)  
- The dataset includes 200 records representing a diverse mix of customers across different income levels and spending behaviors

---

## 3. Data Cleaning & Preprocessing

Prior to modeling, several preprocessing steps were performed:

- **Standardization**: All numerical variables were scaled using standard normalization to ensure fair treatment by distance-based algorithms.
- **Missing Values**: The dataset contained no missing values and required no imputation.
- **Categorical Encoding**: Gender was one-hot encoded where necessary, though clustering was primarily performed on numerical variables.
- **Dimensionality Reduction**: PCA was applied to reduce the dataset to two principal components for visualization without compromising interpretability.

---

## 4. Exploratory Data Analysis (EDA)

Basic statistical analysis revealed the following:

- Spending Scores were well distributed but showed patterns by income and age groups.
- Visualizations confirmed the potential for natural groupings based on Annual Income and Spending Score.

---

## 5. Modeling Approach

The following clustering algorithms were applied using PCA-reduced data:

### K-Means Clustering

K-Means identified five distinct customer segments. The clusters showed meaningful separation in terms of spending behavior and income.

![K-Means Clustering](https://github.com/pclaridy/mall-customer-segmentation/blob/main/KMeans_Clustering.png)

### DBSCAN Clustering

DBSCAN captured one dense primary cluster and several outliers. This approach was less sensitive to outliers and noise compared to K-Means.

![DBSCAN Clustering](https://github.com/pclaridy/mall-customer-segmentation/blob/main/DBSCAN_Clustering.png)

### Agglomerative Clustering

This hierarchical method also revealed five segments. Some overlap was observed in PCA space, but the grouping logic remained consistent with other models.

![Agglomerative Clustering](https://github.com/pclaridy/mall-customer-segmentation/blob/main/Agglomerative_Clustering.png)

### Spectral Clustering

Spectral Clustering identified complex cluster shapes not easily detected by linear methods. It highlighted nonlinear group boundaries and captured subtler customer distinctions.

![Spectral Clustering](https://github.com/pclaridy/mall-customer-segmentation/blob/main/Spectral_Clustering.png)

### Gaussian Mixture Model Clustering

GMM revealed probabilistic clusters with flexible shapes and overlapping boundaries. This method was helpful in interpreting mixed behaviors among certain segments.

![Gaussian Mixture Model Clustering](https://github.com/pclaridy/mall-customer-segmentation/blob/main/GMM_Clustering.png)

---

## 6. Evaluation Metrics

Several techniques were used to evaluate and validate clustering performance:

- **Elbow Method**: Helped determine the optimal number of clusters for K-Means.
- **Silhouette Score**: Measured the separation distance between clusters. Scores supported the five-cluster solution as a good fit.
- **Davies-Bouldin Index**: Provided further validation of compact and well-separated clusters.

These internal validation metrics supported the selection of five clusters as optimal.

---

## 7. Outcome

The clustering analysis produced five well-differentiated customer groups. Key characteristics included:

- **Cluster 1**: Young, low-income, high-spending individuals – likely students or impulse buyers
- **Cluster 2**: Older, high-income, low-spending individuals – possibly financially stable but conservative shoppers
- **Cluster 3**: Middle-aged, moderate-income, medium-spending individuals – average customers with steady behavior
- **Cluster 4**: High-income, high-spending customers – likely premium targets for luxury promotions
- **Cluster 5**: Low-income, low-spending individuals – budget-conscious customers

These profiles can inform targeted messaging, product placement, and loyalty initiatives.

---

## 8. Tools Used

- **Python**: Data analysis and modeling
- **Pandas, NumPy**: Data manipulation and computation
- **Matplotlib, Seaborn**: Data visualization
- **Scikit-learn**: Clustering algorithms and preprocessing
- **PCA**: Dimensionality reduction for visualization

---

## 9. Business Impact / Use Case

This segmentation analysis equips mall marketers and management teams with concrete behavioral profiles for customer targeting. Understanding who shops where and how much they spend enables:

- Smarter promotional campaigns aligned with customer habits
- Optimized layout and inventory decisions based on cluster preferences
- Improved customer engagement by tailoring experiences to specific segments

The insights gained from this analysis support data-driven decision-making in customer retention, upselling, and operational strategy.
