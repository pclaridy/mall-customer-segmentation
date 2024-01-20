# Customer Mall Segmentation

This project focuses on customer mall data segmentation, leveraging advanced data analytics techniques to uncover meaningful patterns and insights within the dataset. By segmenting mall customers based on their behavior and characteristics, this project aims to provide valuable information that can guide marketing strategies, enhance customer experiences, and optimize business operations for malls and retail establishments.

## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Clustering](#clustering)
  - [K-Means Clustering](#k-means-clustering-pca-reduced-data)
  - [DBSCAN Clustering](#dbscan-clustering-pca-reduced-data)
  - [Agglomerative Clustering](#agglomerative-clustering-pca-reduced-data)
  - [Spectral Clustering](#spectral-clustering-pca-reduced-data)
  - [Gaussian Mixture Model Clustering](#gaussian-mixture-model-clustering-pca-reduced-data)
- [Conclusion](#conclusion)

## Installation

To run this project locally, you can follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/pclaridy/mall-customer-segmentation/blob/main.git
   cd mall-customer-segmentation
2. Install the required dependencies by   running:
pip install -r requirements.txt
3. Run the project by executing the provided code in your preferred Python environment.

## Exploratory Data Analysis

The dataset consists of customer information, including age, gender, annual income, and spending score. Basic statistics and visualizations were performed to understand the distribution of data and relationships between variables.

## Clustering

### K-Means Clustering (PCA-reduced data)

K-Means clustering reveals five distinct customer segments, with clear differentiation in spending habits and income levels.

![K-Means Clustering](https://github.com/pclaridy/mall-customer-segmentation/blob/main/KMeans%20Clustering.png)


### DBSCAN Clustering (PCA-reduced data)

DBSCAN identifies a primary cluster with several outliers, suggesting a majority segment with similar characteristics and a few anomalies.

![DBSCAN Clustering](https://github.com/pclaridy/mall-customer-segmentation/blob/main/DBSCAN%20Clustering.png)

### Agglomerative Clustering (PCA-reduced data)

Agglomerative Clustering indicates five customer groups with varying density and connection to each other within the PCA-reduced feature space.

![Agglomerative Clustering](https://github.com/pclaridy/mall-customer-segmentation/blob/main/Agglomerative%20Clustering.png)

### Spectral Clustering (PCA-reduced data)

Spectral Clustering captures complex, non-linear relationships, dividing customers into five well-defined yet subtly overlapping segments.

![Spectral Clustering](https://github.com/pclaridy/mall-customer-segmentation/blob/main/Spectral%20Clustering.png)

### Gaussian Mixture Model Clustering (PCA-reduced data)

The Gaussian Mixture Model suggests the existence of subgroups that might correspond to different types of customer behavior or profiles.

![Gaussian Mixture Model Clustering](https://github.com/pclaridy/mall-customer-segmentation/blob/main/GMM%20Clustering.png)

## Conclusion

In conclusion, this project successfully segmented mall customers into distinct groups using various clustering techniques. These segments can provide valuable insights for marketing and business strategies. Further analysis and interpretation of each cluster's characteristics can help optimize mall operations and enhance the shopping experience.

