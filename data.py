# data.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import random


def generate_exoplanet_data(num_planets=50, seed=6):
    """
    Generates a synthetic exoplanet dataset.

    Parameters:
    - num_planets (int): Number of exoplanets to generate.
    - seed (int): Random seed for reproducibility.

    Returns:
    - pd.DataFrame: DataFrame containing exoplanet data.
    """
    random.seed(seed)
    np.random.seed(seed)

    data = {
        'planet_name': [f'Planet_{i}' for i in range(1, num_planets + 1)],
        'mass_Earth_masses': np.random.uniform(0.1, 10, num_planets),  # Mass in Earth masses
        'radius_Earth_radii': np.random.uniform(0.5, 2.5, num_planets),  # Radius in Earth radii
        'orbital_period_days': np.random.uniform(10, 500, num_planets),  # Orbital period in days
        'temperature_K': np.random.uniform(300, 2500, num_planets),  # Equilibrium temperature in Kelvin
        'semi_major_axis_AU': np.random.uniform(0.01, 10, num_planets),  # Semi-major axis in AU
        'eccentricity': np.random.uniform(0, 0.6, num_planets)  # Orbital eccentricity
    }
    df = pd.DataFrame(data)

    # Feature Scaling using StandardScaler
    features = ['mass_Earth_masses', 'radius_Earth_radii', 'orbital_period_days', 'temperature_K']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])
    df_scaled = pd.DataFrame(scaled_features, columns=features)

    # K-Means Clustering with 3 clusters
    k = 3
    kmeans = KMeans(n_clusters=k, random_state=seed)
    kmeans.fit(df_scaled)
    df['cluster'] = kmeans.labels_

    # Principal Component Analysis (PCA) for Dimensionality Reduction to 2 components
    pca = PCA(n_components=2, random_state=seed)
    pca_features = pca.fit_transform(df_scaled)
    df_pca = pd.DataFrame(pca_features, columns=['PCA1', 'PCA2'])
    df = pd.concat([df, df_pca], axis=1)

    return df


def generate_wine_data(num_wines=50, seed=6):
    """
    Generates a synthetic wine dataset.

    Parameters:
    - num_wines (int): Number of wines to generate.
    - seed (int): Random seed for reproducibility.

    Returns:
    - pd.DataFrame: DataFrame containing wine data.
    """
    random.seed(seed)
    np.random.seed(seed)

    data = {
        'wine_name': [f'Wine_{i}' for i in range(1, num_wines + 1)],
        'alcohol': np.random.normal(loc=13.0, scale=0.8, size=num_wines),  # % by volume
        'malic_acid': np.random.normal(loc=2.0, scale=1.0, size=num_wines),  # g/L
        'ash': np.random.normal(loc=2.0, scale=0.3, size=num_wines),  # g/L
        'alcalinity_of_ash': np.random.normal(loc=20.0, scale=5.0, size=num_wines),  # mg/L
        'magnesium': np.random.normal(loc=100.0, scale=15.0, size=num_wines),  # mg/L
        'total_phenols': np.random.normal(loc=2.5, scale=0.5, size=num_wines),  # g/L
        'flavanoids': np.random.normal(loc=2.0, scale=0.7, size=num_wines),  # g/L
        'nonflavanoid_phenols': np.random.normal(loc=0.3, scale=0.1, size=num_wines),  # g/L
        'proanthocyanins': np.random.normal(loc=1.5, scale=0.3, size=num_wines),  # g/L
        'color_intensity': np.random.normal(loc=5.0, scale=1.5, size=num_wines),  # Intensity
        'hue': np.random.normal(loc=1.0, scale=0.2, size=num_wines),  # Hue
        'od280/od315': np.random.normal(loc=3.0, scale=0.5, size=num_wines),  # Optical density ratio
        'proline': np.random.normal(loc=500.0, scale=100.0, size=num_wines)  # mg/L
    }

    df_wine = pd.DataFrame(data)

    # Clip values to realistic ranges to avoid negative or unrealistic values
    df_wine['alcohol'] = df_wine['alcohol'].clip(lower=10.0, upper=15.0)
    df_wine['malic_acid'] = df_wine['malic_acid'].clip(lower=0.5, upper=5.0)
    df_wine['ash'] = df_wine['ash'].clip(lower=1.0, upper=4.0)
    df_wine['alcalinity_of_ash'] = df_wine['alcalinity_of_ash'].clip(lower=10.0, upper=30.0)
    df_wine['magnesium'] = df_wine['magnesium'].clip(lower=50.0, upper=170.0)
    df_wine['total_phenols'] = df_wine['total_phenols'].clip(lower=0.8, upper=4.0)
    df_wine['flavanoids'] = df_wine['flavanoids'].clip(lower=0.4, upper=4.0)
    df_wine['nonflavanoid_phenols'] = df_wine['nonflavanoid_phenols'].clip(lower=0.1, upper=1.0)
    df_wine['proanthocyanins'] = df_wine['proanthocyanins'].clip(lower=0.3, upper=3.0)
    df_wine['color_intensity'] = df_wine['color_intensity'].clip(lower=1.0, upper=13.0)
    df_wine['hue'] = df_wine['hue'].clip(lower=0.0, upper=2.0)
    df_wine['od280/od315'] = df_wine['od280/od315'].clip(lower=0.0, upper=4.0)
    df_wine['proline'] = df_wine['proline'].clip(lower=100.0, upper=2000.0)

    # Feature Scaling using StandardScaler
    features = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
                'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
                'proanthocyanins', 'color_intensity', 'hue', 'od280/od315', 'proline']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_wine[features])
    df_scaled = pd.DataFrame(scaled_features, columns=features)

    # K-Means Clustering with 3 clusters (e.g., representing different wine cultivars)
    k = 3
    kmeans = KMeans(n_clusters=k, random_state=seed)
    kmeans.fit(df_scaled)
    df_wine['cluster'] = kmeans.labels_

    # Principal Component Analysis (PCA) for Dimensionality Reduction to 2 components
    pca = PCA(n_components=2, random_state=seed)
    pca_features = pca.fit_transform(df_scaled)
    df_pca = pd.DataFrame(pca_features, columns=['PCA1', 'PCA2'])
    df_wine = pd.concat([df_wine, df_pca], axis=1)

    return df_wine
