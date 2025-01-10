import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import IsolationForest


def detect_outliers_3d(
        data,
        n_outliers=10,
        detection_features=None,
        visualization_features=None,
        contamination='auto',
        random_state=42
):
    """
    Detects outliers in multi-dimensional data and visualizes them in a 3D scatter plot based on specified features.

    Parameters:
    - data (pd.DataFrame or np.ndarray): The input data containing multiple features.
    - n_outliers (int): The number of outliers to detect. Ignored if contamination is set.
    - detection_features (list of str or int, optional): Features to use for outlier detection.
                                                         If None, all features are used.
    - visualization_features (list of str or int, optional): Exactly three features to use for 3D visualization.
                                                            If None, the first three features are used.
    - contamination (str or float): The proportion of outliers in the dataset.
                                    If 'auto', uses n_outliers to determine contamination.
    - random_state (int): Random state for reproducibility.

    Returns:
    - outliers (pd.DataFrame): DataFrame containing the detected outliers.
    - fig (matplotlib.figure.Figure): The active 3D scatter plot figure.
    """
    # Convert input data to DataFrame if it's not
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    elif not isinstance(data, pd.DataFrame):
        raise TypeError("Data should be a pandas DataFrame or a NumPy array.")

    # Handle detection features
    if detection_features is None:
        detection_features = data.columns.tolist()
    else:
        # Validate detection features
        for feature in detection_features:
            if feature not in data.columns and not (isinstance(feature, int) and feature < data.shape[1]):
                raise ValueError(f"Detection feature '{feature}' is not in the data.")

    # Handle visualization features
    if visualization_features is None:
        if data.shape[1] < 3:
            raise ValueError("Data must have at least three features for visualization.")
        visualization_features = detection_features[:3]
    else:
        if len(visualization_features) != 3:
            raise ValueError("Exactly three visualization features must be specified for 3D plotting.")
        # Validate visualization features
        for feature in visualization_features:
            if feature not in data.columns and not (isinstance(feature, int) and feature < data.shape[1]):
                raise ValueError(f"Visualization feature '{feature}' is not in the data.")

    # Extract data for detection
    X_detection = data[detection_features].values

    # Determine contamination parameter
    if contamination == 'auto':
        contamination_ratio = n_outliers / len(X_detection)
        contamination_ratio = min(max(contamination_ratio, 0.0001), 0.5)  # Ensure it's between 0.0001 and 0.5
    else:
        contamination_ratio = contamination

    # Initialize Isolation Forest
    clf = IsolationForest(contamination=contamination_ratio, random_state=random_state)
    clf.fit(X_detection)
    scores = clf.decision_function(X_detection)
    predictions = clf.predict(X_detection)  # -1 for outliers, 1 for inliers

    # Identify outliers
    outlier_indices = np.where(predictions == -1)[0]
    outliers = data.iloc[outlier_indices]

    # If 'auto' was used and more outliers than desired, select top n_outliers
    if contamination == 'auto' and len(outliers) > n_outliers:
        # Sort outliers by their anomaly score (lower scores are more anomalous)
        sorted_outliers = outliers.copy()
        sorted_outliers['anomaly_score'] = scores[outlier_indices]
        sorted_outliers = sorted_outliers.sort_values(by='anomaly_score')
        outliers = sorted_outliers.head(n_outliers).drop(columns=['anomaly_score'])

    # Plotting
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Extract visualization data
    X_vis = data[visualization_features].values

    # Plot all data points as inliers initially
    ax.scatter(X_vis[:, 0], X_vis[:, 1], X_vis[:, 2], c='blue', alpha=0.5, label='Inliers')

    # Plot outliers
    if not outliers.empty:
        outlier_X_vis = outliers[visualization_features].values
        ax.scatter(outlier_X_vis[:, 0], outlier_X_vis[:, 1], outlier_X_vis[:, 2],
                   c='red', s=50, label='Outliers')

    ax.set_xlabel(visualization_features[0])
    ax.set_ylabel(visualization_features[1])
    ax.set_zlabel(visualization_features[2])
    ax.legend()
    plt.title('3D Outlier Detection')
    plt.tight_layout()
    plt.show()

    return outliers.reset_index(drop=True), fig
