from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame


def pca_reduction(df: DataFrame, n_components: int = 2):
    """
       Performs Principal Component Analysis (PCA) to reduce the dimensionality of a dataset.

       Parameters:
           df (DataFrame): The input dataset as a pandas DataFrame, containing numerical features.
           n_components (int): The number of principal components to retain. Default is 2.

       Returns:
           ndarray: A NumPy array containing the dataset transformed to the reduced-dimensional space.
                    The shape of the output is (number of samples, n_components).
    """
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    pca = PCA(n_components=n_components)
    df_reduced = pca.fit_transform(df_scaled)

    return df_reduced
