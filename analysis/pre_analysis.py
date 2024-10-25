import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pandas as pd


def visualize_distribution(data, columns, bins=30, save_to_file=None):
    """
    Visualize the distribution of data for each specified column and check for normality.

    Parameters:
    - data (pd.DataFrame): Input data.
    - columns (list of str): List of column names to visualize.
    - bins (int): Number of bins for histogram.
    - save_to_file (str or None): File path to save the plots (if specified).
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")

    for column_name in columns:
        if column_name not in data.columns:
            print(f"Column '{column_name}' not found in DataFrame.")
            continue

        values = data[column_name].dropna()

        # Remove any NaN or inf values
        values = values[~np.isinf(values)]

        # Create figure and axes for visualizations
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        # Histogram and KDE
        sns.histplot(values, bins=bins, kde=True, ax=axs[0], color='skyblue')
        axs[0].set_title(f'Distribution of {column_name}')
        axs[0].set_xlabel(column_name)
        axs[0].set_ylabel('Frequency')

        # Q-Q Plot for Normality Check
        stats.probplot(values, dist="norm", plot=axs[1])
        axs[1].set_title('Q-Q Plot')

        # Normality Test (Shapiro-Wilk for N <= 5000, Anderson-Darling otherwise)
        if len(values) <= 5000:
            shapiro_test = stats.shapiro(values)
            p_value = shapiro_test.pvalue
            normal = "Yes" if p_value > 0.05 else "No"
            test_name = "Shapiro-Wilk"
        else:
            ad_test = stats.anderson(values, dist='norm')
            p_value = ad_test.significance_level[np.argmax(ad_test.statistic < ad_test.critical_values)]
            normal = "Yes" if ad_test.statistic < ad_test.critical_values[-1] else "No"
            test_name = "Anderson-Darling"

        # Boxplot for Outlier Visualization
        sns.boxplot(x=values, ax=axs[2], color='lightcoral')
        axs[2].set_title(f'Boxplot of {column_name}')

        # Display normality test result
        plt.suptitle(
            f'Distribution Analysis of {column_name}\nNormal Distribution: {normal} ({test_name} p = {p_value:.4f})')
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

        # Save or show plot
        if save_to_file:
            plt.savefig(f"{save_to_file}_{column_name}.png")
            print(f"Plot for '{column_name}' saved as {save_to_file}_{column_name}.png")
        else:
            plt.show()
        plt.close(fig)