import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

def missing_data_heatmap(df, describe_rows=False, filepath=None):
    """
    Creates a heatmap to visualize the pattern of missing data across the dataset.

    Parameters:
        df (pd.DataFrame): The data frame to visualize missing data for.
        describe_rows (bool): Whether to label each row individually. Default is False.
        filepath (str or None): If provided, the heatmap will be saved to this file. Default is None (plot is not saved).

    Returns:
        None: Displays or saves the heatmap.
    """
    sns.set(style="whitegrid")

    plt.figure(figsize=(12, 8))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis", yticklabels=df.index if describe_rows else False)

    plt.title("Missing Data Heatmap", fontsize=16)
    plt.xlabel("Columns", fontsize=14)
    plt.ylabel("Rows", fontsize=14)

    yellow_patch = mpatches.Patch(color='yellow', label='Missing')
    purple_patch = mpatches.Patch(color='purple', label='Not Missing')
    plt.legend(handles=[yellow_patch, purple_patch], loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
    
    plt.tight_layout()

    if filepath:
        plt.savefig(filepath, bbox_inches='tight')
        print(f"Heatmap saved to {filepath}")
    plt.show()