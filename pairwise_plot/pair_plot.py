import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def pair_plot(df: pd.DataFrame, exclude_categorical: bool = True) -> None:

    if exclude_categorical:
        data = df.copy(True).select_dtypes(include='number')
        sns.pairplot(data)
        plt.show()
    else:
        for category in df.select_dtypes(include=['object', 'category']).columns:
            sns.pairplot(df, hue=category)
            plt.show()
