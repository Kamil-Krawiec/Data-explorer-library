import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub

def feature_distribution_by_target(df, feature, target, directory=None):
  """Plots the distribution of a feature segmented by each category of the target variable.

  Args:
    df: The DataFrame containing the data.
    feature: The name of the feature to be plotted.
    target: The name of the target variable.
  """
  if not isinstance(df, pd.DataFrame):
    raise TypeError("Input data must be a pandas DataFrame.")
  columns = df.columns.tolist()
  if feature not in columns:
    raise TypeError("Selected feature does not exists.")
  if target not in columns:
    raise TypeError("Selected target does not exists.")

  sns.displot(data=df, x=feature, hue=target, kind='kde', fill=True)
  plt.title(f'Distribution of {feature} by {target} categories')
  plt.xlabel(feature)
  plt.ylabel("Probability Density")

  if directory:
    plt.savefig(f"{directory}/{feature}_by_{target}.png", bbox_inches='tight')
  else:
    plt.show()
  plt.close()

def save_all_distributions(df, directory):
    """Plots the distribution of each feature by each target.

    Args:
        df: The DataFrame containing the data.
    """
    for feature in df.select_dtypes(include=['number']):
        for target in df.select_dtypes(exclude=['number']):
            if target != feature:
                feature_distribution_by_target(df, feature, target, directory)

##### TESTS ######

path = kagglehub.dataset_download("lainguyn123/student-performance-factors")
df = pd.read_csv(f"{path}/StudentPerformanceFactors.csv")

# data = {'income': [30000, 45000, 60000, 52000, 75000, 28000, 42000, 80000, 38000, 50000],
#         'age': [25, 32, 40, 35, 48, 22, 38, 55, 30, 42],
#         'job_type': ['Blue Collar', 'White Collar', 'White Collar', 'Blue Collar', 'Management', 'Blue Collar', 'White Collar', 'Management', 'Blue Collar', 'White Collar']}
# df = pd.DataFrame(data)

save_all_distributions(df, 'results')

####################
