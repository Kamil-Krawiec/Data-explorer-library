from init_data_report import create_initial_data_report
import pandas as pd
import numpy as np


if __name__ == '__main__':
    data = {
        'Num1': np.random.randint(1,100,6),  # Numerical column 1
        'Num2': np.random.random(6) * 100,  # Numerical column 2
        'Cat1': ['A','B','A','C','B','C'],  # Categorical column 1
        'Cat2': ['X','Y','Y','X','Z','X'],  # Categorical column 2
        'Mix1': ['Cat1',20,'Cat2',30.5,10,'Cat3'],  # Mixed column 1
        'Mix2': [np.nan,45,'Hello','World',3.14,'Bye']  # Mixed column 2
    }
    df = pd.DataFrame(data)
    df['Cat1'] = df['Cat1'].astype('category')
    df['Cat2'] = df['Cat2'].astype('category')
    create_initial_data_report(df, save_to_file=True)