import pandas as pd

from pairwise_plot.pair_plot import pair_plot

data = pd.DataFrame({
    'Department': ['Sales', 'Engineering', 'HR', 'Sales', 'Engineering', 'HR', 'Sales', 'HR', 'Engineering', 'Sales'],
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Female'],
    'Age': [29, 34, 40, 24, 30, 32, 28, 45, 38, 27],
    'YearsAtCompany': [3, 5, 7, 2, 4, 6, 3, 10, 8, 1],
    'Salary': [50000, 60000, 55000, 45000, 70000, 52000, 49000, 75000, 67000, 48000]
})

pair_plot(data, False)