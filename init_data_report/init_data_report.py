import os

import numpy as np
import scipy.stats as stats
import pandas as pd


def create_initial_data_report(data: pd.DataFrame, print_to_console: bool = True,
                               save_to_file: bool = None, filename: str = "initial_data_report") -> None:
    report_string = []
    report_string.append("----------INITIAL DATA REPORT----------\n")
    report_string.append(f"Number of rows: {data.shape[0]}\n")
    report_string.append(f"Number of columns: {data.shape[1]}\n")
    for column in data.columns:
        report_string.append(f"\"{column}\" column basic information\n\tColumn data type: {data[column].dtype}\n")
        report_string.append(f"\tNumber of non-null values {data[column].count()}\n")
        if pd.api.types.is_numeric_dtype(data[column].dtype):
            numeric_column_report(data,column,report_string)
        elif isinstance(data[column].dtype, pd.CategoricalDtype):
            categorical_column_report(data,column,report_string)

    if save_to_file:
        with open(f"{filename}.txt","w") as file:
            for string in report_string:
                file.write(string)

    if print_to_console:
        string_to_print = ""
        for string in report_string:
            string_to_print += string
        print(string_to_print)


def categorical_column_report(data: pd.DataFrame, column: str, report_string: list[str]) -> None:
    report_string.append(f"\tNumber of unique values: {data[column].nunique()}\n")
    modes = data[column].mode()
    if len(modes) > 1:
        report_string.append(f"\tModes: \n")
        for mode in modes:
            report_string.append(f"\t\t{mode}\n")
    else:
        report_string.append(f"\tMode: {modes[0]}\n")

    report_string.append(f"\tFrequency: {data[column].value_counts().iloc(0)}\n")

def numeric_column_report(data: pd.DataFrame,column: str,report_string: list[str]) -> None:
    percentiles = data[column].quantile([0.25,0.5,0.75])
    report_string.append(f"\tMean: {data[column].mean()}\n\tStandard Deviation: {data[column].std()}\n"
                      f"\tMedian: {data[column].median()}\n\t")
    modes = data[column].mode()
    if len(modes) > 1:
        report_string.append(f"\tModes: \n")
        for mode in modes:
            report_string.append(f"\t\t{mode}\n")
    else:
        report_string.append(f"\tMode: {modes[0]}\n")

    report_string.append(f"\tLowest value : {data[column].min()}\n")

    for quantile,value in percentiles.items():
        report_string.append(f"\tPercentile {quantile * 100}%: {value}\n")

    report_string.append(f"\tHighest value : {data[column].max()}\n")
