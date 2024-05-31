import pandas as pd
import numpy as np


def Database_Reader(DataBase_Name, full_read=True, feature_list=None, traffic_category=True):
    """
    Reads data from a CSV file and returns selected features and samples.

    This function reads the specified CSV file and can operate in two modes:
    'full_read' to read all features and 'part_read' to read only specified features.
    It also includes an option to read either 'traffic_category' or 'Label', but not both simultaneously.

    Args:
        DataBase_Name (str): The name of the CSV file to read.
        full_read (bool, optional):
            If True, reads all features.
            If False, reads only specified features. Defaults to True.
        feature_list (list of str, optional): A list of feature names to read in 'part_read' mode.
            Required if full_read is False.
        traffic_category (bool, optional): If True, reads 'traffic_category' column.
            If False, reads 'Label' column. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - Feature_List (list of str): A list of feature names that were read.
            - Data (list of lists): A 2D list where each sublist represents a sample with its feature values.

    Raises:
        ValueError: If both 'traffic_category' and 'Label' are requested,
        or if 'feature_list' is not provided in 'part_read' mode.
    """

    file_path = r"C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\ALLFLOWMETER_HIKARI2021.csv"
    df = pd.read_csv(file_path)

    # 定义traffic_category的映射
    traffic_category_map = {
        "Benign": 0,
        "Background": 1,
        "Bruteforce-XML": 2,
        "Bruteforce": 3,
        "Probing": 4,
        "XMRIGCC CryptoMiner": 5
    }

    # 如果是full_read模式
    if full_read:
        if traffic_category:
            features = [col for col in df.columns if col != 'Label']
            df['traffic_category'] = df['traffic_category'].map(traffic_category_map)
            df_selected = df[features]
            Feature_List = df_selected.columns.tolist()
            Data = df_selected.to_numpy().tolist()
        else:
            features = [col for col in df.columns if col != 'traffic_category']
            df_selected = df[features]
            Feature_List = df_selected.columns.tolist()
            Data = df_selected.to_numpy().tolist()
    else:
        if feature_list is None:
            raise ValueError("feature_list must be provided for part_read mode.")

        if traffic_category:
            features = feature_list + ['traffic_category']
            df['traffic_category'] = df['traffic_category'].map(traffic_category_map)
        else:
            features = feature_list + ['Label']

        df_selected = df[features]
        Feature_List = df_selected.columns.tolist()
        Data = df_selected.to_numpy().tolist()

    return Feature_List, Data


