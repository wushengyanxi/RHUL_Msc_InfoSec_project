import pandas as pd
import os


def read_csv_file(file_path):
    """
    read csv file and return DataFrame
    """
    df = pd.read_csv(file_path)
    return df


def split_and_save_csv(df, output_dir):
    """
    Splits the DataFrame based on the 'traffic_category' feature and saves each subset to a separate CSV file.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data to be split.
        output_dir (str): The directory where the CSV files will be saved.

    """
    categories = df['traffic_category'].unique()
    for category in categories:
        category_df = df[df['traffic_category'] == category]
        output_file_path = os.path.join(output_dir, f"{category}_set.csv")
        category_df.to_csv(output_file_path, index=False)
        print(
            f"The number of samples which have value \"{category}\" on feature \"traffic_category\" is {len(category_df)}")


def main():
    input_file_path = (r"C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\DataSet_Preprocesser\ALLFLOWMETER_HIKARI2021"
                       r".csv")
    output_dir = r"C:\Users\wushe\Desktop\RHUL_Msc_InfoSec_project\DataSet_Preprocesser\Sub_database"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = read_csv_file(input_file_path)
    split_and_save_csv(df, output_dir)


if __name__ == "__main__":
    main()



