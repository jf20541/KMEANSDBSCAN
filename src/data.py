import pandas as pd
import config


def clean_data(dataframe):
    # define the index to state_city
    return dataframe.rename(columns={"Unnamed: 0": "State_City"}).set_index(
        "State_City"
    )


def clean_colname(dataframe):
    # clean columns, set them to lower case, no spaces, no dashes
    return [
        x.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(r"/", "_")
        .replace("\\", "_")
        .replace(".", "_")
        .replace("$", "")
        .replace("%", "")
        for x in dataframe.columns
    ]


if __name__ == "__main__":
    df = pd.read_csv(config.TRAINING_FILE)
    df_col = clean_colname(df)[1:]
    df = clean_data(df)
    df.columns = df_col
    if df.isnull().sum().any() == False:
        print("Data is Clean")
    else:
        print("Null Values Found, Clean Data")
