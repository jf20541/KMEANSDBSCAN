import pandas as pd
import config
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import data


if __name__ == "__main__":
    df = pd.read_csv(config.TRAINING_FILE)
    df = data.clean_data(df)
    df.columns = data.clean_colname(df)
    df.to_csv(config.TESTING_FILE, index_label="State_City")
