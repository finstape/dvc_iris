import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

def prepare_data():
    iris = load_iris()
    data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    data['target'] = iris.target

    # Split data into train and test sets
    train, test = train_test_split(data, test_size=0.2, random_state=42)

    # Save splits
    train.to_csv('data/train.csv', index=False)
    test.to_csv('data/test.csv', index=False)

if __name__ == "__main__":
    prepare_data()
