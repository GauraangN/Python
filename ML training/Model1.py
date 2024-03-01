import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report

# Define column(Which is a list) names for the dataset
cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3trans", "fAlpha", "fDist", "class"]

# Read the dataset from a CSV file and assign column names
df = pd.read_csv(r"ML training\magic04.data", names=cols)

# Convert the "class" column to binary (0 or 1), where 'g' is 1 and 'h' is 0
df["class"] = (df["class"] == 'g').astype(int)
print(df.head())

# Visualize histograms for each feature based on the target class
for label in cols[:-1]:
    plt.hist(df[df["class"] == 1][label], color="blue", label="gamma", alpha=0.7, density=True)
    plt.hist(df[df["class"] == 0][label], color="red", label="hadron", alpha=0.7, density=True)
    plt.title(label)
    plt.ylabel("probability")
    plt.xlabel(label)
    plt.legend()
    #plt.show()

# Split the dataset into training, validation, and test sets
train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

# Define a function to scale the dataset and perform oversampling if specified
def scale_dataset(dataframe, oversample=False):
    X = dataframe[dataframe.columns[:-1]].values
    Y = dataframe[dataframe.columns[-1]].values

    # Standardize the feature values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Optionally perform oversampling using RandomOverSampler
    if oversample:
        ros = RandomOverSampler()
        X, Y = ros.fit_resample(X, Y)

    # Combine the scaled features and target variable into a single array
    data = np.hstack((X, np.reshape(Y, (len(Y), 1))))

    return data, X, Y

# Scale and oversample the training, validation, and test sets
train, x_train, y_train = scale_dataset(train, oversample=True)
valid, x_valid, y_valid = scale_dataset(valid, oversample=False)
test, x_test, y_test = scale_dataset(test, oversample=False)

# K-Nearest Neighbors (KNN) Model
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=1)

# Train the KNN model on the training set
knn_model.fit(x_train, y_train)

# Make predictions on the test set
y_pred = knn_model.predict(x_test)

# Print predicted and actual labels for the test set
print(y_pred)
print(y_test)

print(classification_report(y_test, y_pred))