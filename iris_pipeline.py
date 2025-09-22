# Required Packages
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score


# Load Data
def load_dataset():
    X, y = load_iris(return_X_y=True)
    return X,y

# Split Data to train (80%) and test (20%)
def split_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test

# Build The Model and Train Model on train data
def train(X_train, y_train):
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return model

# Evaluate Model on test data
def get_accuracy(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def plot_model(model, X_test, y_test):
    # Plot the confusion matrix for the model
    ConfusionMatrixDisplay.from_estimator(estimator=model, X=X_test, y=y_test)
    plt.title("Confusion Matrix")
    plt.show()

def plot_iris_data(iris):
    _, ax = plt.subplots()
    scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
    ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
    _ = ax.legend(
        scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
    )


if __name__ == "__main__":

    X, y = load_dataset()

    X_train, X_test, y_train, y_test = split_data(X,y)
    model = train(X_train, y_train)
    accuracy = get_accuracy(model, X_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")

    plot_iris_data(load_iris())
    plot_model(model, X_test, y_test)