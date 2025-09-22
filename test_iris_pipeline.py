from iris_pipeline import load_dataset,split_data, train, get_accuracy

def test_model_accuracy():
    X, y = load_dataset()

    X_train, X_test, y_train, y_test = split_data(X,y)
    model = train(X_train, y_train)
    accuracy = get_accuracy(model, X_test, y_test)

    assert accuracy > 0.8, "Model accuracy is below 80%."