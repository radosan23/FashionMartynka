import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from time import time
import json
# todo: grid search, graph, save to file


def save_results(data, append=False):
    mode = 'at' if append else 'wt'
    with open('model_scores.txt', mode) as f:
        json.dump(data, f, indent=4)


def scale(data):
    return data / data.max()


def fit_predict_score(model, x_train, y_train, x_test, y_test, verbose=False, showtime=False):
    start = time()
    model.fit(x_train, y_train)
    acc_train = accuracy_score(y_train, model.predict(x_train))
    acc_test = accuracy_score(y_test, model.predict(x_test))
    run_time = time() - start
    if verbose:
        print(f'{model} train score: {acc_train}, test score: {acc_test}')
        if showtime:
            print(f'{model} train time: {run_time:.3f}')
    return acc_test


def main():
    # prepare datasets
    train_set = pd.read_csv('data/fashion-mnist_train.csv')
    test_set = pd.read_csv('data/fashion-mnist_test.csv')
    x_train = scale(train_set.drop('label', axis=1))
    y_train = train_set['label']
    x_test = scale(test_set.drop('label', axis=1))
    y_test = test_set['label']

    models = [MLPClassifier(hidden_layer_sizes=(60, 80, 60), solver='adam', alpha=0.001),
              KNeighborsClassifier(5),
              DecisionTreeClassifier()]
    scores = {}
    for model in models:
        scores[type(model).__name__] = fit_predict_score(model, x_train, y_train, x_test, y_test,
                                                         verbose=True, showtime=True)

    print(scores)
    save_results(scores, append=True)


if __name__ == '__main__':
    main()
