import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from time import time
import json


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
    return {'train_score': acc_train, 'test_score': acc_test, 'train_time': run_time}


def plot_graph(data):
    df = pd.DataFrame(data)
    x = np.arange(df.shape[0])
    fig, ax = plt.subplots()
    ax.bar(x - 0.2, height=df.loc['test_score'], width=0.4, label='accuracy')
    ax.bar(x + 0.2, height=df.loc['train_time'] / df.loc['train_time'].max(), width=0.4, label='train time')
    fig.suptitle('relative train time and accuracy of different models')
    ax.set_ylabel('accuracy score')
    ax.set_xticks(x, [name.split('(')[0] for name in df.columns])
    ax.legend()
    plt.show()


def main():
    # prepare datasets
    train_set = pd.read_csv('data/fashion-mnist_train.csv')
    test_set = pd.read_csv('data/fashion-mnist_test.csv')
    x_train = scale(train_set.drop('label', axis=1))
    y_train = train_set['label']
    x_test = scale(test_set.drop('label', axis=1))
    y_test = test_set['label']

    # parameter space for grid search
    knn_params = {'n_neighbors': [3, 5, 51], 'algorithm': ['auto', 'brute']}
    tree_params = {'criterion': ['gini', 'entropy'], 'max_depth': [50, 180, None]}

    # include grid search in model comparison
    use_grid = False

    # set grid search for models
    knn_grid = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=knn_params, scoring='accuracy', n_jobs=-1)
    tree_grid = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=tree_params, scoring='accuracy', n_jobs=-1)

    # models to compare
    models = [MLPClassifier(hidden_layer_sizes=(60, 80, 60), solver='adam', alpha=0.01, max_iter=5),
              KNeighborsClassifier(n_neighbors=3),
              DecisionTreeClassifier(criterion='gini', max_depth=80)]
    if use_grid:
        models.extend([knn_grid, tree_grid])

    # train models, check accuracy and training time
    scores = {}
    for model in models:
        scores[str(model)] = fit_predict_score(model, x_train, y_train, x_test, y_test,
                                               verbose=True, showtime=True)

    # print and save results to file
    print(scores)
    if use_grid:
        print('Best grid search models:', knn_grid.best_estimator_, tree_grid.best_estimator_, sep='\n')
    save_results(scores, append=True)
    plot_graph(scores)


if __name__ == '__main__':
    main()
