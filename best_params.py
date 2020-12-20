from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from pre_processing import X_train, X_test, y_test, y_train
from sklearn.neural_network import MLPClassifier


# best_params file returns the best_models tuned to best performing parameters
best_models = {1: "naive_bayes", 2: "logistic_regression", 3: "decision_tree",
                  4: "svm", 5: "random_forest", 6: "ada_boost", 7: "gradient_boost", 8: "xg_boost", 9: "neural_net"}


def best_model(model_num):
    # Naive Bayes
    if model_num == 1:
        clf = BernoulliNB()
        clf.fit(X_train, y_train)
        return clf

    # Logistic Regression
    if model_num == 2:
        param_grid = {'penalty': ['l1', 'l2'],
                      'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        grid = GridSearchCV(LogisticRegression(random_state=0),
                            param_grid, refit=True)
        grid.fit(X_train, y_train)
        return grid

    # decision_tree
    if model_num == 3:
        param_grid = {'criterion': ['gini', "entropy"],
                      'splitter': ["best", "random"],
                      'max_depth': range(1, 20),
                      'min_samples_split': range(2, 10, 1)}
        grid = GridSearchCV(DecisionTreeClassifier(random_state=0),
                            param_grid, refit=True)
        grid.fit(X_train, y_train)
        return grid

    # SVM
    if model_num == 4:
        param_grid = {'C': [0.1, 1, 10, 100, 1000],
                      'gamma': [1, 0.1, 0.01, 0.001],
                      'kernel': ['rbf', 'poly', 'sigmoid']
                      }
        grid = GridSearchCV(SVC(random_state=0), param_grid, refit=True)
        grid.fit(X_train, y_train)
        return grid

    # Random Forest
    if model_num == 5:
        param_grid = {'n_estimators': range(10, 100, 5),
                      'criterion': ['gini', 'entropy'],
                      'max_depth': range(1, 20),
                      'min_samples_split': range(2, 10, 1),
                      'max_features': ['auto', 'sqrt', 'log2']
                      }
        grid = GridSearchCV(RandomForestClassifier(random_state=0, n_jobs=10),
                            param_grid, refit=True, n_jobs=10)
        grid.fit(X_train, y_train)
        return grid

    # ADA Boost
    if model_num == 6:
        param_grid = {'n_estimators': range(10, 100, 5),
                      'algorithm': ['SAMME', 'SAMME.R'],
                      'learning_rate': [0.001, 0.01, 0.1, 1.0, 10.0]
                      }
        grid = GridSearchCV(AdaBoostClassifier(random_state=0),
                            param_grid, refit=True)
        grid.fit(X_train, y_train)
        return grid

    # Gradient Boost
    if model_num == 7:
        param_grid = {'n_estimators': [100, 150, 200, 250, 300, 350, 400, 450, 500],
                      'criterion': ['friedman_mse', 'mse'],
                      'max_features': ['sqrt', 'log2'],
                      'learning_rate': [0.001, 0.01, 0.1, 1.0, 2, 4],
                      'loss': ['deviance', 'exponential'],
                      }
        grid = GridSearchCV(GradientBoostingClassifier(
            random_state=0), param_grid, refit=True)
        grid.fit(X_train, y_train)
        return grid

    # XG Boost
    if model_num == 8:
        param_grid = {'n_estimators': range(0, 100, 5),
                      'learning_rate': [0.01, 0.1],
                      'max_depth': range(2, 6)
                      }
        grid = GridSearchCV(XGBClassifier(random_state=0),
                            param_grid, refit=True)
        grid.fit(X_train, y_train)
        return grid

    # Neural Net
    if model_num == 9:
        param_grid = {'alpha': [x * 0.0001 for x in range(1, 1000, 2)],
                      'learning_rate_init': [0.01, 0.001, 0.0001],
                      'max_iter': [100, 300, 1000],
                      'hidden_layer_sizes': [(128, 64, 32), (128, 32, 8), (256, 128, 64, 32), (64, 32, 16), (64, 32), (64, 16)],
                      }
        grid = GridSearchCV(MLPClassifier(random_state=42),
                            param_grid, refit=True, n_jobs=10)
        grid.fit(X_train, y_train)
        return grid
