from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from pre_processing import X_train, X_test, y_test, y_train
from sklearn.neural_network import MLPClassifier


# best_params file returns the best_models tuned to best performing parameters
best_models = {1: "naive_bayes", 2: "logistic_regression", 3: "decision_tree",
                  4: "svm", 5: "random_forest", 6: "ada_boost", 7: "gradient_boost", 8: "xg_boost", 9: "neural_net"}

# Naive Bayes
clf = BernoulliNB()
clf.fit(X_train, y_train)
best_models[1] = clf

# Logistic Regression
param_grid = {'penalty': ['l1', 'l2'],
              'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
grid = GridSearchCV(LogisticRegression(random_state=0),
                    param_grid, refit=True, verbose=3)
grid.fit(X_train, y_train)
best_models[2] = grid

# decision_tree
param_grid = {'criterion': ['gini', "entropy"],
              'splitter': ["best", "random"],
              'max_depth': range(10, 50, 5),
              'min_samples_split': range(2, 10, 1)}
grid = GridSearchCV(DecisionTreeClassifier(random_state=0),
                    param_grid, refit=True, verbose=3)
grid.fit(X_train, y_train)
best_models[3] = grid

# SVM
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001],
              'kernel': ['rbf', 'poly', 'sigmoid']
              }
grid = GridSearchCV(SVC(random_state=0), param_grid, refit=True, verbose=3)
grid.fit(X_train, y_train)
best_models[4] = grid

# Random Forest
param_grid = {'n_estimators': range(10, 100, 5),
              'criterion': ['gini', 'entropy'],
              'max_depth': range(10, 50, 5),
              'min_samples_split': range(2, 10, 1),
              'max_features': ['auto', 'sqrt', 'log2']
              }
grid = GridSearchCV(RandomForestClassifier(random_state=0, n_jobs=10),
                    param_grid, refit=True, verbose=3, n_jobs=10)
grid.fit(X_train, y_train)
best_models[5] = grid

# ADA Boost
param_grid = {'n_estimators': range(10, 100, 5),
              'algorithm': ['SAMME', 'SAMME.R'],
              'learning_rate': [0.001, 0.01, 0.1, 1.0, 10.0]
              }
grid = GridSearchCV(AdaBoostClassifier(random_state=0),
                    param_grid, refit=True, verbose=3)
grid.fit(X_train, y_train)
best_models[6] = grid

# Gradient Boost
param_grid = {'n_estimators': [100, 150, 200, 250, 300, 350, 400, 450, 500],
              'criterion': ['friedman_mse', 'mse'],
              'max_features': ['sqrt', 'log2'],
              'learning_rate': [0.001, 0.01, 0.1, 1.0, 2, 4],
              'loss': ['deviance', 'exponential'],
              }
grid = GridSearchCV(GradientBoostingClassifier(
    random_state=0), param_grid, refit=True, verbose=3)
grid.fit(X_train, y_train)
best_models[7] = grid

# XG Boost
param_grid = {'n_estimators': range(0, 100, 5),
              'learning_rate': [0.01, 0.1],
              'max_depth': range(2, 6)
              }
grid = GridSearchCV(XGBClassifier(random_state=0),
                    param_grid, refit=True, verbose=3)
grid.fit(X_train, y_train)
best_models[8] = grid

# Neural Net
param_grid = {'alpha': [x * 0.0001 for x in range(1, 1000, 2)],
              'learning_rate_init': [0.01, 0.001, 0.0001],
              'max_iter': [100, 300, 1000],
              'hidden_layer_sizes': [(128, 64, 32), (128, 32, 8), (256, 128, 64, 32), (64, 32, 16), (64, 32), (64, 16)],
              }
grid = GridSearchCV(MLPClassifier(random_state=42),
                    param_grid, refit=True, verbose=3, n_jobs=10)
grid.fit(X_train, y_train)
best_models[9] = grid
