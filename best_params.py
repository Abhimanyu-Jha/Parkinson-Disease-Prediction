from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from pre_processing import X_train, X_test, y_test, y_train


# best_params file returns the best_models tuned to best performing parameters
best_models = {1: "naive_bayes", 2: "logistic_regression", 3: "decision_tree",
                  4: "svm", 5: "random_forest", 6: "ada_boost", 7: "gradient_boost", 8: "xg_boost", 9: "neural_net"}

# Naive Bayes
clf = BernoulliNB()
clf.fit(X_train, y_train)
best_models[1] = clf

# Logistic Regression


# Random Forest
param_grid = {'n_estimators': range(10, 100, 5),
              'criterion': ['gini', 'entropy'],
              'max_depth': range(10, 50, 5),
              'min_samples_split': range(2, 10, 1),
              'max_features': ['auto', 'sqrt', 'log2']
              }
grid = GridSearchCV(RandomForestClassifier(n_jobs=10),
                    param_grid, refit=True, verbose=3, n_jobs=10)
grid.fit(X_train, y_train)
best_models[5] = grid
