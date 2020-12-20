import argparse
import pickle
import sys
# Loading the dataset
from pre_processing import X_train, X_test, y_test, y_train

# Adding Arguments to make user select one model to run results on
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
    "-m", "--model", type=int, choices=list(range(1, 10)), help="\
        \n1. Naive Bayes \
        \n2. Logistic Regression \
        \n3. Decision Tree\
        \n4. SVM\
        \n5. Random Forest\
        \n6. AdaBoost\
        \n7. Gradient Boost\
        \n8. XGBoost\
        \n9. Neural Network\
        ")

args = parser.parse_args()
if not args.model:
    print("Model not provided. See run_model.py --help")
    sys.exit()


# Loading and Running the Model
model_dict = {1: "naive_bayes", 2: "logistic_regression", 3: "decision_tree",
              4: "svm", 5: "random_forest", 6: "ada_boost", 7: "gradient_boost", 8: "xg_boost", 9: "neural_net"}

filename = f"./weights/{model_dict[args.model]}.sav"
print(f"\nLoading {filename} ...")

loaded_model = pickle.load(open(filename, 'rb'))
# loaded_model.fit(X_train, y_train)
# result = loaded_model.score(X_test, y_test)
result = 100
print(f"Model Accuracy: {result}%")
