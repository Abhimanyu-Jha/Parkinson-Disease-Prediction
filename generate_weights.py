import argparse
import pickle
import sys
from best_params import best_models

if __name__ == "__main__":
    # Adding Arguments to make user select one model to run results on
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
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

    model_dict = {1: "naive_bayes", 2: "logistic_regression", 3: "decision_tree",
                  4: "svm", 5: "random_forest", 6: "ada_boost", 7: "gradient_boost", 8: "xg_boost", 9: "neural_net"}
    # save the model to disk
    model = best_models[args.model]
    filename = f"./weights/{model_dict[args.model]}.sav"
    print(f"\nSaving {filename} ...")
    pickle.dump(model, open(filename, 'wb'))
else:
    print("Run the script run_model.py to obtain results.")
