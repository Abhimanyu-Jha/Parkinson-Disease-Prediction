import pandas as pd
import numpy as np
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv('./dataset/pd_speech_features.csv')
id = df['id']
label = df['class']
n = len(id)
id_label = {}
for i in range(n):
    id_label[id[i]] = label[i]

train_healthy_id = []
test_healthy_id = []
train_pd_id = []
test_pd_id = []
train_n_healthy = 58
test_n_healthy = 6
train_n_pd = 170
test_n_pd = 18

train_count_healthy = 0
test_count_healthy = 0
train_count_pd = 0
test_count_pd = 0
for key, val in id_label.items():
    if val == 0:
        if train_count_healthy < train_n_healthy:
            train_healthy_id.append(key)
            train_count_healthy += 1
        else:
            test_healthy_id.append(key)
            train_count_healthy += 1
    else:
        if train_count_pd < train_n_pd:
            train_pd_id.append(key)
            train_count_pd += 1
        else:
            test_pd_id.append(key)
            test_count_pd += 1

train_healthy_id.extend(train_pd_id)
test_healthy_id.extend(test_pd_id)
train_id = train_healthy_id
test_id = test_healthy_id
train_id = np.array(train_id)
test_id = np.array(test_id)

trainX = None
trainY = None
for i in range(train_id.shape[0]):
    vals = df[df['id'] == train_id[i]].values
    if i == 0:
        trainX = vals[:, 1:-1]
        trainY = vals[:, -1].reshape((-1, 1))
    else:
        trainX = np.append(trainX, vals[:, 1:-1], axis=0)
        trainY = np.append(trainY, vals[:, -1].reshape((-1, 1)), axis=0)

testX = None
testY = None
for i in range(test_id.shape[0]):
    vals = df[df['id'] == test_id[i]].values
    if i == 0:
        testX = vals[:, 1:-1]
        testY = vals[:, -1].reshape((-1, 1))
    else:
        testX = np.append(testX, vals[:, 1:-1], axis=0)
        testY = np.append(testY, vals[:, -1].reshape((-1, 1)), axis=0)
trainY = trainY.reshape((-1,))

trainX_copy = trainX
trainY_copy = trainY
testX_copy = testX
testY_copy = testY

sel = SelectKBest(f_classif, k=490)
trainX_copy = sel.fit_transform(trainX_copy, trainY_copy)
testX_copy = sel.transform(testX_copy)

sc = StandardScaler()
trainX_copy = sc.fit_transform(trainX_copy)
testX_copy = sc.transform(testX_copy)

pca = PCA(n_components=0.99)
trainX_copy = pca.fit_transform(trainX_copy)
testX_copy = pca.transform(testX_copy)

if __name__ == "__main__":
    print("Data Pre-processed Succesfully. Run the script run_model.py")
else:
    X_train, X_test, y_test, y_train = trainX_copy, testX_copy, testY_copy, testY_copy
