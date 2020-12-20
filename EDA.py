from pre_processing import X_train, X_test, y_test, y_train
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# Perform EDA
df = pd.read_csv('./dataset/pd_speech_features.csv')

print('Dataset Shape:')
print(df.shape)
print()

print('Class Distribution:')
print(df['class'].value_counts())
print()


positives = df[df['class'] == 1]
negatives = df[df['class'] == 0]

print('Positive Class (Patients with PD) Gender Distribution')
print(positives['gender'].value_counts())
print()

print('Negative Class (Patients without PD) Gender Distribution')
print(negatives['gender'].value_counts())
print()

# Number of patients counts


def print_patient_dist(df):
    ids = np.unique(df['id'].values)
    fem = 0
    male = 0
    for i in ids:
        if df[df['id'] == i]['gender'].iloc[0] == 0:
            fem += 1
        else:
            male += 1
    print('Unique Males: ', male)
    print('Unique Females: ', fem)


print('Positive Class Unique Patients Distribution')
print_patient_dist(positives)
print()

print('Negative Class Unique Patients Distribution')
print_patient_dist(negatives)
print()

print('Database Info')
print(df.info())
print()


# Plot distribution of 3 important features
df_selected = df[['DFA', 'mean_MFCC_2nd_coef', 'tqwt_entropy_log_dec_12']]

print(df_selected.describe())

plt.gcf().set_size_inches(12, 5)
ax = sns.histplot(df['DFA'], color='#3EADA7', alpha=1, edgecolor='white')
ax.set_title('DCA')
plt.show()

plt.gcf().set_size_inches(12, 5)
ax = sns.histplot(df['mean_MFCC_2nd_coef'],
                  color='#3EADA7', alpha=1, edgecolor='white')
ax.set_title('mean_MFCC_2nd_coef')
plt.show()

plt.gcf().set_size_inches(12, 5)
ax = sns.histplot(df['tqwt_entropy_log_dec_12'],
                  color='#3EADA7', alpha=1, edgecolor='white')
ax.set_title('tqwt_entropy_log_dec_12')
plt.show()


# Peform tsne visualization on the preprocessed data
tsne_results = TSNE(n_components=2, random_state=1).fit_transform(X_train)

DF = pd.DataFrame({'y': y_train.reshape((-1,))})
DF['tsne1'] = tsne_results[:, 0]
DF['tsne2'] = tsne_results[:, 1]


plt.figure(figsize=(5, 5))
ax = sns.scatterplot(
    x="tsne1", y="tsne2",
    hue="y",
    palette=sns.color_palette("hls", 2),
    data=DF,
    legend=False,
    alpha=1
)

ax.set_title('TSNE Visualization')
plt.show()
