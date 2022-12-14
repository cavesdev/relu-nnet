from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

OUTPUT_COLUMN = "Attrition"
# COLUMNS = ["Attrition", "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "TotalWorkingYears", "Age", "YearsWithCurrManager", "HourlyRate", "JobLevel"]
# data = pd.read_csv('data.csv', usecols=COLUMNS)
data = pd.read_csv('data.csv')
columns_in_order = data.columns.values

# print('Discretizar valores de output a 0 - 1...')
# data[OUTPUT_COLUMN] = pd.cut(data[OUTPUT_COLUMN], bins=2, labels=False)
#
# data[OUTPUT_COLUMN].value_counts().plot(kind='bar', title="WorkLifeBalance counts")
# plt.show()

print('Convertir los strings a valores numéricos...')
for column in data.columns:
    if data[column].dtype == 'O':
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])

print('Undersampling para balancear los outputs...')
count_class_0, count_class_1 = data[OUTPUT_COLUMN].value_counts(sort=False)
classes = data[OUTPUT_COLUMN].unique()

df_class_0 = data[data[OUTPUT_COLUMN] == classes[0]]
df_class_1 = data[data[OUTPUT_COLUMN] == classes[1]]

df_class_1_under = df_class_1.sample(count_class_0)
df_test_under = pd.concat([df_class_1_under, df_class_0], axis=0)

df_test_under[OUTPUT_COLUMN].value_counts().plot(kind='bar', title='Count (target)');
plt.show()

data = df_test_under

print('Estandarizar los datos...')
data_scaled = MinMaxScaler().fit_transform(data)

df = pd.DataFrame(data_scaled, columns=columns_in_order)

output = df[OUTPUT_COLUMN]
df = df.drop(OUTPUT_COLUMN, axis=1)

pca = PCA(n_components=10)
df = pca.fit_transform(df)

print('Dividir dataset en train, validation, test...')
x_train, x, y_train, y = train_test_split(
    df,
    output,
    test_size=0.2,
    random_state=10,
    stratify=output
)

x_test, x_cv, y_test, y_cv = train_test_split(
    x,
    y,
    test_size=0.5,
    random_state=10,
    stratify=y
)

np.save(os.path.join('data', 'x_train.npy'), x_train)
np.save(os.path.join('data', 'x_cv.npy'), x_cv)
np.save(os.path.join('data', 'x_test.npy'), x_test)
np.save(os.path.join('data', 'y_train.npy'), y_train)
np.save(os.path.join('data', 'y_cv.npy'), y_cv)
np.save(os.path.join('data', 'y_test.npy'), y_test)
