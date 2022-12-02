from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

OUTPUT_COLUMN = "Attrition"
# COLUMNS = ["Attrition", "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "TotalWorkingYears"]
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

print('Dividir dataset en train, validation, test...')
x_train, x, y_train, y = train_test_split(
    df.drop(OUTPUT_COLUMN, axis=1),
    df[OUTPUT_COLUMN],
    test_size=0.2,
    random_state=10,
    stratify=df[OUTPUT_COLUMN]
)

x_test, x_cv, y_test, y_cv = train_test_split(
    x,
    y,
    test_size=0.5,
    random_state=10,
    stratify=y
)

# TODO: remove
print(x_test.values.shape)
print(y_test.values.shape)

# output = np.array([[x] for x in df.loc[:,OUTPUT_COLUMN].to_list()])
# df = df.drop(OUTPUT_COLUMN, axis=1)
# columns_in_order = np.delete(columns_in_order, np.where(columns_in_order == OUTPUT_COLUMN))
# input = df[columns_in_order].values

np.save(os.path.join('data', 'x_train.npy'), x_train)
np.save(os.path.join('data', 'x_cv.npy'), x_cv)
np.save(os.path.join('data', 'x_test.npy'), x_test)
np.save(os.path.join('data', 'y_train.npy'), y_train)
np.save(os.path.join('data', 'y_cv.npy'), y_cv)
np.save(os.path.join('data', 'y_test.npy'), y_test)
