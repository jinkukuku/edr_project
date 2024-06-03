import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 14, 8

# RANDOM_SEED와 LABELS 설정
RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]

df = pd.read_csv("/content/drive/MyDrive/hw_1500.csv")

# 데이터 초기 세팅
df = pd.read_csv("/content/drive/MyDrive/hw_1500.csv")
df.shape
df_dropna = df.dropna()
df = df_dropna
df['parent_elevated'] = df['parent_elevated'].replace({'False': 0, 'True': 1})
df = df.drop(df.columns[0:6], axis=1)
df = df.drop(['flag'], axis=1)

df['label_int'] = df['add_label'].apply(lambda x: 0 if x == "normal" else 1)

import re

def preprocess(text):
    text = re.sub(r'[^\w\s] ', '', text)  # 특수 문자 제거
    text = text.lower()  # 소문자로 변환
    return text

# parent_image_path 열에 대해 토큰화를 수행하는 함수
def tokenize(text):
    tokens = text.split("\\")
    # 리스트의 두 번째 요소와 마지막 요소만 추출하여 반환
    return [tokens[1], tokens[-1]]

# parent_image_path 열에 대해 전처리 및 토큰화 수행
df['parent_image_path'] = df['parent_image_path'].apply(preprocess)

# 새로운 열을 추가하여 토큰화된 결과를 저장
df[['second_path', 'last_path']] = df['parent_image_path'].apply(tokenize).apply(pd.Series)
df.shape

df['parent_image_path']

df = df.drop(['parent_image_path'], axis=1)
df.shape

label = df['add_label']

data = np.array(label)

type(label)

# [parent_image_path] second_path, last_path  전처리

from sklearn.preprocessing import OneHotEncoder

df_temp = df

second_unique_data = list(set(df['second_path']))
last_unique_data = list(set(df['last_path']))

print(second_unique_data)
print(last_unique_data)
# list -> array  형 변환
second_data = np.array(second_unique_data)
last_data = np.array(last_unique_data)

# <second path>

# OneHotEncoder 객체 생성
encoder = OneHotEncoder()

# 토큰을 원핫 인코딩하여 변환
tokens_encoded = encoder.fit_transform(second_data.reshape(-1,1))

# 희소 행렬을 밀집 행렬로 변환하여 출력
print(tokens_encoded.toarray())
r_data = np.array(df['second_path'])

# 실제 데이터
real_data_encoded = encoder.transform(r_data.reshape(-1,1))

# <last path>

tokens_encoded_ = encoder.fit_transform(last_data.reshape(-1,1))

# 희소 행렬을 밀집 행렬로 변환하여 출력
print(tokens_encoded_.toarray())
r_data_ = np.array(df['last_path'])

# 실제 데이터
real_data_encoded_ = encoder.transform(r_data_.reshape(-1,1))
print(real_data_encoded_.toarray().shape)

new_column = df['label_int'].values
type(new_column)

# 배열 합치기
_real_data_encoded = real_data_encoded.toarray()
_real_data_encoded_ = real_data_encoded_.toarray()
encoded_data = _real_data_encoded.reshape(len(_real_data_encoded), -1)
encoded_data_ = _real_data_encoded_.reshape(len(_real_data_encoded), -1)

input_data = np.hstack((encoded_data, encoded_data_, new_column.reshape(-1,1)))

input_data.shape



# # 새로운 열을 추가할 데이터
# new_column =

# # 열 추가
# new_arr = np.hstack((arr, new_column.reshape(-1, 1)))


# edLabel = np.hstack(())

X_train, X_test = train_test_split(input_data, test_size=0.2, random_state=RANDOM_SEED)
X_train = X_train[(X_train[:, 20]==0)]
X_train = np.delete(X_train,20,axis=1)

y_test = X_test[:,20]
X_test = np.delete(X_test, 20, axis=1)
# X_train = X_train.values
# X_test = X_test.values
X_train.shape


# 입력 데이터의 차원 설정
input_dim = X_train.shape[1]
encoding_dim = 10

# 오토인코더 모델 정의
input_layer = Input(shape=(input_dim, ))

encoder = Dense(encoding_dim, activation="tanh",
                activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)

nb_epoch = 100
batch_size = 32


autoencoder.compile(optimizer='adam',
                    loss='mean_squared_error',
                    metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath="model.h5",
                               verbose=0,
                               save_best_only=True)
tensorboard = TensorBoard(log_dir='/logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)
history = autoencoder.fit(X_train, X_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(X_test, X_test),
                    verbose=1,
                    callbacks=[checkpointer, tensorboard]).history

autoencoder = load_model('model.h5')

plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right');

predictions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - predictions, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse,
                        'true_class': y_test})
error_df.describe()

fig = plt.figure()
ax = fig.add_subplot(111)
normal_error_df = error_df[(error_df['true_class']== 0) & (error_df['reconstruction_error'] < 10)]
_ = ax.hist(normal_error_df.reconstruction_error.values, bins=10)

plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

predictions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - predictions, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse,
                        'true_class': y_test})
error_df.describe()


fig = plt.figure()
ax = fig.add_subplot(111)
normal_error_df = error_df[(error_df['true_class']== 0) & (error_df['reconstruction_error'] < 10)]
_ = ax.hist(normal_error_df.reconstruction_error.values, bins=10)

fig = plt.figure()
ax = fig.add_subplot(111)
fraud_error_df = error_df[error_df['true_class'] == 1]
_ = ax.hist(fraud_error_df.reconstruction_error.values, bins=10)

from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)

fpr, tpr, thresholds = roc_curve(error_df.true_class, error_df.reconstruction_error)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.001, 1])
plt.ylim([0, 1.001])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show();

precision, recall, th = precision_recall_curve(error_df.true_class, error_df.reconstruction_error)
plt.plot(recall, precision, 'b', label='Precision-Recall curve')
plt.title('Recall vs Precision')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

plt.plot(th, precision[1:], 'b', label='Threshold-Precision curve')
plt.title('Precision for different threshold values')
plt.xlabel('Threshold')
plt.ylabel('Precision')
plt.show()

threshold = 0.08
groups = error_df.groupby('true_class')
fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='',
            label= "Fraud" if name == 1 else "Normal")
ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.show();

y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
conf_matrix = confusion_matrix(error_df.true_class, y_pred)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()
