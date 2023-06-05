# -*- coding: utf-8 -*-
"""
Created on Wed May  3 14:15:02 2023

@author: howger
"""

import os 
import numpy as np 
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from mlxtend.plotting import scatterplotmatrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# In[Read data]
train_data = pd.read_excel('C:\\Users\\howger\\Desktop\\kaggle2\\train.xlsx')
test_data = pd.read_excel('C:\\Users\\howger\\Desktop\\kaggle2\\test.xlsx')
A = test_data.iloc[:,1:12]

import seaborn as sns
import matplotlib.pyplot as plt
# In[]
import pandas as pd

# 讀取 Excel 檔案

y = train_data["Power"]
train_data.drop(["Power"], axis="columns", inplace=True)
train_data.drop(["id"], axis="columns", inplace=True)

# 取得原始的 column name
original_columns = train_data.columns.tolist()

# 建立新的 column name 字串型態
new_columns = ['{:03d}'.format(i) for i in range(101, 112)]

# 重新排列 column name
train_data.columns = new_columns
A.columns = new_columns

# 印出結果
print(train_data.columns)
# In[one-hot enocoder_針對   "所在城市" 進行處理]
import pandas as pd
import pandas as pd

# 定義城市標籤
city_labels = ['North Bend', 'Seattle', 'Bellevue', 'Bothell', 'Federal Way', 'Kirkland', 'Issaquah', 'Woodinville',
               'Shoreline', 'Auburn', 'Maple Valley', 'Normandy Park', 'Fall City', 'Renton', 'Redmond', 'Sammamish',
               'Carnation', 'Snoqualmie', 'Kent', 'Kenmore', 'Newcastle', 'Mercer Island', 'Burien', 'Black Diamond',
               'Ravensdale', 'Covington', 'Clyde Hill', 'Algona', 'Lake Forest Park', 'Duvall', 'Skykomish', 'Tukwila',
               'Des Moines', 'Vashon', 'Yarrow Point', 'SeaTac', 'Medina', 'Snoqualmie Pass', 'Enumclaw', 'Pacific',
               'Preston', 'Milton','Inglewood-Finn Hill','Beaux Arts Village']

# 建立編碼字典
encoded_labels = {label: i for i, label in enumerate(city_labels)}

# 將訓練數據進行編碼
train_data["101"] = train_data["101"].map(encoded_labels)

# 將測試數據進行編碼
A["101"] = A["101"].map(encoded_labels)



# In[特徵評估]
#　使用隨機森林法
from  sklearn.ensemble import RandomForestClassifier

X = train_data
# 下面這行請選擇你的特徵 元素 [:10]，代表開頭到第10行是特徵
feat_labels = X.columns[:11]
tree = RandomForestClassifier(n_estimators=500, random_state=1)
tree.fit(X, y.astype('int'))
importances = tree.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(X.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))



plt.bar(range(X.shape[1]), 
        importances[indices],
        align='center',
        color = 'blue')

plt.xticks(range(X.shape[1]), 
            feat_labels[indices] )
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
plt.ylabel('Importance')
plt.grid(axis='y',linestyle='--')
plt.xlabel('Feature')
plt.title("RandomForest Feature Importances_Pressure")
#plt.savefig('images/04_09.png', dpi=300)
plt.show() 

# In[drop out the features]
X = train_data
X.drop(["110"], axis="columns", inplace=True)
X.drop(["106"], axis="columns", inplace=True)
X.drop(["109"], axis="columns", inplace=True)
X.drop(["107"], axis="columns", inplace=True)
X.drop(["111"], axis="columns", inplace=True)


A.drop(["110"], axis="columns", inplace=True)
A.drop(["106"], axis="columns", inplace=True)
A.drop(["107"], axis="columns", inplace=True)
A.drop(["109"], axis="columns", inplace=True)
A.drop(["111"], axis="columns", inplace=True)

# In[移除標準差][data removing from standard deviation]

import seaborn as sns
import matplotlib.pyplot as plt
# train_data.drop(["id"], axis="columns", inplace=True)
print("Shape Of The Before Ouliers: ", train_data.shape)
n = 3  # 使用 1.5 IQR(Q3-Q1) 為離群值判斷基準
columns = train_data.columns.tolist()
columns.remove('Power')


for column in columns:
    # 繪製標準差圖表
    
    # 移除離群值
    IQR = np.percentile(train_data[column], 75) - np.percentile(train_data[column], 25)
    train_data = train_data[train_data[column] < np.percentile(train_data[column], 75) + n * IQR]
    train_data = train_data[train_data[column] > np.percentile(train_data[column], 25) - n * IQR]

print("Shape Of The After Ouliers: ", train_data.shape)





# In[MinMax scaler]


from sklearn.preprocessing import MinMaxScaler
scalerx = MinMaxScaler()
X = scalerx.fit_transform(X) # 將訓練數據 轉換成0-1的矩陣
A = scalerx.fit_transform(A) 
# split 會變成陣列
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=None)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1, random_state=None)

# In[Gradient Boosting model]


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


# 建立梯度提升機模型
model = GradientBoostingRegressor()

# 訓練模型
model.fit(X_train, y_train)

# 預測測試集
y_prediction = model.predict(X_test)
y_prediction_train = model.predict(X_train)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
print("=========== Gradient Boosting =============")
print('R2 Test: ',r2_score(y_test, y_prediction))
print('R2 Train: ',r2_score(y_train,y_prediction_train))
print()

print('MSE test: ',mean_squared_error(y_test, y_prediction))
print('MSE train', mean_absolute_error(y_train, y_prediction_train))
print()
print('MAE test: %.3f' %(mean_absolute_error(y_test, y_prediction)))
print('MAE train: %.3f' %(mean_absolute_error(y_train, y_prediction_train)))



# In[Enumeration testing]
# =====================================================
# ====================================================
# ====================================================
from itertools import combinations

feature_counts = np.arange(1, 12)  # 特徵數量範圍從1到6

best_mae_error = float('inf')  # 初始化最佳的MAE誤差為無限大
best_feature_combination = None  # 初始化最佳的特徵組合為空

# 空矩陣用來存儲所有排列組合的分數
all_combinations = []
all_scores = []

for k in feature_counts:
    # 生成特徵的排列組合
    feature_combinations = combinations(range(6), k)
    
    for feature_combination in feature_combinations:
        selected_X_train = X_train[:, feature_combination]
        selected_X_val = X_val[:, feature_combination]
        
        # 建立模型
        model = keras.Sequential([
            layers.Dense(4, activation='LeakyReLU', input_shape=(k,)),
            layers.Dense(4, activation='LeakyReLU'),
            layers.Dense(4, activation='LeakyReLU'),
            layers.Dense(4, activation='LeakyReLU'),
            layers.Dense(1)
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(0.0002),
            loss=keras.losses.MeanSquaredError(),
            metrics=[keras.metrics.MeanAbsoluteError()]
        )

        history = model.fit(selected_X_train, y_train,
                            batch_size=1,
                            epochs=100,
                            validation_data=(selected_X_val, y_val),
                            verbose=0)

        mae_error = history.history['val_mean_absolute_error'][-1]  # 取得最後一個epoch的MAE誤差

        if mae_error < best_mae_error:
            best_mae_error = mae_error
            best_feature_combination = feature_combination
        
        # 將特徵組合和對應的MAE誤差存入矩陣
        all_combinations.append(feature_combination)
        all_scores.append(mae_error)
        print("=======")
        print(mae_error)

# 輸出最佳的特徵組合和對應的MAE誤差
print("最佳特徵組合：", best_feature_combination)
print("最佳MAE誤差：", best_mae_error)

# 輸出所有排列組合的特徵組合和對應的MAE誤差
for combination, score in zip(all_combinations, all_scores):
    print("特徵組合：", combination)
    print("MAE誤差：", score)



# =====================================================
# ====================================================
# ====================================================

# In[修改後模型架構]
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

# 建立一個Sequential型態的model
model = keras.Sequential(name='model-3')
# 第1層全連接層設為64個unit，將輸入形狀設定為(11, )，而實際上我們輸入的數據形狀為(batch_size, 11)
model.add(layers.Dense(6, activation='ReLU', input_shape=(6,)))
# 第2層全連接層設為64個unit
model.add(layers.Dense(10, activation='ReLU'))
model.add(layers.Dense(10, activation='ReLU'))
model.add(layers.Dense(10, activation='ReLU'))
model.add(layers.Dense(1))
model.compile(
    # optimizers使用Adam(0.002)
    optimizer=keras.optimizers.Adam(0.00005),
    # loss使用MAE Loss
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.MeanAbsoluteError()]
)

history = model.fit(X_train, y_train,
                    batch_size=1,
                    epochs=100,
                    validation_data=(X_val, y_val)
                    )


# In[神經元 數量測試] [MAE] 
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# 設定待測試的神經元數量和網路層數範圍
neuron_counts = [4,6,8,10 ]  # 不同的神經元數量
layer_counts = [1, 2, 3,4,5,6]  # 不同的網路層數

# 建立一個用於儲存結果的字典
results = {}

# 迴圈測試不同的神經元數量和網路層數
for neuron_count in neuron_counts:
    for layer_count in layer_counts:
        # 建立模型
        model = keras.Sequential()
        model.add(layers.Dense(neuron_count, activation='relu', input_shape=(10,)))
        for _ in range(layer_count - 1):
            model.add(layers.Dense(neuron_count, activation='relu'))
        model.add(layers.Dense(1))

        # 編譯模型
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # 訓練模型
        history = model.fit(X_train, y_train, epochs=100, verbose=0)

        # 評估模型
        _, mae = model.evaluate(X_test, y_test)

        # 儲存結果
        results[(neuron_count, layer_count)] = mae

# 輸出結果熱力圖
import matplotlib.pyplot as plt
import seaborn as sns

mae_values = np.array(list(results.values())).reshape(len(neuron_counts), len(layer_counts))
sns.heatmap(mae_values, annot=True, cmap='coolwarm', xticklabels=layer_counts, yticklabels=neuron_counts)
plt.xlabel('Layers')
plt.ylabel('Neuron numbers')
plt.title('MAE')
plt.show()

# In[神經元 數量測試] [R2] 
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# 設定待測試的神經元數量和網路層數範圍
neuron_counts = [4, 6, 8, 10]  # 不同的神經元數量
layer_counts = [9,10,11,12,13]  # 不同的網路層數

# 建立一個用於儲存結果的字典
results = {}

# 迴圈測試不同的神經元數量和網路層數
for neuron_count in neuron_counts:
    for layer_count in layer_counts:
        # 建立模型
        model = keras.Sequential()
        model.add(layers.Dense(neuron_count, activation='relu', input_shape=(10,)))
        for _ in range(layer_count - 1):
            model.add(layers.Dense(neuron_count, activation='relu'))
        model.add(layers.Dense(1))

        # 編譯模型
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # 訓練模型
        history = model.fit(X_train, y_train, epochs=100, verbose=0)

        # 評估模型
        y_pred = model.predict(X_test)
        y_test = y_test.flatten()
        y_pred = y_pred.flatten()


        r_squared = 1 - np.sum(np.square(y_test - y_pred)) / np.sum(np.square(y_test - np.mean(y_test)))


        # 儲存結果
        results[(neuron_count, layer_count)] = r_squared

# 輸出結果熱力圖
import matplotlib.pyplot as plt
import seaborn as sns

r_squared_values = np.array(list(results.values())).reshape(len(neuron_counts), len(layer_counts))
sns.heatmap(r_squared_values, annot=True, cmap='coolwarm', xticklabels=layer_counts, yticklabels=neuron_counts)
plt.xlabel('Layers')
plt.ylabel('Neuron numbers')
plt.title('R Squared')
plt.show()


# In[Gradient Boosting]


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


# 建立梯度提升機模型
model = GradientBoostingRegressor()

# 訓練模型
model.fit(X_train, y_train)

# 預測測試集
y_prediction = model.predict(X_test)



from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
print(r2_score(y_test, y_prediction))
print()

print('MSE_test: ',mean_squared_error(y_test, y_prediction))
print()
print('MAE test: %.3f' %(mean_absolute_error(y_test, y_prediction)))
print('MAE train: %.3f' %(mean_absolute_error(y_train, y_prediction_train)))
# In[顯示訓練結果和過程]
# 繪製損失趨勢
loss = history.history["loss"]
val_loss = history.history["val_loss"]  # 取得驗證損失

epochs = range(1, len(loss) + 1)

plt.title('Mean square error')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.plot(epochs, loss, "b-", label=" Training Loss")
plt.plot(epochs, val_loss, "r-", label="Validation Loss")  # 繪製驗證損失
plt.legend()  # 顯示圖例
plt.show()





# In[20230517_1026_繪製 MAE 平均誤差]

import matplotlib.pyplot as plt

# 獲取訓練集的 MAE 歷史值
mae_train = history.history['mean_absolute_error']

# 獲取驗證集的 MAE 歷史值
mae_val = history.history['val_mean_absolute_error']

# 繪製 MAE 的圖形
epochs = range(1, len(mae_train) + 1)
plt.plot(epochs, mae_train, 'b', label=' Training MAE')
plt.plot(epochs, mae_val, 'r', label='Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.show()

# In[測試數據_r2 coefficient_sk learn]
from sklearn.metrics import r2_score
y_pred = model.predict(X_test)
print(r2_score(y_test, y_pred))


# In[submission ]
##### ----------------------------------submission--------------------#######
submission = model.predict(A)




# In[prediction_submissison_version][單純向量機器\ 正規化版本]

predicted_ans = model.predict(A)
test_ids = test_data['id']
my_submission = pd.DataFrame({'id':test_ids, 'Power':predicted_ans[:,0]})
my_submission.to_csv('submission.csv', index=False)