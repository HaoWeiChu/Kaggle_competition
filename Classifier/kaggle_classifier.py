# -*- coding: utf-8 -*-
"""
Created on Wed May  3 14:15:02 2023

@author: howger
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
# In[Read data]

train_data = pd.read_excel('train.xlsx')


test_data = pd.read_excel('test.xlsx')
A = test_data.iloc[:, 1:15]


# In[]
import pandas as pd

# 讀取 Excel 檔案

y = train_data["Underclocking"]
train_data.drop(["Underclocking"], axis="columns", inplace=True)

# 取得原始的 column name
original_columns = train_data.columns.tolist()

# 建立新的 column name
new_columns = ['{:03d}'.format(i) for i in range(101, 115)]

# 重新排列 column name
train_data.columns = new_columns
A.columns = new_columns

# 印出結果
print(train_data.columns)


# In[Feature_selection]
#　使用隨機森林_評估特徵
from  sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt

# 選擇特徵範圍，[:14]，代表開頭到第14行是特徵
feat_labels = X.columns[:14]
tree = RandomForestClassifier(n_estimators=500, random_state=1)
tree.fit(X, y)
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

plt.show() 




# In[drop out the features]


X = train_data
X.drop(["112"], axis="columns", inplace=True)
X.drop(["114"], axis="columns", inplace=True)
X.drop(["107"], axis="columns", inplace=True)


A.drop(["112"], axis="columns", inplace=True)
A.drop(["114"], axis="columns", inplace=True)
A.drop(["107"], axis="columns", inplace=True)

# In[MinMax scaler]
# ====== 0-1 scaler ==============
from sklearn.preprocessing import MinMaxScaler
scalerx = MinMaxScaler()
X = scalerx.fit_transform(X) # 將訓練數據 轉換成0-1的矩陣
A = scalerx.fit_transform(A) # 將測試數據 轉換成0-1的矩陣

# split 會變成陣列
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)



# In[standard scaler]
# ============ standard scaler ==========

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
A = scaler.transform(A) # submission data




# In[決策邊界 繪製]
'''
利用 兩個特徵 進行 分類法:
    目前有錯誤.....無法繪製決策邊界
'''
from matplotlib.colors import ListedColormap
from distutils.version import LooseVersion
import matplotlib.pyplot as plt
import matplotlib
from  sklearn.ensemble import RandomForestClassifier
import numpy as np

tree = RandomForestClassifier(n_estimators = 250, max_depth = 64)
tree.fit(X_train,y_train)
# Create our predictions
prediction = tree.predict(X_test)
# Create confusion matrix
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
confusion_matrix(y_test, prediction)
# Display accuracy score
print(accuracy_score(y_test, prediction))
# Display F1 score


# In[Confusion_Matrix plot]
confusion_matrix = np.array([[458, 44], [129, 150]])

labels = ['Positive', 'Negative']
tick_labels = ['Positive', 'Negative']

plt.imshow(confusion_matrix, cmap='jet')

# 添加顏色條
plt.colorbar()

# 添加刻度標籤
plt.xticks(np.arange(len(tick_labels)), tick_labels)
plt.yticks(np.arange(len(labels)), labels)

# 添加數值標籤
for i in range(len(labels)):
    for j in range(len(tick_labels)):
        plt.text(j, i, confusion_matrix[i, j], ha='center', va='center', color='white')

# 設置標題和軸標籤
plt.title('Confusion Matrix')
plt.xlabel(' True')
plt.ylabel('Prediciton')

# 顯示圖表
plt.show()

# In[output score]
print('訓練集: ',tree.score(X_train,y_train))
print('測試集: ',tree.score(X_test,y_test))

# In[Submission data prediction][單純向量機器\ 正規化版本]

predicted_ans = tree.predict(A)
test_ids = test_data['id']
my_submission = pd.DataFrame({'id':test_ids, 'Underclocking':predicted_ans})
my_submission.to_csv('submission.csv', index=False)