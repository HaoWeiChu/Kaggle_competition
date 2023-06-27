# Kaggle_competition
## [Classifier](https://github.com/HaoWeiChu/Kaggle_competition/blob/main/Classifier/Kaggle_Classifier.ipynb)
* 題目:手機降頻(Underclocking)預測

* 題目描述:在研發手機產品過程中，需考慮手機過熱問題; 若手機過熱仍持續加溫，會讓使用者感受不佳甚至是慢性燙傷之情形發生。為了避免此種情形，當手機達到臨界狀態，必須"降頻"以降低過熱之發生。決定此臨界降頻的因素有許多，例如手機螢幕溫度、電池電量、CPU跑分階級等等，皆需考量。

* 目標:請使用Data中之14個影響降頻與否的特徵，來預測手機是否需要降頻(Yes--1 or No--0)
* 此分類問題利用Accuracy來評估分數(Score)

=========== Results =============
* accuracy test:  0.6956521739130435
* accuracy train:  1.0

## [Regression](https://github.com/HaoWeiChu/Kaggle_competition/blob/main/Regression/Kaggle_Regression.ipynb)
* 題目:風力發電機功率(Power)預測 [迴歸類問題]

* 題目描述:風力發電機依照其建構特徵、設置地點等因素之不同，有不一樣的發電功率。

* 目標:請使用Data中之11個特徵，來預測風力發電機的發電功率。
* 此迴歸問題利用R2_score (coefficient of determination) 來評估分數。
  
=========== Results =============
1. DNN Regression
* R2 Test:  0.10984360149112637
* R2 Train:  0.17235688783986436

2. Linear Regression
* R2 Test:  0.10984360149112637
* R2 Train:  0.17235688783986436

3.  Gradient Boosting
* R2 Test:  0.10135157903816949
* R2 Train:  0.8492494162613897

