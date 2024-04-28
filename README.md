# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Scaling for the feature in the data set.

STEP 4:Apply Feature Selection for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```
![image](https://github.com/swethaselvarajm/EXNO-4-DS/assets/119525603/6e19decc-aca1-4c5d-a5a8-808e5fa87e4b)
```
data.isnull().sum()
```
![image](https://github.com/swethaselvarajm/EXNO-4-DS/assets/119525603/271aa063-6a80-4c28-811a-7171a56ff2b9)
```
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/swethaselvarajm/EXNO-4-DS/assets/119525603/b47d1c32-8a17-45e2-a2e8-bae88ba106e9)
```
data2=data.dropna(axis=0)
data2
```
![image](https://github.com/swethaselvarajm/EXNO-4-DS/assets/119525603/45dae742-c63c-4a66-9f22-528abdeb80c7)
```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/swethaselvarajm/EXNO-4-DS/assets/119525603/793856c0-3130-491d-913c-079c91df14ec)
```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/swethaselvarajm/EXNO-4-DS/assets/119525603/db98fa80-01d2-48b9-9602-249ed3de56e5)
```
data2
```
![image](https://github.com/swethaselvarajm/EXNO-4-DS/assets/119525603/5a0bcd74-1b11-43fe-8a06-5ebdec810fb3)
```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
 ```
![image](https://github.com/swethaselvarajm/EXNO-4-DS/assets/119525603/15759418-8554-4894-b17d-189bf8738a23)
```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/swethaselvarajm/EXNO-4-DS/assets/119525603/e6f4ca22-5ff3-4931-928a-41f32f27ff05)
```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```

![image](https://github.com/swethaselvarajm/EXNO-4-DS/assets/119525603/b76be506-0fe7-45d9-a7b7-c0ab9d85dc93)
```
y=new_data['SalStat'].values
print(y)
```

![image](https://github.com/swethaselvarajm/EXNO-4-DS/assets/119525603/d347c35b-17f2-464b-937a-6f250f1eb9b6)
```
x=new_data[features].values
print(x)
```

![image](https://github.com/swethaselvarajm/EXNO-4-DS/assets/119525603/a6c71ac3-9d02-489e-b8b9-f608a8dcd4a9)
```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```

![image](https://github.com/swethaselvarajm/EXNO-4-DS/assets/119525603/be5771c2-c53a-4abe-8093-b8a133ab3a28)
```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```

![image](https://github.com/swethaselvarajm/EXNO-4-DS/assets/119525603/5bc37c28-adcf-42bb-bba3-af3968706a0d)
```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```

![image](https://github.com/swethaselvarajm/EXNO-4-DS/assets/119525603/12854589-eb9c-498d-935f-a787d3dcfd1b)
```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```

![image](https://github.com/swethaselvarajm/EXNO-4-DS/assets/119525603/73ca3858-916c-4764-a1c0-df6266f07f48)
```
data.shape
```

![image](https://github.com/swethaselvarajm/EXNO-4-DS/assets/119525603/3e7ec22d-61c7-46e9-81b5-b5de3c0d8e54)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```

![image](https://github.com/swethaselvarajm/EXNO-4-DS/assets/119525603/588e94d8-2242-4f82-917c-767918b6d72c)
```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```

![image](https://github.com/swethaselvarajm/EXNO-4-DS/assets/119525603/d9652b0e-b598-4474-b929-2e21eae2748d)
```
tips.time.unique()
```

![image](https://github.com/swethaselvarajm/EXNO-4-DS/assets/119525603/85335a63-e7e1-4769-a559-292172b20e10)
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```

![image](https://github.com/swethaselvarajm/EXNO-4-DS/assets/119525603/34568951-2d9e-4f97-861a-3e610b496809)
```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```

![image](https://github.com/swethaselvarajm/EXNO-4-DS/assets/119525603/a4521743-8b08-4407-aa79-2f71e04ff4b7)


# RESULT:
Thus, Feature selection and Feature scaling has been used on thegiven dataset.
