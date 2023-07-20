# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from scipy import stats
from sklearn.model_selection import train_test_split
# dataset = pd.read_csv('hiring.csv')

# dataset['experience'].fillna(0, inplace=True)

# dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)

# X = dataset.iloc[:, :3]

# #Converting words to integer values
# def convert_to_int(word):
#     word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
#                 'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
#     return word_dict[word]

# X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

# y = dataset.iloc[:, -1]

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()
#Fitting model with trainig data
# regressor.fit(X, y)



from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score
dataset = pd.read_csv('trainfy.csv')
z_scores = stats.zscore(dataset)
threshold = 3
outlier_indices = np.where(np.abs(z_scores) > threshold)[0]
dataset = dataset.drop(outlier_indices)
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25, random_state=0)
print('Training dataset size is ', X_train.shape)
print('Testing dataset size is  ', X_test.shape)
reg=RandomForestRegressor(n_estimators=50, random_state=1)
reg.fit(X_train, Y_train)


# Saving model to disk
pickle.dump(reg, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
# print(model.predict([[2, 9, 6]]))