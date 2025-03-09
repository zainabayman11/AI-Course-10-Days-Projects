
import pandas as pd
dataset=pd.read_csv("Salary_Data.csv")
X=dataset.iloc[:,[0]].values
y=dataset.iloc[:,1].values
#splitting the dataset into train test 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.30)
# training  import algorithm >>> model 
from sklearn.linear_model import LinearRegression
model=LinearRegression()
#   fit <<learn    predict  << test
model.fit(X_train,y_train) 
#testing
ypred=model.predict(X_test)


#visualization  training 
import matplotlib.pyplot as plt
plt.scatter(X_train,y_train, color='red')
plt.plot(X_train,model.predict(X_train), color='blue')
plt.title("salary vs years of experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# testin visualization 
plt.scatter(X_test,y_test, color='red')
plt.plot(X_train,model.predict(X_train), color='blue')
plt.plot(X_test,ypred, color='green')
plt.title("salary vs years of experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
#   y =W*X+B
W=model.coef_
B=model.intercept_
print("Y=",W,"*X","+",B)
# evaluation metrics 
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

mae=mean_absolute_error(y_test,ypred)
print(mae)
mse=mean_squared_error(y_test,ypred)
print(mse)

r2=r2_score(y_test,ypred)
print(r2)

model.predict([[6.8]])
