import numpy as np, pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df=load_diabetes(as_frame=True).frame
X,y=df.drop(columns="target"),df["target"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)
model=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),Ridge(alpha=1.0))
model.fit(X_train,y_train); pred=model.predict(X_test)
print("MAE:",mean_absolute_error(y_test,pred))
print("RMSE:",np.sqrt(mean_squared_error(y_test,pred)))
print("R-squared:",r2_score(y_test,pred))
