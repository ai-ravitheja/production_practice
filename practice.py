import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import joblib
df=pd.read_csv(r"C:\Users\guest user\Documents\AI Engineer\MachineLearning\sklearn\Housing.csv")
X=df.drop("price",axis=1)
y=df["price"]
num_cols_X=X.select_dtypes(include=["int64","float64"]).columns
cat_cols_X=X.select_dtypes(include=["object"]).columns
num_pipe=Pipeline([("imputer",SimpleImputer(strategy='mean'))])
cat_pipe=Pipeline([("imputer",SimpleImputer(strategy='most_frequent')),("encoder",OneHotEncoder())])
preprocessor=ColumnTransformer([("num",num_pipe,num_cols_X),("cat",cat_pipe,cat_cols_X)])
model=XGBRegressor(objective="reg:squarederror",random_state=42)
pipe=Pipeline([("preprocessor",preprocessor),("model",model)])
params={'model__n_estimators': [50,100,200,300],
        'model__learning_rate': [0.05,0.1],
        'model__max_depth': [3,4],
        'model__subsample': [0.8,1.0],
        'model__colsample_bytree': [0.8,1.0],
        "model__min_child_weight": [1, 3, 5]
}
grid=GridSearchCV(estimator=pipe,param_grid=params,cv=5,scoring='r2',n_jobs=-1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
grid.fit(X_train,y_train)
y_pred=grid.predict(X_test)
print("R2score",r2_score(y_test,y_pred))
print(grid.best_score_)
print(grid.best_params_)
joblib.dump(grid.best_estimator_,"XGBRegressor.pkl")
