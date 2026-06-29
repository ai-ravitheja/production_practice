import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

import joblib
df = pd.read_csv(r"C:\Users\guest user\Documents\AI Engineer\MachineLearning\sklearn\Housing.csv")
X = df.drop("price", axis=1)
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
num_cols = X.select_dtypes(include=["int64","float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns
num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),("scaler", StandardScaler())])
cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),("encoder", OneHotEncoder(handle_unknown="ignore"))])
preprocessor = ColumnTransformer([("num", num_pipe, num_cols),("cat", cat_pipe, cat_cols)])
models = [
(LinearRegression(),
{
    "model":[LinearRegression()],
}),
(RandomForestRegressor(random_state=42),
{
    "model":[RandomForestRegressor(random_state=42)],
    "model__n_estimators":[100,200],
    "model__max_depth":[5,10,None]
})
]
best_score = 0
best_model = None

for model, params in models:

    pipe = Pipeline([("preprocessor", preprocessor),("model", model)])

    grid = GridSearchCV(estimator=pipe,param_grid=params,cv=5,scoring="r2")

    grid.fit(X_train, y_train)

    print(grid.best_score_)
    print(grid.best_params_)

    if grid.best_score_ > best_score:
        best_score = grid.best_score_
        best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)

print("R2_SCORE:", r2_score(y_test, y_pred))
joblib.dump(best_model,"best_model.pkl")
