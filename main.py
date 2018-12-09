import pandas as pd
import xgboost

xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)

df = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

cols = ['YearBuilt', 'BsmtFinSF1', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'LotArea', 'OverallQual', 'OverallCond',
        'TotalBsmtSF']
df = df.fillna(0)
test = test.fillna(0)

x = df[cols]
y = df.SalePrice
xgb.fit(x, y)

y_pred = xgb.predict(test[cols])
test['SalePrice'] = y_pred
result = test[['Id', 'SalePrice']]
result.to_csv('result.csv', index=False)
