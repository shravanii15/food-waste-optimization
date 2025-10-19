import argparse
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

def main(args):
    df = pd.read_csv(args.data, parse_dates=['date'])
    X = pd.get_dummies(df[['sku', 'store_id', 'temperature']], columns=['sku'])
    y = df['quantity_sold']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, args.out)
    print(f'Model trained and saved to {args.out}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    main(args)
