import argparse
import pandas as pd
import matplotlib.pyplot as plt
import joblib

def main(args):
    df = pd.read_csv(args.data, parse_dates=['date'])
    model = joblib.load(args.model)
    X = pd.get_dummies(df[['sku', 'store_id', 'temperature']], columns=['sku'])
    X = X.reindex(columns=model.feature_names_in_, fill_value=0)
    df['predicted'] = model.predict(X)
    for sku in df['sku'].unique():
        sku_df = df[df['sku'] == sku]
        plt.figure()
        plt.plot(sku_df['date'], sku_df['quantity_sold'], marker='o', label='Actual')
        plt.plot(sku_df['date'], sku_df['predicted'], marker='x', label='Predicted')
        plt.title(f'Demand for {sku}')
        plt.xlabel('Date')
        plt.ylabel('Quantity')
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    main(args)
