import argparse
import pandas as pd
import joblib
import json

def simulate(df, preds, shelf_life_days=3, kg_per_unit=0.2):
    df = df.copy()
    df['predicted'] = preds
    df['diff'] = df['predicted'] - df['quantity_sold']
    df['overstock_kg'] = (df['diff'].clip(lower=0)) * kg_per_unit
    total_overstock_kg = df['overstock_kg'].sum()
    days = (df['date'].max() - df['date'].min()).days + 1
    factor = 30 / max(days, 1)
    estimated_monthly_kg = total_overstock_kg * factor
    return {'estimated_monthly_kg_discarded': estimated_monthly_kg}

def main(args):
    df = pd.read_csv(args.data, parse_dates=['date'])
    model = joblib.load(args.model)
    X = pd.get_dummies(df[['sku', 'store_id', 'temperature']], columns=['sku'])
    X = X.reindex(columns=model.feature_names_in_, fill_value=0)
    preds = model.predict(X)
    metrics = simulate(df, preds)
    with open(args.out, 'w') as f:
        json.dump(metrics, f, indent=2)
    print('Simulation complete. Metrics saved to', args.out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    main(args)
