import pandas as pd
import numpy as np
import joblib
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

RANDOM_STATE = 42

def load_data(path):
    return pd.read_csv(path, sep=';')

def split_data(df):
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=RANDOM_STATE)
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_train = df_train.quality
    y_val = df_val.quality
    y_test = df_test.quality

    del df_train['quality']
    del df_val['quality']
    del df_test['quality']

    return df_train, df_val, df_test, y_train, y_val, y_test

def grid_search(X_train, y_train, X_val, y_val):
    scores = []
    for depth in [5, 10, 15, 20, None]:
        for n in range(10, 201, 10):
            rf = RandomForestRegressor(n_estimators=n, max_depth=depth, random_state=RANDOM_STATE, n_jobs=-1)
            rf.fit(X_train, y_train)
            preds = rf.predict(X_val)
            rmse = root_mean_squared_error(y_val, preds)
            scores.append((depth if depth is not None else -1, n, rmse))
    return scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/winequality-red.csv')
    parser.add_argument('--out', default='model.joblib')
    args = parser.parse_args()

    df = load_data(args.data)
    df = df.fillna(0)

    df_train, df_val, df_test, y_train, y_val, y_test = split_data(df)

    X_train = df_train.values
    X_val = df_val.values
    X_test = df_test.values
    feature_names = df_train.columns.tolist()

    print('Starting grid search (this will take some time)...')
    scores = grid_search(X_train, y_train, X_val, y_val)
    df_scores = pd.DataFrame(scores, columns=['max_depth', 'n_estimators', 'rmse'])
    best_row = df_scores.loc[df_scores.rmse.idxmin()]
    best_depth = None if best_row.max_depth == -1 else int(best_row.max_depth)
    best_n = int(best_row.n_estimators)
    print('Best params found:', best_depth, best_n, 'val_rmse=', best_row.rmse)

    # retrain final model on train+val
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.hstack([y_train, y_val])
    final_rf = RandomForestRegressor(n_estimators=best_n, max_depth=best_depth, random_state=RANDOM_STATE, n_jobs=-1)
    final_rf.fit(X_trainval, y_trainval)

    preds_test = final_rf.predict(X_test)
    test_rmse = root_mean_squared_error(y_test, preds_test)
    print('Test RMSE:', test_rmse)

    joblib.dump({'model': final_rf, 'features': feature_names}, args.out)
    print('Saved model to', args.out)
