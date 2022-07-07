import pandas as pd
import numpy as np
import lightgbm as lgb

from typing import Dict


def join_datasets(df: pd.DataFrame, other_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """in-place join some features from other_dfs to df"""
    print('joining datasets...')
    # geo-clusters
    df['city_id'] = df['geo_cluster_id'].map(
        other_dfs['df_geo']['city_id']
    ).astype(np.int8)
    
    # sku-based features, integer-based ids
    cols_to_join = [
        'category_id',
        'product_type_id',
        'brand_id',
        'trademark_id',
        'origin_country_id',
        'commodity_group_id',
    ]
    for c in cols_to_join:
        df[c] = df['sku_id'].map(other_dfs['df_sku'][c])
    return df


def add_features_to_skeleton(
    df: pd.DataFrame, 
    date_col: str = 'date',
    target_col: str = 'sales',
    group_cols : tuple = ('sku_id', 'geo_cluster_id'),
) -> pd.DataFrame:
    print('calculating temporal features...')
    df['weekday'] = df['date'].dt.weekday.astype(np.int8)
    df['week_no'] = df['date'].dt.isocalendar().week.astype(np.int8)
    df['month'] = df['date'].dt.month.astype(np.int8)
    print('calculating price features')
    df['price_change_perc'] = (
        df['price'] / df.groupby(list(group_cols))['price'].shift() - 1
    ).replace([np.inf, -np.inf, np.nan], 0.).astype(np.float32)
    df['price_change_logdiff'] = np.log(
        df['price'] / df.groupby(list(group_cols))['price'].shift()
    ).replace([np.inf, -np.inf, np.nan], 0.).astype(np.float32)
    return df


def percentage_mae(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    grouping: np.ndarray
) -> float:
    """
    given grouping col values, calculates per-group MAE 
    and averages it across group col with equal weight
    """
    def _wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.abs(y_true - y_pred).sum() / y_true.sum()
    
    group_metrics = []
    for gr in grouping.unique():
        idx = (grouping == gr)
        gr_metric = _wape(y_true[idx], y_pred[idx])
        group_metrics.append(gr_metric)
        
    group_metrics = np.array(group_metrics)
    agg_metric = group_metrics.mean() # over group axis
    
    return agg_metric


# https://stackoverflow.com/questions/63399806/
# how-to-pass-additional-parameters-to-lgbm-custom-loss-function
def perc_mae_lgb():   
    def _perc_mae_lgb(preds, train_data: lgb.Dataset):
        score = 0.
        y_true = train_data.get_label()
        grouping = train_data.grouping_ # custom column to be used in grouping calcs
#         y_pred = preds.reshape(-1, 1)
        # do whatever with ur custom vars and calculate score....
        perc_mae_score = percentage_mae(
            y_true=y_true.ravel(),
            y_pred=preds.ravel(),
            grouping=grouping,
        )
        return 'perc_mae', perc_mae_score, False
    return _perc_mae_lgb


def calculate_scores(
    df: pd.DataFrame, 
    idx_val: np.ndarray, 
    idx_test: np.ndarray,
    pred_val: np.ndarray,
    pred_test: np.ndarray,
    grouping_col: str = 'date',
    target_col: str = 'sales',
):
    """
    Given test/val idx and correspondent predictions,
    calculates metrics (public/private proxy).
    Assure similar period length 
    and temporal ordering is preserved
    """
    score_pub = percentage_mae(
        y_true=df.loc[idx_val, target_col],
        y_pred=pred_val,
        grouping=df.loc[idx_val, grouping_col],
    )
    score_pr = percentage_mae(
        y_true=df.loc[idx_test, target_col],
        y_pred=pred_test,
        grouping=df.loc[idx_test, grouping_col],
    )

    print(
        f'public : {score_pub:.5f}\n'
        f'private: {score_pr:.5f}\n'
    )
    
    return score_pub, score_pr
