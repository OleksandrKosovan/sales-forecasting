# AfterParty solution 

## Pipeline: 

0. Run `notebooks/FN-eda-data-cleaning-and-no-ml-baseline-v2`, which will produce `data/processed/df_train.parquet` and `data/processed/df_test.parquet`
1. Run `notebooks/TimeCV_DayModels_Id2Median_PriceChangeFeature_final_version.ipynb`, which will produce `submissions/sub_TimeCV_DayModels_Id2Median_PriceChangeFeature.csv`
2. Run `notebooks/time_cv_nn.ipynb`, which will produce `submissions/baseline_tmax50_milr1e6_emb10xLR_lr3e4_bs256_nodrop_ver2_9models_leshapreproc.csv`
3. Run `notebooks/blend_of_blends.ipynb`, which will produce `submissions/tree_and_nn_mean_blend_final.csv`

## Detailed information:

AfterParty_Approach_Hack4Retail.pdf