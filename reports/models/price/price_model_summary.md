# Phase 12 — Supplier Purchase-Price Forecasting Models: Summary Report

## Executive Summary
Phase 12 trained, evaluated, and ranked advanced machine learning models for **supplier purchase-price forecasting** across six horizons (1, 3, 7, 14, 21, and 30 days).
Three distinct modeling approaches were evaluated:
1. **Direct Purchase-Price Regression** (LightGBM, CatBoost, XGBoost)
2. **Percentage-Change Regression & Price Reconstruction** $\hat{y} = y_{current} \times (1 + \hat{\Delta})$ (LightGBM, CatBoost, XGBoost)
3. **Price Direction Prediction**: Derived from regression vs Dedicated multiclass Classifiers (LightGBM, CatBoost, XGBoost)

Evaluation strictly adhered to the Phase 10 chronological 33-fold walk-forward validation framework.
Zero future leakage was allowed: training sets for horizon $H$ used strictly $t \le T - H$,
and feature inputs strictly respected the approved 163 temporally safe price features from `reports/leakage/price_features_temporally_safe.csv`.
The final holdout period (**2024-10-01 to 2024-12-31**) remained strictly protected and untouched.

## Best Candidate Purchase-Price Models per Horizon
```text
 horizon       task      champion_model   family          approach      mae  baseline_mae  mae_reduction_pct                     status
       1 regression lightgbm_pct_change lightgbm percentage_change 0.203675      0.203080          -0.292567 BEST_CANDIDATE_PRICE_MODEL
       3 regression catboost_pct_change catboost percentage_change 0.207654      0.207279          -0.180887 BEST_CANDIDATE_PRICE_MODEL
       7 regression catboost_pct_change catboost percentage_change 0.209010      0.209460           0.214869 BEST_CANDIDATE_PRICE_MODEL
      14 regression lightgbm_pct_change lightgbm percentage_change 0.210426      0.212361           0.911335 BEST_CANDIDATE_PRICE_MODEL
      21 regression lightgbm_pct_change lightgbm percentage_change 0.209853      0.210612           0.360371 BEST_CANDIDATE_PRICE_MODEL
      30 regression catboost_pct_change catboost percentage_change 0.210195      0.211080           0.419517 BEST_CANDIDATE_PRICE_MODEL
```

## Direct Regression vs. Percentage-Change Regression
```text
 horizon               model      mae  baseline_mae  mae_reduction_pct     rmse      bias
       1     catboost_direct 0.273683      0.203080         -34.765901 0.368348  0.016789
       1 catboost_pct_change 0.204457      0.203080          -0.678076 0.313968  0.008950
       1     lightgbm_direct 0.242748      0.203080         -19.533206 0.342814  0.011787
       1 lightgbm_pct_change 0.203675      0.203080          -0.292567 0.313727  0.008838
       1      xgboost_direct 0.246792      0.203080         -21.524303 0.345283  0.009939
       1  xgboost_pct_change 0.204056      0.203080          -0.480639 0.316420  0.009021
       3     catboost_direct 0.279163      0.207279         -34.679845 0.374871  0.006533
       3 catboost_pct_change 0.207654      0.207279          -0.180887 0.319918  0.000225
       3     lightgbm_direct 0.245343      0.207279         -18.363583 0.349854 -0.000520
       3 lightgbm_pct_change 0.209155      0.207279          -0.905007 0.321875  0.000460
       3      xgboost_direct 0.245607      0.207279         -18.491006 0.351298 -0.002570
       3  xgboost_pct_change 0.208305      0.207279          -0.495016 0.322174  0.001864
       7     catboost_direct 0.278297      0.209460         -32.863979 0.379942  0.014037
       7 catboost_pct_change 0.209010      0.209460           0.214869 0.323058  0.007396
       7     lightgbm_direct 0.244785      0.209460         -16.864569 0.352160  0.009371
       7 lightgbm_pct_change 0.209233      0.209460           0.108282 0.325663  0.008300
       7      xgboost_direct 0.247228      0.209460         -18.031208 0.355799  0.005847
       7  xgboost_pct_change 0.211195      0.209460          -0.828191 0.329020  0.009347
      14     catboost_direct 0.263764      0.212361         -24.205587 0.356242  0.021689
      14 catboost_pct_change 0.212778      0.212361          -0.196529 0.319608  0.020795
      14     lightgbm_direct 0.239381      0.212361         -12.723504 0.335764  0.020513
      14 lightgbm_pct_change 0.210426      0.212361           0.911335 0.316217  0.021787
      14      xgboost_direct 0.240601      0.212361         -13.298039 0.335993  0.018573
      14  xgboost_pct_change 0.211012      0.212361           0.635206 0.316182  0.021340
      21     catboost_direct 0.278081      0.210612         -32.035193 0.379982  0.010910
      21 catboost_pct_change 0.211138      0.210612          -0.250191 0.320509  0.006409
      21     lightgbm_direct 0.246407      0.210612         -16.996151 0.350538  0.006321
      21 lightgbm_pct_change 0.209853      0.210612           0.360371 0.320914  0.007261
      21      xgboost_direct 0.245966      0.210612         -16.786643 0.350380  0.004189
      21  xgboost_pct_change 0.209948      0.210612           0.315112 0.319321  0.011018
      30     catboost_direct 0.283244      0.211080         -34.187652 0.385933 -0.010741
      30 catboost_pct_change 0.210195      0.211080           0.419517 0.316262 -0.008077
      30     lightgbm_direct 0.254086      0.211080         -20.374121 0.357670 -0.008114
      30 lightgbm_pct_change 0.210707      0.211080           0.176708 0.318291 -0.008754
      30      xgboost_direct 0.261618      0.211080         -23.942222 0.368814 -0.015165
      30  xgboost_pct_change 0.212592      0.211080          -0.716019 0.319047 -0.008264
```

## Price Direction Performance (Accuracy & Balanced Accuracy)
```text
 horizon              method   family          approach  count  accuracy  balanced_accuracy  macro_f1  weighted_f1  macro_precision  macro_recall
       1 catboost_classifier catboost        classifier    792  0.436869           0.333333  0.202695     0.265653         0.145623      0.333333
       1     catboost_direct catboost            direct    792  0.535354           0.449555  0.447480     0.533695         0.459901      0.449555
       1 catboost_pct_change catboost percentage_change    792  0.635101           0.562676  0.559467     0.647500         0.562881      0.562676
       1 lightgbm_classifier lightgbm        classifier    792  0.614899           0.549324  0.545400     0.633313         0.553987      0.549324
       1     lightgbm_direct lightgbm            direct    792  0.555556           0.471904  0.466405     0.548366         0.485081      0.471904
       1 lightgbm_pct_change lightgbm percentage_change    792  0.640152           0.577385  0.571208     0.654537         0.575160      0.577385
       1  xgboost_classifier  xgboost        classifier    792  0.623737           0.556136  0.552227     0.640237         0.558952      0.556136
       1      xgboost_direct  xgboost            direct    792  0.545455           0.466259  0.461226     0.541815         0.480640      0.466259
       1  xgboost_pct_change  xgboost percentage_change    792  0.632576           0.571536  0.565085     0.647570         0.569570      0.571536
       3 catboost_classifier catboost        classifier    792  0.428030           0.333333  0.199823     0.256591         0.142677      0.333333
       3     catboost_direct catboost            direct    792  0.545455           0.458243  0.456974     0.541656         0.465839      0.458243
       3 catboost_pct_change catboost percentage_change    792  0.642677           0.564306  0.562104     0.653616         0.564809      0.564306
       3 lightgbm_classifier lightgbm        classifier    792  0.619949           0.550816  0.548438     0.641499         0.561554      0.550816
       3     lightgbm_direct lightgbm            direct    792  0.558081           0.474174  0.471270     0.556043         0.490104      0.474174
       3 lightgbm_pct_change lightgbm percentage_change    792  0.619949           0.535890  0.536907     0.636300         0.544889      0.535890
       3  xgboost_classifier  xgboost        classifier    792  0.599747           0.526687  0.528008     0.624926         0.545268      0.526687
       3      xgboost_direct  xgboost            direct    792  0.556818           0.464500  0.461515     0.552460         0.478791      0.464500
       3  xgboost_pct_change  xgboost percentage_change    792  0.642677           0.557708  0.557132     0.654062         0.560822      0.557708
       7 catboost_classifier catboost        classifier    792  0.440657           0.333333  0.203915     0.269569         0.146886      0.333333
       7     catboost_direct catboost            direct    792  0.559343           0.488075  0.484455     0.556849         0.496349      0.488075
       7 catboost_pct_change catboost percentage_change    792  0.617424           0.545379  0.544577     0.632854         0.551567      0.545379
       7 lightgbm_classifier lightgbm        classifier    792  0.589646           0.534368  0.530888     0.613391         0.547181      0.534368
       7     lightgbm_direct lightgbm            direct    792  0.565657           0.479950  0.471843     0.555455         0.491404      0.479950
       7 lightgbm_pct_change lightgbm percentage_change    792  0.616162           0.544735  0.543807     0.632326         0.551838      0.544735
       7  xgboost_classifier  xgboost        classifier    792  0.608586           0.546985  0.543965     0.628252         0.555170      0.546985
       7      xgboost_direct  xgboost            direct    792  0.558081           0.470125  0.461695     0.548583         0.482447      0.470125
       7  xgboost_pct_change  xgboost percentage_change    792  0.609848           0.539856  0.538927     0.626713         0.547605      0.539856
      14 catboost_classifier catboost        classifier    792  0.422980           0.333333  0.198166     0.251461         0.140993      0.333333
      14     catboost_direct catboost            direct    792  0.574495           0.488281  0.486372     0.573953         0.496049      0.488281
      14 catboost_pct_change catboost percentage_change    792  0.619949           0.544174  0.542633     0.635487         0.549097      0.544174
      14 lightgbm_classifier lightgbm        classifier    792  0.611111           0.550288  0.545154     0.632674         0.557099      0.550288
      14     lightgbm_direct lightgbm            direct    792  0.574495           0.470313  0.466011     0.565614         0.482617      0.470313
      14 lightgbm_pct_change lightgbm percentage_change    792  0.611111           0.535241  0.534939     0.630383         0.545093      0.535241
      14  xgboost_classifier  xgboost        classifier    792  0.621212           0.549514  0.547051     0.638674         0.554873      0.549514
      14      xgboost_direct  xgboost            direct    792  0.574495           0.463728  0.458724     0.564980         0.474694      0.463728
      14  xgboost_pct_change  xgboost percentage_change    792  0.619949           0.542070  0.541282     0.636206         0.548558      0.542070
      21 catboost_classifier catboost        classifier    792  0.465909           0.333333  0.211886     0.296159         0.155303      0.333333
      21     catboost_direct catboost            direct    792  0.570707           0.458163  0.455075     0.576833         0.473241      0.458163
      21 catboost_pct_change catboost percentage_change    792  0.643939           0.530023  0.532381     0.670232         0.548198      0.530023
      21 lightgbm_classifier lightgbm        classifier    792  0.613636           0.530866  0.525383     0.648327         0.548750      0.530866
      21     lightgbm_direct lightgbm            direct    792  0.589646           0.479521  0.472557     0.590758         0.496686      0.479521
      21 lightgbm_pct_change lightgbm percentage_change    792  0.646465           0.535389  0.534971     0.667740         0.545801      0.535389
      21  xgboost_classifier  xgboost        classifier    792  0.614899           0.531581  0.527024     0.651790         0.552838      0.531581
      21      xgboost_direct  xgboost            direct    792  0.580808           0.476317  0.468795     0.583130         0.494148      0.476317
      21  xgboost_pct_change  xgboost percentage_change    792  0.652778           0.550267  0.546489     0.674512         0.556824      0.550267
      30 catboost_classifier catboost        classifier    792  0.441919           0.333333  0.204320     0.270879         0.147306      0.333333
      30     catboost_direct catboost            direct    792  0.542929           0.448453  0.444930     0.538213         0.451501      0.448453
      30 catboost_pct_change catboost percentage_change    792  0.616162           0.529217  0.529972     0.633383         0.538561      0.529217
      30 lightgbm_classifier lightgbm        classifier    792  0.597222           0.526445  0.524961     0.622160         0.540849      0.526445
      30     lightgbm_direct lightgbm            direct    792  0.555556           0.470269  0.462775     0.548580         0.480850      0.470269
      30 lightgbm_pct_change lightgbm percentage_change    792  0.613636           0.536654  0.534787     0.631870         0.543505      0.536654
      30  xgboost_classifier  xgboost        classifier    792  0.594697           0.517571  0.517644     0.616666         0.530908      0.517571
      30      xgboost_direct  xgboost            direct    792  0.540404           0.454019  0.446554     0.534216         0.464581      0.454019
      30  xgboost_pct_change  xgboost percentage_change    792  0.608586           0.521054  0.522328     0.625819         0.531093      0.521054
```

## Segmented Performance Highlights
### By Product Category (7-Day Horizon)
```text
 horizon            category               model  count      mae     rmse     mape    smape      bias  direction_accuracy
       7 Cooking Oils & Fats     catboost_direct    132 0.484958 0.625762 0.017647 0.017824 -0.263778            0.621212
       7 Cooking Oils & Fats catboost_pct_change    132 0.448768 0.567019 0.016471 0.016482 -0.003061            0.598485
       7 Cooking Oils & Fats     lightgbm_direct    132 0.469825 0.614565 0.017111 0.017262 -0.231673            0.613636
       7 Cooking Oils & Fats lightgbm_pct_change    132 0.457302 0.576804 0.016784 0.016797 -0.001055            0.575758
       7 Cooking Oils & Fats      xgboost_direct    132 0.481732 0.624279 0.017536 0.017705 -0.264823            0.613636
       7 Cooking Oils & Fats  xgboost_pct_change    132 0.460850 0.583379 0.016913 0.016928 -0.003338            0.590909
       7        Dairy & Eggs     catboost_direct    132 0.220334 0.293453 0.019747 0.019650  0.009047            0.643939
       7        Dairy & Eggs catboost_pct_change    132 0.191739 0.242719 0.017184 0.017146  0.000693            0.583333
       7        Dairy & Eggs     lightgbm_direct    132 0.191448 0.245629 0.017211 0.017121  0.032181            0.598485
       7        Dairy & Eggs lightgbm_pct_change    132 0.191027 0.241982 0.017120 0.017082  0.000854            0.606061
       7        Dairy & Eggs      xgboost_direct    132 0.192892 0.249585 0.017356 0.017248  0.045980            0.560606
       7        Dairy & Eggs  xgboost_pct_change    132 0.191824 0.243410 0.017196 0.017152  0.004020            0.583333
       7         Dry Staples     catboost_direct    132 0.370258 0.464610 0.016410 0.016476 -0.113297            0.621212
       7         Dry Staples catboost_pct_change    132 0.359244 0.455376 0.016026 0.015979  0.047569            0.583333
       7         Dry Staples     lightgbm_direct    132 0.370423 0.465873 0.016411 0.016483 -0.111576            0.636364
       7         Dry Staples lightgbm_pct_change    132 0.354950 0.454955 0.015831 0.015789  0.046247            0.590909
       7         Dry Staples      xgboost_direct    132 0.374712 0.468551 0.016595 0.016675 -0.122538            0.606061
       7         Dry Staples  xgboost_pct_change    132 0.358988 0.459955 0.016011 0.015967  0.050387            0.590909
       7    Fresh Vegetables     catboost_direct    264 0.204723 0.244509 0.051784 0.049805  0.191756            0.458333
       7    Fresh Vegetables catboost_pct_change    264 0.067200 0.089626 0.015056 0.015015  0.005557            0.659091
       7    Fresh Vegetables     lightgbm_direct    264 0.151874 0.172365 0.038302 0.037352  0.140312            0.443182
       7    Fresh Vegetables lightgbm_pct_change    264 0.066692 0.089931 0.014897 0.014855  0.006603            0.659091
       7    Fresh Vegetables      xgboost_direct    264 0.150318 0.170431 0.038002 0.037042  0.143261            0.450758
       7    Fresh Vegetables  xgboost_pct_change    264 0.067688 0.090277 0.015084 0.015042  0.006408            0.643939
       7      Poultry & Meat     catboost_direct    132 0.184786 0.230240 0.025787 0.025573  0.068738            0.553030
       7      Poultry & Meat catboost_pct_change    132 0.119909 0.149479 0.016649 0.016654 -0.011940            0.621212
       7      Poultry & Meat     lightgbm_direct    132 0.133262 0.172099 0.018717 0.018486  0.086668            0.659091
       7      Poultry & Meat lightgbm_pct_change    132 0.118736 0.148061 0.016492 0.016490 -0.009449            0.606061
       7      Poultry & Meat      xgboost_direct    132 0.133398 0.172939 0.018742 0.018504  0.089939            0.666667
       7      Poultry & Meat  xgboost_pct_change    132 0.120132 0.148610 0.016689 0.016683 -0.007803            0.606061
```

### Imported vs. Domestic Products (7-Day Horizon)
```text
 horizon  imported_flag origin_type               model  count      mae     rmse     mape    smape      bias  direction_accuracy
       7              0    Domestic     catboost_direct    660 0.298062 0.403864 0.030391 0.029729 -0.013863            0.575758
       7              0    Domestic catboost_pct_change    660 0.233623 0.350403 0.016312 0.016288  0.008195            0.619697
       7              0    Domestic     lightgbm_direct    660 0.266478 0.378466 0.024449 0.024122 -0.011428            0.589394
       7              0    Domestic lightgbm_pct_change    660 0.233891 0.353227 0.016228 0.016204  0.009063            0.619697
       7              0    Domestic      xgboost_direct    660 0.269917 0.382825 0.024569 0.024245 -0.016949            0.577273
       7              0    Domestic  xgboost_pct_change    660 0.235888 0.356897 0.016358 0.016332  0.010404            0.613636
       7              1    Imported     catboost_direct    132 0.179473 0.224955 0.031202 0.030490  0.153539            0.477273
       7              1    Imported catboost_pct_change    132 0.085947 0.110853 0.014880 0.014849  0.003400            0.606061
       7              1    Imported     lightgbm_direct    132 0.136319 0.167078 0.023811 0.023442  0.113367            0.446970
       7              1    Imported lightgbm_pct_change    132 0.085945 0.111769 0.014881 0.014847  0.004488            0.598485
       7              1    Imported      xgboost_direct    132 0.133783 0.163662 0.023386 0.022990  0.119823            0.462121
       7              1    Imported  xgboost_pct_change    132 0.087728 0.112448 0.015189 0.015155  0.004059            0.590909
```

### By Supplier (7-Day Horizon)
```text
 horizon supplier_id               model  count      mae     rmse     mape    smape      bias  direction_accuracy
       7  SUP_BER_04     catboost_direct    132 0.370258 0.464610 0.016410 0.016476 -0.113297            0.621212
       7  SUP_BER_04 catboost_pct_change    132 0.359244 0.455376 0.016026 0.015979  0.047569            0.583333
       7  SUP_BER_04     lightgbm_direct    132 0.370423 0.465873 0.016411 0.016483 -0.111576            0.636364
       7  SUP_BER_04 lightgbm_pct_change    132 0.354950 0.454955 0.015831 0.015789  0.046247            0.590909
       7  SUP_BER_04      xgboost_direct    132 0.374712 0.468551 0.016595 0.016675 -0.122538            0.606061
       7  SUP_BER_04  xgboost_pct_change    132 0.358988 0.459955 0.016011 0.015967  0.050387            0.590909
       7  SUP_CAM_03     catboost_direct    132 0.229972 0.262612 0.072366 0.069120  0.229972            0.439394
       7  SUP_CAM_03 catboost_pct_change    132 0.048453 0.061460 0.015231 0.015181  0.007714            0.712121
       7  SUP_CAM_03     lightgbm_direct    132 0.167429 0.177495 0.052794 0.051261  0.167258            0.439394
       7  SUP_CAM_03 lightgbm_pct_change    132 0.047440 0.060687 0.014914 0.014863  0.008717            0.719697
       7  SUP_CAM_03      xgboost_direct    132 0.166852 0.176942 0.052618 0.051094  0.166698            0.439394
       7  SUP_CAM_03  xgboost_pct_change    132 0.047647 0.060459 0.014980 0.014928  0.008756            0.696970
       7  SUP_IMP_06     catboost_direct    132 0.179473 0.224955 0.031202 0.030490  0.153539            0.477273
       7  SUP_IMP_06 catboost_pct_change    132 0.085947 0.110853 0.014880 0.014849  0.003400            0.606061
       7  SUP_IMP_06     lightgbm_direct    132 0.136319 0.167078 0.023811 0.023442  0.113367            0.446970
       7  SUP_IMP_06 lightgbm_pct_change    132 0.085945 0.111769 0.014881 0.014847  0.004488            0.598485
       7  SUP_IMP_06      xgboost_direct    132 0.133783 0.163662 0.023386 0.022990  0.119823            0.462121
       7  SUP_IMP_06  xgboost_pct_change    132 0.087728 0.112448 0.015189 0.015155  0.004059            0.590909
       7  SUP_KLS_01     catboost_direct    132 0.184786 0.230240 0.025787 0.025573  0.068738            0.553030
       7  SUP_KLS_01 catboost_pct_change    132 0.119909 0.149479 0.016649 0.016654 -0.011940            0.621212
       7  SUP_KLS_01     lightgbm_direct    132 0.133262 0.172099 0.018717 0.018486  0.086668            0.659091
       7  SUP_KLS_01 lightgbm_pct_change    132 0.118736 0.148061 0.016492 0.016490 -0.009449            0.606061
       7  SUP_KLS_01      xgboost_direct    132 0.133398 0.172939 0.018742 0.018504  0.089939            0.666667
       7  SUP_KLS_01  xgboost_pct_change    132 0.120132 0.148610 0.016689 0.016683 -0.007803            0.606061
       7  SUP_LST_02     catboost_direct    132 0.220334 0.293453 0.019747 0.019650  0.009047            0.643939
       7  SUP_LST_02 catboost_pct_change    132 0.191739 0.242719 0.017184 0.017146  0.000693            0.583333
       7  SUP_LST_02     lightgbm_direct    132 0.191448 0.245629 0.017211 0.017121  0.032181            0.598485
       7  SUP_LST_02 lightgbm_pct_change    132 0.191027 0.241982 0.017120 0.017082  0.000854            0.606061
       7  SUP_LST_02      xgboost_direct    132 0.192892 0.249585 0.017356 0.017248  0.045980            0.560606
       7  SUP_LST_02  xgboost_pct_change    132 0.191824 0.243410 0.017196 0.017152  0.004020            0.583333
       7  SUP_SOP_05     catboost_direct    132 0.484958 0.625762 0.017647 0.017824 -0.263778            0.621212
       7  SUP_SOP_05 catboost_pct_change    132 0.448768 0.567019 0.016471 0.016482 -0.003061            0.598485
       7  SUP_SOP_05     lightgbm_direct    132 0.469825 0.614565 0.017111 0.017262 -0.231673            0.613636
       7  SUP_SOP_05 lightgbm_pct_change    132 0.457302 0.576804 0.016784 0.016797 -0.001055            0.575758
       7  SUP_SOP_05      xgboost_direct    132 0.481732 0.624279 0.017536 0.017705 -0.264823            0.613636
       7  SUP_SOP_05  xgboost_pct_change    132 0.460850 0.583379 0.016913 0.016928 -0.003338            0.590909
```

### Historical Price Volatility Segments (7-Day Horizon)
```text
 horizon volatility_group               model  count      mae     rmse     mape    smape      bias
       7             HIGH     catboost_direct    264 0.204723 0.244509 0.051784 0.049805  0.191756
       7             HIGH catboost_pct_change    264 0.067200 0.089626 0.015056 0.015015  0.005557
       7             HIGH     lightgbm_direct    264 0.151874 0.172365 0.038302 0.037352  0.140312
       7             HIGH lightgbm_pct_change    264 0.066692 0.089931 0.014897 0.014855  0.006603
       7             HIGH      xgboost_direct    264 0.150318 0.170431 0.038002 0.037042  0.143261
       7             HIGH  xgboost_pct_change    264 0.067688 0.090277 0.015084 0.015042  0.006408
       7              LOW     catboost_direct    264 0.295296 0.388572 0.018078 0.018063 -0.052125
       7              LOW catboost_pct_change    264 0.275491 0.364883 0.016605 0.016562  0.024131
       7              LOW     lightgbm_direct    264 0.280936 0.372405 0.016811 0.016802 -0.039698
       7              LOW lightgbm_pct_change    264 0.272988 0.364376 0.016475 0.016435  0.023550
       7              LOW      xgboost_direct    264 0.283802 0.375388 0.016975 0.016962 -0.038279
       7              LOW  xgboost_pct_change    264 0.275406 0.367972 0.016603 0.016560  0.027203
       7           MEDIUM     catboost_direct    264 0.334872 0.471481 0.021717 0.021699 -0.097520
       7           MEDIUM catboost_pct_change    264 0.284338 0.414641 0.016560 0.016568 -0.007500
       7           MEDIUM     lightgbm_direct    264 0.301544 0.451280 0.017914 0.017874 -0.072502
       7           MEDIUM lightgbm_pct_change    264 0.288019 0.421085 0.016638 0.016644 -0.005252
       7           MEDIUM      xgboost_direct    264 0.307565 0.458057 0.018139 0.018104 -0.087442
       7           MEDIUM  xgboost_pct_change    264 0.290491 0.425686 0.016801 0.016806 -0.005570
```

### Price Shock Periods vs Normal (7-Day Horizon)
```text
 horizon     price_shock_period               model  count      mae     rmse     mape    smape     bias
       7          Normal (< 5%)     catboost_direct    723 0.263564 0.355638 0.029723 0.029061 0.011789
       7          Normal (< 5%) catboost_pct_change    723 0.187452 0.290238 0.014279 0.014265 0.006308
       7          Normal (< 5%)     lightgbm_direct    723 0.227380 0.323335 0.023188 0.022876 0.007587
       7          Normal (< 5%) lightgbm_pct_change    723 0.188925 0.295079 0.014327 0.014313 0.007087
       7          Normal (< 5%)      xgboost_direct    723 0.230390 0.327502 0.023230 0.022916 0.004116
       7          Normal (< 5%)  xgboost_pct_change    723 0.190935 0.297405 0.014507 0.014491 0.007890
       7 Shock (|change| >= 5%)     catboost_direct     69 0.432675 0.575915 0.038947 0.038182 0.037592
       7 Shock (|change| >= 5%) catboost_pct_change     69 0.434903 0.561497 0.034873 0.034739 0.018796
       7 Shock (|change| >= 5%)     lightgbm_direct     69 0.427150 0.572749 0.036439 0.035880 0.028062
       7 Shock (|change| >= 5%) lightgbm_pct_change     69 0.422029 0.552254 0.033577 0.033428 0.021014
       7 Shock (|change| >= 5%)      xgboost_direct     69 0.423662 0.573758 0.036338 0.035774 0.023978
       7 Shock (|change| >= 5%)  xgboost_pct_change     69 0.423485 0.561930 0.033512 0.033368 0.024609
```

### Commodity-Sensitive Products (7-Day Horizon)
```text
 horizon      commodity_mapping               model  count      mae     rmse     mape    smape      bias
       7   Chicken (Feed Grain)     catboost_direct    132 0.184786 0.230240 0.025787 0.025573  0.068738
       7   Chicken (Feed Grain) catboost_pct_change    132 0.119909 0.149479 0.016649 0.016654 -0.011940
       7   Chicken (Feed Grain)     lightgbm_direct    132 0.133262 0.172099 0.018717 0.018486  0.086668
       7   Chicken (Feed Grain) lightgbm_pct_change    132 0.118736 0.148061 0.016492 0.016490 -0.009449
       7   Chicken (Feed Grain)      xgboost_direct    132 0.133398 0.172939 0.018742 0.018504  0.089939
       7   Chicken (Feed Grain)  xgboost_pct_change    132 0.120132 0.148610 0.016689 0.016683 -0.007803
       7 Cooking Oil (Palm Oil)     catboost_direct    132 0.484958 0.625762 0.017647 0.017824 -0.263778
       7 Cooking Oil (Palm Oil) catboost_pct_change    132 0.448768 0.567019 0.016471 0.016482 -0.003061
       7 Cooking Oil (Palm Oil)     lightgbm_direct    132 0.469825 0.614565 0.017111 0.017262 -0.231673
       7 Cooking Oil (Palm Oil) lightgbm_pct_change    132 0.457302 0.576804 0.016784 0.016797 -0.001055
       7 Cooking Oil (Palm Oil)      xgboost_direct    132 0.481732 0.624279 0.017536 0.017705 -0.264823
       7 Cooking Oil (Palm Oil)  xgboost_pct_change    132 0.460850 0.583379 0.016913 0.016928 -0.003338
       7      Eggs (Feed Grain)     catboost_direct    132 0.220334 0.293453 0.019747 0.019650  0.009047
       7      Eggs (Feed Grain) catboost_pct_change    132 0.191739 0.242719 0.017184 0.017146  0.000693
       7      Eggs (Feed Grain)     lightgbm_direct    132 0.191448 0.245629 0.017211 0.017121  0.032181
       7      Eggs (Feed Grain) lightgbm_pct_change    132 0.191027 0.241982 0.017120 0.017082  0.000854
       7      Eggs (Feed Grain)      xgboost_direct    132 0.192892 0.249585 0.017356 0.017248  0.045980
       7      Eggs (Feed Grain)  xgboost_pct_change    132 0.191824 0.243410 0.017196 0.017152  0.004020
       7    Other Non-Commodity     catboost_direct    264 0.204723 0.244509 0.051784 0.049805  0.191756
       7    Other Non-Commodity catboost_pct_change    264 0.067200 0.089626 0.015056 0.015015  0.005557
       7    Other Non-Commodity     lightgbm_direct    264 0.151874 0.172365 0.038302 0.037352  0.140312
       7    Other Non-Commodity lightgbm_pct_change    264 0.066692 0.089931 0.014897 0.014855  0.006603
       7    Other Non-Commodity      xgboost_direct    264 0.150318 0.170431 0.038002 0.037042  0.143261
       7    Other Non-Commodity  xgboost_pct_change    264 0.067688 0.090277 0.015084 0.015042  0.006408
       7            Rice (Rice)     catboost_direct    132 0.370258 0.464610 0.016410 0.016476 -0.113297
       7            Rice (Rice) catboost_pct_change    132 0.359244 0.455376 0.016026 0.015979  0.047569
       7            Rice (Rice)     lightgbm_direct    132 0.370423 0.465873 0.016411 0.016483 -0.111576
       7            Rice (Rice) lightgbm_pct_change    132 0.354950 0.454955 0.015831 0.015789  0.046247
       7            Rice (Rice)      xgboost_direct    132 0.374712 0.468551 0.016595 0.016675 -0.122538
       7            Rice (Rice)  xgboost_pct_change    132 0.358988 0.459955 0.016011 0.015967  0.050387
```

## Example Forecast Traces
Below are selected representative forecast traces demonstrating row-level predictions for later procurement decision making:

- **Forecast Origin**: 2021-12-31 | **Horizon**: 7d | **Model**: lightgbm_direct
  - Store: `STR_JHR_003` | Product: `PRD_CHK_01` | Supplier: `SUP_KLS_01`
  - Current Purchase Price: **RM 7.36**
  - Predicted Future Price: **RM 7.28** (Actual: **RM 7.18**)
  - Expected Absolute Change: **RM -0.08** | Pct Change: **-1.04%**
  - Predicted Direction: **DOWN** (Actual: **DOWN**)
  - Raw Uncertainty: [p10: RM 7.01, p50: RM 7.21, p90: RM 7.70]

- **Forecast Origin**: 2021-12-31 | **Horizon**: 7d | **Model**: lightgbm_direct
  - Store: `STR_JHR_003` | Product: `PRD_EGG_02` | Supplier: `SUP_LST_02`
  - Current Purchase Price: **RM 11.22**
  - Predicted Future Price: **RM 11.27** (Actual: **RM 11.18**)
  - Expected Absolute Change: **RM +0.05** | Pct Change: **+0.42%**
  - Predicted Direction: **FLAT** (Actual: **FLAT**)
  - Raw Uncertainty: [p10: RM 11.00, p50: RM 11.19, p90: RM 11.69]

- **Forecast Origin**: 2021-12-31 | **Horizon**: 7d | **Model**: lightgbm_direct
  - Store: `STR_JHR_003` | Product: `PRD_OIL_05` | Supplier: `SUP_SOP_05`
  - Current Purchase Price: **RM 26.87**
  - Predicted Future Price: **RM 26.64** (Actual: **RM 26.84**)
  - Expected Absolute Change: **RM -0.23** | Pct Change: **-0.87%**
  - Predicted Direction: **DOWN** (Actual: **FLAT**)
  - Raw Uncertainty: [p10: RM 26.37, p50: RM 26.56, p90: RM 27.05]

- **Forecast Origin**: 2021-12-31 | **Horizon**: 7d | **Model**: lightgbm_direct
  - Store: `STR_JHR_003` | Product: `PRD_ONN_06` | Supplier: `SUP_IMP_06`
  - Current Purchase Price: **RM 5.74**
  - Predicted Future Price: **RM 5.92** (Actual: **RM 5.72**)
  - Expected Absolute Change: **RM +0.18** | Pct Change: **+3.12%**
  - Predicted Direction: **UP** (Actual: **FLAT**)
  - Raw Uncertainty: [p10: RM 5.65, p50: RM 5.84, p90: RM 6.34]

## Model Training Runtime
```text
model_family  total_walk_forward_runtime_seconds  folds_evaluated
    lightgbm                              245.83               33
    catboost                              450.17               33
     xgboost                              411.61               33
```

All candidate models and metadata have been serialized under `models/price/<family>/`.
Reusable inference is available via `predict_purchase_price()` and `predict_price_direction()`.