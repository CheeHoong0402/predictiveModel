# Explainability and SHAP Value Limitations

## 1. Correlation vs. Causation
- **Statistical Attribution Only**: SHAP (Shapley Additive Explanations) attributes predictive variance to input features based on historical correlations within the training dataset.
- **No Causal Invariance**: A large positive contribution from a feature (e.g. `diesel_price_lag_1` on purchase price) indicates that historical price spikes coincide with fuel adjustments; it does not constitute mathematical proof that subsidizing diesel will reduce wholesale chicken prices.

## 2. Feature Collinearity & Attribution Sharing
- **Correlated Feature Clusters**: When highly collinear features coexist (e.g. `demand_lag_7` vs `demand_rolling_7_mean`), tree split algorithms distribute importance across the cluster.
- **Family Aggregation**: To prevent misleading interpretations of individual collinear features, business users should prioritize **Business Feature Families** rather than single isolated lag terms.

## 3. Out-of-Distribution Inputs
- **Extrapolation Bounds**: Tree-based ensembles partition space with orthogonal splits. If input features fall far outside observed historical bounds (e.g., unprecedented extreme currency shock), TreeSHAP contributions project along boundary tree leaves.
- **Uncertainty Flagging**: Extreme predictions are paired with wider Phase 16 calibrated prediction intervals ($P_{90} - P_{10}$) to signal elevated epistemic uncertainty.

## 4. Tree Depth and Higher-Order Interactions
- **Interaction Constraints**: TreeSHAP faithfully accounts for interactions up to the maximum depth of the trees (depth 4-8 in our LightGBM, CatBoost, and XGBoost models). Complex higher-order interactions across more than 8 features are smoothed.