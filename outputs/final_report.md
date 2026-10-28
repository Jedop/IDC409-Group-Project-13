# Final evaluation report

## 1. ASR - word error rate
| asr_model | condition | n_clips | mean_wer | corpus_wer |
|---|---|---|---|---|
| base.en | clean | 60 | 0.051 | 0.046 |
| base.en | noisy_10dB | 60 | 0.167 | 0.130 |
| base.en | noisy_5dB | 60 | 0.235 | 0.195 |
| base.en | noisy_0dB | 60 | 0.420 | 0.398 |
| tiny.en | clean | 60 | 0.079 | 0.064 |
| tiny.en | noisy_10dB | 60 | 0.238 | 0.187 |
| tiny.en | noisy_5dB | 60 | 0.351 | 0.302 |
| tiny.en | noisy_0dB | 60 | 0.620 | 0.580 |

## 2a. Keyword detection
| asr_model | condition | precision | recall | f1 |
|---|---|---|---|---|
| base.en | clean | 0.977 | 0.935 | 0.956 |
| base.en | noisy_10dB | 0.944 | 0.739 | 0.829 |
| base.en | noisy_5dB | 0.868 | 0.717 | 0.786 |
| base.en | noisy_0dB | 0.793 | 0.500 | 0.613 |
| tiny.en | clean | 0.976 | 0.870 | 0.919 |
| tiny.en | noisy_10dB | 0.872 | 0.739 | 0.800 |
| tiny.en | noisy_5dB | 0.886 | 0.674 | 0.765 |
| tiny.en | noisy_0dB | 0.618 | 0.457 | 0.525 |

## 2b. Matching to the expected transcript
| asr_model | condition | n_clips | exact_accuracy | fuzzy_accuracy | mean_similarity |
|---|---|---|---|---|---|
| base.en | clean | 60 | 0.667 | 0.983 | 0.987 |
| base.en | noisy_10dB | 60 | 0.350 | 0.883 | 0.936 |
| base.en | noisy_5dB | 60 | 0.150 | 0.700 | 0.895 |
| base.en | noisy_0dB | 60 | 0.033 | 0.367 | 0.797 |
| tiny.en | clean | 60 | 0.517 | 0.950 | 0.974 |
| tiny.en | noisy_10dB | 60 | 0.267 | 0.767 | 0.906 |
| tiny.en | noisy_5dB | 60 | 0.083 | 0.567 | 0.844 |
| tiny.en | noisy_0dB | 60 | 0.033 | 0.167 | 0.688 |

## 3. ML - regression (sorted by MAE)
| feature_set | model | n_features | MAE | RMSE | R2 |
|---|---|---|---|---|---|
| all | decision_tree | 9 | 0.115 | 0.175 | 0.601 |
| all | random_forest | 9 | 0.126 | 0.178 | 0.585 |
| reduced | random_forest | 6 | 0.130 | 0.184 | 0.558 |
| reduced | decision_tree | 6 | 0.131 | 0.194 | 0.509 |
| pca95 | linear_regression | 6 | 0.132 | 0.172 | 0.613 |
| reduced | linear_regression | 6 | 0.132 | 0.172 | 0.613 |
| all | linear_regression | 9 | 0.133 | 0.174 | 0.606 |
| pca95 | random_forest | 6 | 0.135 | 0.196 | 0.498 |
| pca95 | neural_net | 6 | 0.142 | 0.187 | 0.542 |
| pca95 | svr | 6 | 0.142 | 0.198 | 0.490 |
| reduced | svr | 6 | 0.142 | 0.198 | 0.490 |
| all | svr | 9 | 0.144 | 0.197 | 0.493 |
| reduced | neural_net | 6 | 0.144 | 0.196 | 0.498 |
| all | neural_net | 9 | 0.147 | 0.204 | 0.456 |
| pca95 | decision_tree | 6 | 0.174 | 0.283 | -0.046 |
| all | baseline_mean | 9 | 0.239 | 0.289 | -0.093 |

## 3. ML - classification (sorted by F1)
| feature_set | model | n_features | accuracy | precision | recall | F1 |
|---|---|---|---|---|---|---|
| all | logistic_regression | 9 | 0.908 | 0.827 | 0.956 | 0.887 |
| reduced | logistic_regression | 6 | 0.908 | 0.827 | 0.956 | 0.887 |
| pca95 | logistic_regression | 6 | 0.908 | 0.827 | 0.956 | 0.887 |
| reduced | svm | 6 | 0.867 | 0.746 | 0.978 | 0.846 |
| pca95 | svm | 6 | 0.867 | 0.746 | 0.978 | 0.846 |
| all | svm | 9 | 0.858 | 0.750 | 0.933 | 0.832 |
| pca95 | random_forest | 6 | 0.833 | 0.698 | 0.978 | 0.815 |
| reduced | decision_tree | 6 | 0.842 | 0.741 | 0.889 | 0.808 |
| reduced | random_forest | 6 | 0.800 | 0.657 | 0.978 | 0.786 |
| all | random_forest | 9 | 0.800 | 0.657 | 0.978 | 0.786 |
| pca95 | decision_tree | 6 | 0.825 | 0.731 | 0.844 | 0.783 |
| all | neural_net | 9 | 0.792 | 0.647 | 0.978 | 0.779 |
| pca95 | neural_net | 6 | 0.783 | 0.638 | 0.978 | 0.772 |
| reduced | neural_net | 6 | 0.758 | 0.611 | 0.978 | 0.752 |
| all | decision_tree | 9 | 0.708 | 0.568 | 0.933 | 0.706 |
| all | baseline_majority | 9 | 0.375 | 0.375 | 1.000 | 0.545 |

## 4. Best classifier - detailed report
```
              precision    recall  f1-score   support

           0       0.97      0.88      0.92        75
           1       0.83      0.96      0.89        45

    accuracy                           0.91       120
   macro avg       0.90      0.92      0.90       120
weighted avg       0.92      0.91      0.91       120

```
