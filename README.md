# Obesity Level Classifier: five models, one Streamlit app

**Course:** Machine Learning, M.Tech (AIML), BITS Pilani WILP · **Assignment 2**
**Student:** Sangeet Brijpuriya · **ID:** 2026aj05903

- **Live Streamlit app:** `<PASTE YOUR STREAMLIT COMMUNITY CLOUD LINK HERE>`
- **GitHub repository:** `<PASTE YOUR GITHUB REPO LINK HERE>`

---

## a. Problem statement

Predict a person's **obesity level** (7 ordered classes, from *Insufficient Weight* to *Obesity Type III*) using only their **eating habits, lifestyle and basic profile** (age, gender, family history, transport, activity, and so on), **without** their height or weight.

Five classifiers (Logistic Regression, Decision Tree, K-Nearest Neighbours, Naive Bayes and Support Vector Machine) are trained on the same data, compared on six metrics, and exposed through a Streamlit app where a test CSV can be uploaded and scored with the saved models.

## b. Dataset description

| Item | Detail |
|---|---|
| Source | UCI Machine Learning Repository, *Estimation of Obesity Levels Based On Eating Habits and Physical Condition* (dataset id 544) |
| Original size | 2,111 rows × 16 features + 1 target |
| Target | `NObeyesdad`, 7 classes: Insufficient_Weight, Normal_Weight, Overweight_Level_I, Overweight_Level_II, Obesity_Type_I, Obesity_Type_II, Obesity_Type_III |
| Problem type | Multi-class classification (7 classes, roughly balanced: 272 to 351 rows per class) |
| Missing values | None |
| Exact duplicate rows | 24, removed before splitting, leaving **2,087 rows** |
| Features used | **14** (minimum required: 12); see below |
| Split | Stratified 80 / 20 with seed 2026: **1,669 train / 418 test** |

**Features used (14)**

| Type | Columns |
|---|---|
| Numeric (6), standardised | `Age`, `FCVC` (vegetable-eating frequency), `NCP` (number of main meals), `CH2O` (daily water intake), `FAF` (physical-activity frequency), `TUE` (time using technology devices) |
| Categorical (8), one-hot encoded | `Gender`, `family_history_with_overweight`, `FAVC` (frequent high-calorie food), `CAEC` (eating between meals), `SMOKE`, `SCC` (calorie monitoring), `CALC` (alcohol), `MTRANS` (transport used) |

**Why `Height` and `Weight` were dropped.** The class label is essentially a BMI band: in the data, the BMI range of each class lines up with the standard BMI categories. With Height and Weight available, any model can recompute BMI and score near-perfectly, which would make the model comparison meaningless. Without them the task is to infer the level from lifestyle alone, which is harder and separates the models clearly.

**Things to be aware of**
- UCI states that 77% of the rows were generated synthetically with the SMOTE technique, and only 23% were collected from real respondents. In the data, about 75% of rows (1,580 of 2,111) have fractional answers to whole-number survey questions. Scores here are therefore not a guide to real-world accuracy.
- Hyper-parameters were tuned with 5-fold stratified cross-validation on the **training split only** (metric: macro-F1). The test split was used once, for the final numbers.
- Precision, Recall and F1 are **macro-averaged** (each of the 7 classes counts equally). AUC is the **macro-averaged one-vs-rest** AUC. MCC is the multi-class Matthews correlation coefficient.

## c. GitHub repository

`<PASTE YOUR GITHUB REPO LINK HERE>`

```
project-folder/
├── app.py                 Streamlit app (loads saved models, never retrains)
├── evaluation.py          column lists + the six metrics (shared by training and app)
├── train_models.py        data cleaning, split, tuning, training, saving
├── requirements.txt
├── README.md
├── test_data.csv          held-out 20 % split (14 features + NObeyesdad), 418 rows
├── data/
│   └── ObesityDataSet_raw_and_data_sinthetic.csv    original UCI file
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    ├── model_summary.json     best hyper-parameters + test metrics
    └── test_metrics.csv
```

Each `.joblib` file is a full scikit-learn pipeline (scaling + one-hot encoding + classifier), so the app needs no separate preprocessing files.

**Run locally**
```bash
pip install -r requirements.txt
python train_models.py        # optional: regenerates model/ and test_data.csv
streamlit run app.py
```

## d. Models used

Test-set results (418 held-out rows):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|:--|--:|--:|--:|--:|--:|--:|
| Logistic Regression | 0.6172 | 0.8925 | 0.6131 | 0.6078 | 0.5897 | 0.5584 |
| Decision Tree | 0.7751 | 0.8652 | 0.7669 | 0.7678 | 0.7667 | 0.7374 |
| KNN | 0.7847 | 0.9265 | 0.7871 | 0.7778 | 0.7701 | 0.7515 |
| Naive Bayes | 0.5383 | 0.8543 | 0.5918 | 0.5238 | 0.4936 | 0.4768 |
| Support Vector Machine | **0.8325** | **0.9706** | **0.8338** | **0.8270** | **0.8276** | **0.8050** |

Settings chosen by cross-validation: Logistic Regression `C=100`; Decision Tree unlimited depth, `min_samples_leaf=1`; KNN `k=3`, distance-weighted; Gaussian Naive Bayes `var_smoothing=0.1`; SVM with RBF kernel, `C=10`, `gamma=0.3`.

### Observations on each model

| ML Model Name | Observation about model performance |
|:--|:--|
| Logistic Regression | Weak (accuracy 0.617). Train accuracy (0.629) is almost the same as test accuracy (0.617), so it is **underfitting**: a linear boundary cannot capture how lifestyle features combine to determine the level. Its AUC (0.893) is much better than its accuracy suggests, meaning it ranks the classes sensibly but often picks the wrong top class. |
| Decision Tree | Accuracy 0.775 and MCC 0.737. CV chose a fully grown tree, which fits the training set perfectly (train accuracy 1.00) but drops to 0.775 on test, a sign of overfitting. Its AUC (0.865) is the second lowest despite good accuracy, because leaf probabilities are coarse and rank classes poorly. |
| KNN | Accuracy 0.785 with the second-best AUC (0.927). `k=3` with distance weighting works well here. A likely reason is that most rows are synthetic points interpolated between real ones, so many test rows have a very similar neighbour in the training set. Train accuracy of 1.00 is expected with distance weighting, because each training point is its own nearest neighbour. |
| Naive Bayes | The weakest model (accuracy 0.538, F1 0.494). It assumes features are independent given the class and Gaussian, but 8 of the 14 inputs are 0/1 one-hot columns and several features are related. It is poor on the middle classes, with recall of only 0.12 for Normal_Weight, 0.21 for Overweight_Level_II and 0.26 for Overweight_Level_I, while Obesity_Type_III is recalled perfectly (1.00). Like Logistic Regression it underfits (train 0.560, test 0.538). |
| Support Vector Machine | Best on all six metrics (accuracy 0.833, AUC 0.971, MCC 0.805). The RBF kernel captures the non-linear structure that Logistic Regression and Naive Bayes miss. Obesity_Type_III is classified perfectly (precision and recall 1.00); the hardest classes are Normal_Weight (F1 0.667) and Overweight_Level_I (recall 0.673). Train accuracy (0.988) vs test (0.833) shows some remaining overfitting, yet it generalises best. |
| **Overall winner for this dataset** | **Support Vector Machine.** It leads on every metric, and it also had the best cross-validated macro-F1 on the training split (0.818 vs 0.751 for KNN), so the ranking is not a quirk of one test split. Caveat: the test set has only 418 rows, so small gaps (for example Decision Tree vs KNN) should not be over-interpreted. Also, even the best model confuses neighbouring levels often: only about a third of the SVM's mistakes are to an adjacent level, which shows that lifestyle data alone is a weak predictor of BMI band. |

## Using the Streamlit app

1. Upload a CSV (the sidebar offers `test_data.csv` as a sample). It must contain the 14 feature columns and `NObeyesdad`.
2. Pick a model from the dropdown.
3. Read the six metrics, the confusion matrix (counts or % per actual level), the classification report, a side-by-side table of all five models on your upload, and row-level predictions.

The app only loads the saved `.joblib` pipelines; it does not retrain on the uploaded data.
