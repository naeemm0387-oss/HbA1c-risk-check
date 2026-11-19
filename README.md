# Do I need a blood sugar test? – HbA1c risk check

A one-minute screening tool that estimates the chance that an adult **without a known diabetes diagnosis** has **elevated HbA1c (5.7% or higher, the prediabetes range or above)**, using 12 questions anyone can answer at home. No lab, no needle.

**Live website:** _[add the Streamlit link here]_

Final project, Data Science bootcamp (WBS Coding School), November 2026.

## The problem

In US national health survey data, about **3 in 10 adults** who were never told they have diabetes already have raised blood sugar. It usually causes no symptoms, so it goes unnoticed. A simple blood test finds it. But **who should get that test?**

## Data

[NHANES](https://wwwn.cdc.gov/nchs/nhanes/) 2021–2023 (US National Health and Nutrition Examination Survey), public data. Files used:

| File | Content |
|---|---|
| `DEMO_L` | demographics: age, sex, ethnicity, education |
| `BMX_L` | body measurements: weight, height, BMI, waist, hip |
| `GHB_L` | HbA1c blood test (the target) |
| `DIQ_L` | diabetes questionnaire (to remove people with a known diagnosis) |
| `BPQ_L` | blood pressure and cholesterol questions |

Final sample: **4,948 adults (20+) without a known diabetes diagnosis**, 31% with elevated HbA1c.

## Method

- **Target:** elevated HbA1c (≥ 5.7%) = 1, otherwise 0
- **13 features:** age, sex, BMI, waist, hip, waist-to-height, waist-to-hip, ethnicity, education, told high blood pressure, told high cholesterol, blood pressure medicine, cholesterol medicine
- **Model:** logistic regression in a scikit-learn pipeline (median/most-frequent imputation, scaling, ordinal and one-hot encoding)
- **Validation:** stratified 80/20 split; model and threshold chosen with 5-fold cross-validation on the training data; test set used once
- **Compared:** decision tree, random forest, gradient boosting, KNN (and tuned random forest and gradient boosting with RandomizedSearchCV). None was clearly better, so the simplest, most explainable model was kept.
- **Threshold:** 0.2, chosen for screening: finding people matters more than avoiding a cheap follow-up blood test

## Results (final test, 990 people never seen by the model)

| Measure | Value | In words |
|---|---|---|
| ROC-AUC | 0.76 | ranks a person with raised blood sugar above one without about 3 out of 4 times (cross-validation: 0.77, so no overfitting) |
| Sensitivity (recall) | 0.89 | finds 9 of 10 people with elevated HbA1c (275 of 309) |
| Precision | 0.44 | 4 of 10 people sent for a test really have elevated HbA1c |
| Specificity | 0.48 | about half of the people without elevated HbA1c get a false alarm |

### Model progress

| Step | ROC-AUC | Measured on |
|---|---|---|
| Dummy baseline | 0.50 | test set |
| MVP: 4 features (age, sex, BMI, waist) | 0.75 | test set |
| 10 features, pipeline | 0.79 | test set |
| 14 features, all people | 0.80 | cross-validation |
| **Final: 13 features, people without a known diagnosis** | **0.77 / 0.76** | **cross-validation / test set** |

The final score is a little lower because people who already know they have diabetes (mostly "easy" cases) were removed. The final model solves the harder, real screening task.

## Website

The Streamlit app asks the 12 questions and shows a risk percentage and a category:

| Score | Category | Advice |
|---|---|---|
| below 20% | Low | no strong signal |
| 20–39% | Moderate | consider an HbA1c test at the next check-up |
| 40% or more | High | ask your doctor for an HbA1c test soon |

## Repository structure

```
├── app.py                  # Streamlit website
├── requirements.txt
├── models/
│   └── hba1c_risk_model.joblib
├── data/                   # NHANES .xpt files (raw)
│   └── processed/          # cleaned CSV files
└── notebook/
    ├── 01_data_cleaning.ipynb
    ├── 02_data_merging.ipynb
    ├── 03_first_model_mvp.ipynb
    ├── 04_extra_features.ipynb
    ├── 05_model_comparison.ipynb
    ├── 06_final_model_all_people.ipynb
    ├── 07_final_model_undiagnosed.ipynb
    └── 08_website.ipynb
```

## How to run

```
pip install -r requirements.txt
python -m streamlit run app.py
```

To rerun the notebooks, put the NHANES 2021–2023 `.xpt` files listed above in `data/` and run the notebooks in order from the `notebook/` folder.

## Limitations

- **Not a diagnosis.** Only a blood test can confirm elevated HbA1c.
- Learned from **US data**; it needs checking in other populations (for example Europe).
- NHANES is cross-sectional: the tool estimates raised blood sugar **today**, not the risk of future diabetes.
- Many false alarms by design (low threshold for screening).
- Survey weights were not used, so results describe this sample, not the whole US population.
- "Prefer not to say" and "Don't know" answers are filled with the most common answer.

## Next steps

Check the model with European data, ask clinicians where the alert line should be, check performance by subgroup (sex, age, ethnicity), and check calibration.

## Author

Mohammed Naeem
