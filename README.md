# E-commerce Customer Purchase Conversion Prediction

**AnalystLab Africa Experience Lab — Week 2 | Track 3: Machine Learning Engineering**

## 1. Project Overview

This project develops a machine learning classification model to predict whether an e-commerce website session will result in a purchase.

The objective is to identify sessions with a higher likelihood of conversion using information available during the session. These predictions can help businesses understand customer behaviour, identify potentially valuable sessions, and make more informed marketing and conversion-optimization decisions.

The project implements an end-to-end machine learning workflow, including data inspection, data quality assessment, exploratory data analysis, feature engineering, chronological data splitting, model training, evaluation, interpretation, and final validation.

## 2. Business Problem

E-commerce businesses generate large volumes of website traffic, but only a proportion of visitors complete purchases.

Understanding which sessions are more likely to convert can help businesses:

- Identify customer segments with different purchasing behaviours.
- Evaluate the relationship between device type and conversion.
- Understand how marketing sources and campaigns relate to purchasing behaviour.
- Support more informed marketing and customer-engagement strategies.
- Identify opportunities to improve the conversion experience.

The machine learning task is formulated as a binary classification problem.

**Target variable:**
- `converted = 1`: The website session resulted in a purchase.
- `converted = 0`: The website session did not result in a purchase.

## 3. Dataset Description

The project uses three datasets:

| Dataset | Description |
|---|---|
| `website_sessions.csv` | Website session information, including device type, marketing source, campaign, referral information, and repeat-session status. |
| `website_pageviews.csv` | Pageview activity associated with website sessions. |
| `orders_mav.csv` | Purchase records associated with website sessions. |

The original datasets contain 472,871 website sessions, 1,048,575 pageviews, and 32,313 orders.

The target was constructed by matching website sessions with order records using `website_session_id`.

### Target Distribution

| Outcome | Sessions | Percentage |
| No purchase | 440,558 | 93.17% |
| Purchase | 32,313 | 6.83% |
| **Total** | **472,871** | **100%** |

The dataset is imbalanced, with purchasing sessions representing approximately 6.83% of all sessions.

Original datasets are not included in this repository. The prepared public modelling dataset is provided under `data/processed/`.

## 4. Exploratory Data Analysis

Exploratory analysis investigated conversion patterns across device types, session types, marketing sources, campaigns, advertising content, referrers, and time.

Key findings included:

- Desktop sessions had a conversion rate of approximately 8.50%, compared with 3.09% for mobile sessions.
- Repeat sessions had a conversion rate of approximately 7.83%, compared with 6.64% for new sessions.
- Marketing source information was missing for approximately 17.62% of sessions.
- Conversion rates varied across marketing campaigns and advertising content.

These findings indicate potentially useful relationships between session characteristics and purchasing behaviour. They represent observed associations rather than evidence of causation.

## 5. Feature Engineering

The project developed two feature sets.

### Core Feature Set

Six input features were selected:

- `device_type`
- `utm_source`
- `utm_campaign`
- `utm_content`
- `http_referer`
- `is_repeat_session`

### Extended Feature Set

The extended feature set included the six core features and three additional time-based features:

- `session_hour`
- `session_day_of_week`
- `session_month`

Missing categorical values were represented using the category `Missing`.

Session identifiers, user identifiers, and timestamps were retained separately as metadata where needed. They were not used as model predictors.

The final public modelling dataset contains the six core features and the target variable.

## 6. Data Preparation and Validation Strategy

A chronological 80/20 train-test split was used to better reflect the practical scenario of predicting future website conversions using historical data.

| Dataset | Sessions |
| Training | 378,295 |
| Testing | 94,576 |
| **Total** | **472,871** |

The split occurred at `2014-11-28 23:32`.

The training data had a conversion rate of approximately 6.48%, while the test data had a conversion rate of approximately 8.23%.

This difference indicates a change in the target distribution over time and is considered when interpreting model performance.

Preprocessing was implemented using scikit-learn pipelines:

- Categorical features: imputation and one-hot encoding.
- Numerical features: median imputation and standardization.
- Unknown categorical values: handled using `handle_unknown="ignore"`.

Preprocessing was fitted on training data to avoid information leakage.

## 7. Model Development

Logistic Regression was implemented as the required baseline classification model.

The following classification algorithms were also evaluated:

- Logistic Regression
- Random Forest
- Gradient Boosting
- Histogram-based Gradient Boosting

Class weighting was used to address the imbalance between purchasing and non-purchasing sessions.

Alternative models were compared using a chronological validation split within the training data. Hyperparameter tuning was not performed.

Logistic Regression was retained as the final model because it satisfied the assignment requirement and offered a more interpretable modelling approach. Gradient Boosting was retained as a comparison model.

## 8. Final Model Evaluation

The final Logistic Regression model was refitted using the complete training dataset and evaluated on the untouched chronological test dataset.

### Test Performance

| Metric | Result |
| Accuracy | 37.75% |
| Precision | 10.39% |
| Recall | 86.05% |
| F1-score | 18.53% |
| ROC-AUC | 0.6052 |
| Average Precision | 0.1052 |
| Brier Score | 0.2454 |

### Confusion Matrix

| | Predicted No Purchase | Predicted Purchase |
| Actual No Purchase | 29,008 | 57,785 |
| Actual Purchase | 1,086 | 6,697 |

### Interpretation

The model identified 6,697 of the 7,783 purchasing sessions in the test dataset, achieving a recall of approximately 86.05%.

However, its precision was approximately 10.39%. This means that only around 10 out of every 100 sessions classified as likely purchases actually resulted in purchases.

The model therefore favours identifying a large proportion of potential purchasing sessions at the cost of generating many false positives.

Its ROC-AUC of 0.6052 indicates modest discriminatory ability. The Average Precision score of 0.1052 is also only modestly above the test-set purchase prevalence of approximately 8.23%.

The results suggest that the model has limited predictive strength and should not be treated as a highly accurate conversion prediction system.

## 9. Feature Importance and Interpretation

Feature interpretation was conducted using Logistic Regression coefficients and Gradient Boosting feature importance.

Device type was the most influential feature in both model interpretation approaches.

For Logistic Regression, the grouped absolute coefficient importance was highest for `device_type`.

For Gradient Boosting, device type accounted for approximately 88.7% of the model's impurity-based feature importance.

This suggests that device type plays a particularly strong role in how the evaluated models distinguish between purchasing and non-purchasing sessions.

Feature importance indicates how a model uses available information. It does not establish that a feature causes a purchase.

## 10. Business Recommendations

Based on the exploratory analysis and model findings, the following actions are recommended:

1. **Investigate mobile conversion performance:** Mobile sessions had substantially lower conversion rates than desktop sessions. Businesses should examine mobile navigation, page performance, checkout usability, and payment experience.

2. **Evaluate marketing campaign performance:** Conversion rates varied across marketing campaigns and advertising content. Campaign performance should be assessed using both traffic volume and actual purchase outcomes.

3. **Improve marketing attribution:** Missing marketing information affected a considerable proportion of sessions. Better tracking and campaign-tagging practices could improve reporting and future modelling.

4. **Support repeat-customer engagement:** Repeat sessions showed a higher observed conversion rate than new sessions. Businesses could investigate appropriate retention and customer-engagement strategies.

5. **Use predictions cautiously:** The current model's low precision means that predictions should be treated as preliminary signals rather than reliable purchase confirmations. It is better suited to further investigation and model improvement than direct automated marketing decisions.

## 11. Limitations and Future Improvements

The project has several limitations:

- The target variable is imbalanced.
- The final model produces a high number of false positives.
- Predictive performance is modest.
- The conversion rate differs between training and testing periods, suggesting temporal distribution shift.
- Missing marketing attribution information limits the available campaign context.
- The model uses session-level information and does not capture the full complexity of customer behaviour.
- The probability estimates may require further calibration assessment.

Potential future improvements include:

- Exploring additional behavioural features derived from pageview activity, provided they are available before the prediction point.
- Investigating strategies for handling class imbalance.
- Assessing probability calibration.
- Monitoring changes in conversion patterns over time.
- Evaluating the model using business-oriented cost and benefit measures.
- Testing the model on more recent and independently collected data.

## 12. Project Structure
```text
Customer_Purchase_Conversion_Prediction/
│
├── data/
│   └── processed/
│       └── public_modeling_dataset.csv
│
├── figures/
│   ├── eda/
│   ├── final_validation/
│   └── model_evaluation/
│
├── models/
│   ├── final/
│   └── stage_09/
│
├── notebooks/
│   └── Customer_Purchase_Conversion_Prediction.ipynb
│
├── reports/
│   ├── eda/
│   ├── error_analysis/
│   ├── feature_engineering/
│   ├── feature_importance/
│   ├── model_evaluation/
│   └── preprocessing/
│
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── features/
│   └── models/
│
├── .gitignore
├── README.md
└── requirements.txt
```
Original datasets and identifier-containing intermediate files are excluded from version control.

## 13. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Jupyter Notebook
- Git and GitHub
- Visual Studio Code

## 14. Reproducibility

### Requirements

The project dependencies are listed in `requirements.txt`.

### Environment Setup

Create and activate a virtual environment:

```bash
python -m venv venv
```

On Windows:

```powershell
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

### Running the Project

The numbered Python scripts under `src/` document the project workflow, from initial dataset inspection through final model validation.

The Jupyter Notebook under `notebooks/` provides a consolidated walkthrough of the analysis and modelling process.

**Dataset availability:** The original CSV files are not distributed in this repository. Reproducing the complete workflow requires access to the original datasets. The prepared public modelling dataset is included for inspection and use with the modelling workflow.

## 15. Conclusion

This project demonstrates an end-to-end machine learning workflow for e-commerce purchase conversion prediction.

It covers data quality assessment, exploratory analysis, feature engineering, chronological validation, classification model comparison, model interpretation, and final testing.

Although the final Logistic Regression model achieved relatively high recall, its low precision and modest discrimination highlight the challenges of predicting purchasing behaviour from session-level information.

The project provides a practical foundation for further work in customer analytics, classification modelling, and machine learning engineering.

---

**Project:** E-commerce Customer Purchase Conversion Prediction  
**Programme:** AnalystLab Africa Experience Lab — Week 2  
**Track:** Track 3 — Machine Learning Engineering
