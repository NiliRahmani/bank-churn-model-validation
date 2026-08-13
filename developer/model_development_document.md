# RB-CHURN-07 Model Development Document

**Model:** Retail Banking Attrition Propensity Model
**Version:** 2.1
**Owner:** Retail Analytics, Customer Value
**Submitted for periodic validation:** 2024-08-15

> This is the submission under review. It is written from the developer's point
> of view and it is not a neutral document: several of its claims are the claims
> the validation report goes on to test. It is included so the review can be read
> against what it was actually given, rather than against a summary of it.

---

## 1. Purpose and intended use

RB-CHURN-07 estimates the probability that a retail banking customer closes
their primary chequing relationship within the next six months.

The score has two approved uses:

1. **Retention campaign targeting.** Customers are ranked by score and the top
   20% are contacted by the retention team each month. 20% is the volume the
   team can handle.
2. **Expected value of attrition.** The predicted probability is multiplied by
   the customer's balance to size the deposits at risk, which feeds the
   quarterly Customer Value pack.

Version 2.1 replaces version 2.0, a logistic regression retired in 2022. The
move to gradient boosting was made to capture non-linear interactions between
tenure, product holding and engagement that the previous model could not
represent.

## 2. Data sources and lineage

| Source | System | Fields |
|---|---|---|
| Customer master | Core banking | age, tenure, province, product holdings |
| Balances | Core banking | balance, estimated income band |
| Bureau | External vendor feed | credit score |
| Digital | Online and mobile platform | logins over trailing 90 days |
| Service | Complaints and contact log | complaints, retention call flag |

Development sample: 24,000 customers observed over 2023-01-01 to 2023-12-31,
with outcomes observed to 2024-06-30.

The extract is pulled monthly into the analytics environment. Standard platform
data quality controls apply at the point of ingestion and no additional cleaning
was carried out for this build.

## 3. Variable selection

Fourteen candidate variables were carried forward. Selection was driven by
predictive contribution measured on the development sample, subject to a review
for business sense.

`retention_call_flag` is the strongest single contributor. It indicates that the
retention team has logged a call against the customer, and it is interpreted as
an early warning signal: the team has good instincts about which customers are
becoming disengaged, and the model is picking that judgement up. Retaining it
adds substantial lift and it was kept for that reason.

`product_bundle_code` is the packaged product the customer holds (STD, PLUS,
SENIOR_PLUS, STUDENT). It is a commercial classification. **No protected
characteristic is used as an input to this model.**

Age is included as a continuous variable because attrition risk declines with
age across the portfolio.

## 4. Model methodology

Histogram-based gradient boosting classifier.

| Setting | Value |
|---|---|
| Train / test split | 70 / 30, random, stratified on the outcome |
| Class balancing | Minority class oversampled to a 50/50 base rate |
| Learning rate | 0.06 |
| Maximum iterations | 250 |
| Maximum leaf nodes | 31 |
| Minimum samples per leaf | 40 |
| Random seed | 42 |

Categorical variables are one-hot encoded. The minority class was oversampled
because the portfolio churn rate is roughly one in ten and the model would
otherwise under-predict the positive class.

## 5. Performance testing

Measured on the 30% held-out test sample:

| Metric | Value |
|---|---|
| AUC | 0.905 |
| Gini | 0.809 |

The model comfortably exceeds the 0.75 AUC threshold in the Retail Analytics
modelling standard. Performance is considered strong and the model is
recommended for production use.

## 6. Implementation and controls

The model is scored monthly in the analytics environment and the ranked list is
delivered to the retention team as a file. Access to the scoring notebook is
restricted to the Retail Analytics team. The model is scheduled for refit
annually.
