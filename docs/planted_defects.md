# Planted defects

The portfolio in `modelval/portfolio.py` is generated, and six defects are put
into it deliberately. This file is the answer key.

It exists because a validation project has an obvious weakness as a piece of
evidence: anyone can write a report that finds problems if they also chose the
problems. Publishing the list is the only way to make the review checkable.
Each defect below names the test that is supposed to catch it, and the finding
that test raises. If a test stops catching its defect, that is a broken test,
and `tests/test_smoke.py` asserts each of these links.

The order of work was: write the generator, write the tests against what a real
review would examine, run them, and only then compare against this list. Two
things that were not planted turned up anyway, and they are recorded at the
bottom because they are the more interesting half of the exercise.

---

## The six

### 1. A post-outcome variable

`retention_call_flag` is set for 78% of customers who churn and 4% of those who
do not. In the story the data tells, the retention team logs a call once a
customer has given notice, so the field is a record of the outcome.

**Caught by** `performance.variable_timing_review` — standalone AUC 0.872
against 0.627 for the next strongest variable.
**Raises** VF-01 (High).

### 2. Population shift between the two windows

Between the development window and the out-of-time window the channel mix moves
from 45% branch / 30% mobile to 18% branch / 58% mobile. Digital logins are
generated from channel, so they move with it. The sensitivity of churn to
complaints and to mobile channel also increases.

**Caught by** `stability.variable_stability` — PSI 0.44 on channel and 0.40 on
digital logins, against a red threshold of 0.25.
**Raises** VF-05 (Medium).

### 3. A proxy for a protected characteristic

`product_bundle_code` is assigned from age: SENIOR_PLUS to 85% of customers
aged 60 and over, STUDENT to 70% of those 24 and under.

**Caught by** `fairness.review` — the bundle code alone identifies customers
aged 60 and over with an AUC of 0.947, and the top-20% selection rate fails the
four-fifths test for two of the four age bands.
**Raises** VF-06 (Medium).

### 4. Duplicated records

1.5% of rows are appended a second time with the same customer identifier.

**Caught by** `quality.assess` — the uniqueness rule.
**Raises** VF-04 (Medium), together with defects 5 and 6.

### 5. A sentinel standing in for a missing value

12% of balances are written as `0` where the value was never supplied, so a nil
balance and an absent one are indistinguishable.

**Caught by** `quality.assess` — the accuracy rule.

### 6. Out-of-range and inconsistently coded values

Credit score uses two sentinel conventions (`999` and `-1`), age is blanked to
`0` on a small number of records, and province arrives as `ON`, `on`,
`Ontario` and `" ON "` from three upstream systems.

**Caught by** `quality.assess` — the validity and consistency rules.

---

## Two that were not planted

Neither of these was designed into the generator as a defect. They need
separating, though, because only one of them is a clean discovery.

**The probabilities are inflated (VF-03).** The recipe oversamples the minority
class to a 50/50 base rate and never maps the output back. Mean predicted
probability comes out at 33% against an observed rate of 10%. Nothing in the
generator causes this -- class balancing does, and the same thing would happen
on real data. It is invisible to AUC, which is why a submission that reports
only AUC would never surface it. This is a genuine discovery.

**The champion loses to its own benchmark (VF-02).** On the out-of-time sample
the corrected gradient-boosted model scores 0.666 and an eight-variable logistic
regression scores 0.700. The benchmark was written to show the champion was
*close* to a simple model, and it came out ahead of one.

This one deserves a caveat rather than a victory lap, and the caveat belongs
here rather than buried: **the generator draws churn from a linear log-odds
model**. A logistic regression is therefore close to the true functional form by
construction, and it is advantaged in this comparison in a way it would not
necessarily be on real data. The finding is still correct as stated -- on this
portfolio, on this period, the simpler model is more accurate, and a real review
would raise exactly that -- but the result should not be read as evidence that
gradient boosting generally loses to logistic regression. It is not, and this
data cannot show that either way.

What survives the caveat is the part that is not about model class at all: the
submission never ran the comparison. A benchmark is cheap, it is required by the
bank's own standard, and it would have told the developer something they did not
know. That is the finding.
