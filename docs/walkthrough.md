# Walkthrough

Notes I wrote for myself while building this, kept in the repository because
they are the part I would want to read first if someone handed me the project.

---

## The project in one paragraph

A retail bank has a churn model in production. It scores every customer monthly,
and the top 20% get contacted by the retention team. The model is up for its
periodic validation. I am not the person who built it; my job is to independently
challenge it and either sign it off, sign it off with conditions, or refuse it.
I refused it, on two findings, and the report sets out what would change that.

## Why the answer is not "the model is bad"

The model is not bad. It replicates, it is documented well enough to rebuild
from scratch, and its ranking is real. The problem is that it was tested in a
way that could not have shown its two actual weaknesses:

- one input is only knowable after the outcome has happened, and
- the only performance evidence comes from a random split of the build period.

Both are testing failures rather than modelling failures, which is the point I
would want to make in an interview. A model risk function is not a second
modelling team. It exists to ask whether the evidence supports the claim.

## The eight findings, in plain language

| Ref | Severity | In one sentence |
|---|---|---|
| VF-01 | High | The model uses a field that only gets filled in after the customer has already said they are leaving. |
| VF-02 | High | Once that field is removed, a plain logistic regression beats the model on the next six months. |
| VF-03 | Medium | The model says a customer has a 33% chance of leaving when the real rate is 10%, because the training data was rebalanced and never converted back. |
| VF-04 | Medium | The input file has duplicate customers, two different codes for "no credit score", and zeros standing in for missing balances. |
| VF-05 | Medium | Customers moved from branch to mobile within six months of the build, and nothing was watching. |
| VF-06 | Medium | The retention campaign contacts 30% of under-30s and 9% of over-60s, and the model gets age through a product code even though age was supposedly not used. |
| VF-07 | Medium | Performance was only ever measured on a random slice of the same year, which cannot detect the shift in VF-05. |
| VF-08 | Low | The development document has no limitations section, no monitoring plan and no benchmark comparison. |

## Things I need to be able to define without hesitating

**AUC / Gini.** AUC is the probability the model scores a randomly chosen
churner above a randomly chosen non-churner. Gini is `2 x AUC - 1`. Banks
usually quote Gini. AUC measures ranking only, which is exactly why it never
showed VF-03.

**KS.** The largest gap between the cumulative score distributions of churners
and non-churners. A second view of separation, common in credit.

**PSI.** Population stability index. Bin the development sample, compare the
share falling in each bin later, sum `(actual - expected) * ln(actual /
expected)`. Convention is under 0.10 fine, 0.10-0.25 watch, over 0.25 act. The
bin edges must come from the development sample; recutting them on the new data
hides the movement you are looking for.

**Calibration.** Whether a predicted 20% actually happens 20% of the time.
Separate from ranking. Matters as soon as anyone multiplies the probability by a
dollar amount, which the business case here does.

**Four-fifths rule.** A group's selection rate divided by the highest group's
rate. Below 0.8 is the conventional trigger for a closer look. It is a
screening threshold, not a verdict.

**Effective challenge.** Critical review by people with the standing,
competence and incentive to disagree. "Incentive" is the part that gets left
out: a validator who reports to the model owner is not independent.

**Out-of-time versus random split.** A random split draws test rows from the
same period as the training rows. An out-of-time sample uses a later period, so
it tests whether the model still works when the world has moved on.

**SR 11-7 and OSFI E-23.** The US Federal Reserve's supervisory guidance on
model risk management, and OSFI's Canadian equivalent. Both frame model risk as
coming from fundamental errors and from correct models used incorrectly, and
both make validation cover conceptual soundness, ongoing monitoring and outcomes
analysis. This report follows that structure.

## Questions I expect, and how I would answer

**"Walk me through your biggest finding."**
The model's strongest single input was `retention_call_flag`. On its own it
separates churners from non-churners at 0.87 AUC, when nothing else in the file
gets past 0.63. That gap is the signal. I checked the data dictionary: the field
is written when the retention team logs a call, and they call people who have
already given notice. So it is not a predictor, it is a record of the outcome,
and it will be empty when the model actually runs. Take it out and refit and AUC
goes from 0.905 to 0.666.

**"How did you know to look?"**
I did not know, I ran the test. Every input gets scored on its own against the
outcome and anything over 0.85 gets challenged on timing. It is a mechanical
check, which is the point -- it does not depend on me having a hunch about the
right variable.

**"What would you have done if the developer pushed back?"**
Asked for evidence of point-in-time availability: the timestamp the field is
written relative to the scoring date. That is a factual question with a factual
answer, and it is their answer to give. If they could show it is populated
before notice is given, the finding falls away and I would close it.

**"What is the weakest part of this work?"**
The portfolio is generated, so the magnitudes are mine rather than a bank's.
And the generator draws churn from a linear log-odds model, which advantages the
logistic benchmark in VF-02 by construction -- I say so in the report's
limitations and in `docs/planted_defects.md`. What transfers is the set of tests
and the thresholds, not the numbers.

**"Why is VF-02 High and not Low?"**
Because of which way the comparison went. If the champion had been marginally
ahead of the benchmark, that is a conversation about whether the complexity is
worth carrying, and it is Low. It was behind. A model that is less accurate than
its own benchmark on the period that matters has nothing left justifying the
monitoring and explainability burden it imposes, so it blocks use.

**"You rated the fairness finding Medium. Why not High?"**
Two things were tangled and I separated them. The selection rates genuinely
differ across age bands, and part of that is real -- the observed churn column
shows older customers do leave less. That alone is not a finding. What is a
finding is that the model reaches it through `product_bundle_code`, which
identifies over-60s at 0.947 AUC, while the submission states no protected
characteristic is used. So the issue is an inaccurate disclosure and an
unjustified proxy, not proven discrimination, and dropping the field costs
0.002 AUC. Medium with a named control, not High.

## What I would do next if this were real

Ask for the point-in-time evidence on the flagged variable, ask for the
monitoring pack, and agree the compensating controls in writing before the model
runs again. Then re-review after the refit rather than waiting for the annual
cycle, because the population is moving faster than the cycle is.
