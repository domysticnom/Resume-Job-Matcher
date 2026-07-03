# Baseline Evaluation Results

## Model

TF-IDF + Logistic Regression baseline model.

## Label Definition

- 0 = low match: matched_score < 0.50
- 1 = medium match: 0.50 <= matched_score < 0.75
- 2 = high match: matched_score >= 0.75

## Dataset

- Total rows after cleaning: 9544
- Train size: 7635
- Test size: 1909

## Accuracy

0.6647

## Precision, Recall, and F1-score

```text
              precision    recall  f1-score   support

   low_match     0.5376    0.7042    0.6098       355
medium_match     0.6960    0.5876    0.6372       822
  high_match     0.7147    0.7322    0.7233       732

    accuracy                         0.6647      1909
   macro avg     0.6494    0.6747    0.6568      1909
weighted avg     0.6737    0.6647    0.6651      1909
```

## Confusion Matrix

```text
[[250  78  27]
 [152 483 187]
 [ 63 133 536]]
```

## Notes

This is a baseline model. The labels are generated from the matched_score column in the resume dataset, so the results should be interpreted as a starting point for comparison rather than a final production-quality evaluation.
