# Domain-Adversarial Mental Health Classification with DistilRoBERTa

## Abstract

Mental health classification from user-generated text is an important natural
language processing task with applications in early risk detection, triage
support, and large-scale behavioral health analysis. However, models trained on
social media or forum data often learn dataset-specific signals that may not
generalize across domains, communities, or collection sources. This paper
presents a comparative study of two transformer-based approaches for mental
health classification using the Murarka et al. Mental Health Classification
Dataset [CITATION_MURARKA_DATASET]. The first approach fine-tunes
DistilRoBERTa as a supervised sequence classifier. The second augments
DistilRoBERTa with a Gradient Reversal Layer (GRL) and an auxiliary domain
classifier in an explicitly labeled proxy-domain adversarial training setup.
Both models are
trained using identical tokenization, optimization settings, and evaluation
splits. Performance is measured using accuracy, precision, recall, macro F1,
weighted F1, confusion matrices, and ROC AUC where defined. The proposed
framework provides a reproducible baseline and a domain-adversarial extension
for studying robust mental health text classification.

## Introduction

The increasing availability of online mental health discourse has motivated the
development of machine learning systems that classify textual signals related to
depression, anxiety, stress, suicidal ideation, and other mental health
conditions [CITATION_SURVEY_MENTAL_HEALTH_NLP]. Transformer language models have
substantially improved text classification performance by learning contextual
representations from large-scale corpora [CITATION_TRANSFORMERS]. Despite these
gains, mental health classification remains challenging because user language is
noisy, subjective, emotionally nuanced, and often domain dependent.

A model trained on one source may overfit lexical artifacts associated with a
particular platform, community, or annotation process. For example, a classifier
may learn topic markers or subreddit-specific language instead of mental health
signals that generalize across populations. This raises an important research
question: can domain-adversarial training improve the robustness of transformer
representations for mental health classification?

This work studies that question by comparing a DistilRoBERTa baseline with a
DistilRoBERTa model augmented by a Gradient Reversal Layer. The GRL model uses
one shared encoder and two heads: a mental health classification head and an
adversarial domain classification head. During backpropagation, the GRL reverses
the gradient from the domain classifier before it reaches the encoder, thereby
discouraging the shared representation from encoding domain-specific features
[CITATION_DANN].

The main contributions of this paper are:

1. A reproducible DistilRoBERTa baseline for mental health classification.
2. A modular DistilRoBERTa + GRL architecture for domain-adversarial learning.
3. A controlled experimental setup using identical hyperparameters across both
   models.
4. A publication-ready evaluation pipeline with metrics, tables, confusion
   matrices, ROC curves, and comparison figures.

## Related Work

### Mental Health Classification

Computational mental health research has used social media, forums, and other
textual data sources to identify signals associated with mental health
conditions [CITATION_DEPRESSION_SOCIAL_MEDIA], [CITATION_REDDIT_MENTAL_HEALTH].
Earlier systems often relied on bag-of-words features, lexicons, topic models,
and classical classifiers such as support vector machines or logistic
regression [CITATION_CLASSICAL_MH_NLP]. More recent work has adopted pretrained
language models, which better capture contextual meaning and improve downstream
classification accuracy [CITATION_BERT_MH].

### Transformer Models for Text Classification

Transformer encoders such as BERT and RoBERTa have become standard for text
classification due to their ability to model long-range dependencies and
contextual semantics [CITATION_BERT], [CITATION_ROBERTA]. DistilRoBERTa is a
compressed variant that preserves much of RoBERTa's representational capacity
while reducing computational cost [CITATION_DISTILROBERTA]. This makes it
appropriate for research settings where strong performance and practical
training efficiency are both important.

### Domain-Adversarial Learning

Domain-adversarial neural networks aim to learn representations that are useful
for a primary task but invariant to domain identity [CITATION_DANN]. A common
implementation uses a Gradient Reversal Layer, which behaves as the identity
function during the forward pass and multiplies gradients by a negative scalar
during the backward pass. This technique has been applied to domain adaptation
in computer vision and NLP [CITATION_DOMAIN_ADAPTATION_NLP]. In mental health
classification, domain-adversarial learning is especially relevant because
language distributions can vary substantially across platforms and user
communities.

## Dataset

This study uses the Murarka et al. Mental Health Classification Dataset
[CITATION_MURARKA_DATASET]. The dataset is provided in three CSV files:

- `both_train.csv`
- `both_val.csv`
- `both_test.csv`

Each example contains `title`, `post`, `class_name`, `class_id`, and `ID`
columns. The model input is built by safely concatenating `title` and `post`;
rows with empty combined text are removed. The target label is `class_name`.
Because the Murarka `both_*.csv` files do not contain a genuine source/domain
column, the GRL experiment uses text-length bins as a documented proxy-domain
strategy. The median word length is computed from the training split only and
then applied unchanged to validation and test. This should not be interpreted
as cross-dataset domain adaptation.

### Dataset Splits

| Split | File | Purpose | Number of Examples |
|---|---|---:|---:|
| Train | `both_train.csv` | Model optimization | TODO |
| Validation | `both_val.csv` | Model selection | TODO |
| Test | `both_test.csv` | Final evaluation | TODO |

### Label Distribution

| Label | Train | Validation | Test |
|---|---:|---:|---:|
| Class 1 | TODO | TODO | TODO |
| Class 2 | TODO | TODO | TODO |
| Class 3 | TODO | TODO | TODO |
| Class 4 | TODO | TODO | TODO |

## Methodology

Let an input text sequence be represented as:

```math
x = (w_1, w_2, \ldots, w_n)
```

where each \(w_i\) is a token. The tokenizer maps the sequence into token IDs
and an attention mask:

```math
(\mathbf{i}, \mathbf{m}) = \mathrm{Tokenizer}(x)
```

The model predicts a mental health label:

```math
\hat{y} = \arg\max_y p(y \mid x)
```

All models are optimized using cross-entropy loss for the primary
classification task:

```math
\mathcal{L}_{cls}
= -\sum_{c=1}^{C} y_c \log \hat{p}_c
```

where \(C\) is the number of mental health classes, \(y_c\) is the one-hot
ground-truth label, and \(\hat{p}_c\) is the predicted class probability.

## DistilRoBERTa

The baseline model uses `distilroberta-base` with a sequence-classification
head. Given token IDs and an attention mask, DistilRoBERTa produces contextual
hidden states:

```math
\mathbf{H} = \mathrm{DistilRoBERTa}(\mathbf{i}, \mathbf{m})
```

The first token representation is used as the pooled sequence feature:

```math
\mathbf{h} = \mathbf{H}_{[CLS]}
```

The classification head maps this feature to class logits:

```math
\mathbf{z}_{cls} = W_{cls}\mathbf{h} + \mathbf{b}_{cls}
```

The predicted probability distribution is:

```math
\hat{\mathbf{p}} = \mathrm{softmax}(\mathbf{z}_{cls})
```

The baseline objective is:

```math
\mathcal{L}_{baseline} = \mathcal{L}_{cls}
```

## DistilRoBERTa + GRL

The GRL model uses the same DistilRoBERTa encoder but adds an auxiliary domain
classification branch. The architecture is:

```text
Input
  |
DistilRoBERTa Encoder
  |
Shared Feature h
  |------------------------------|
  |                              |
Classification Head             Gradient Reversal Layer
  |                              |
Mental Health Prediction         Domain Classifier
```

The domain classifier predicts a domain label \(d\) from the shared feature:

```math
\mathbf{z}_{dom} = W_{dom}\mathrm{GRL}(\mathbf{h}) + \mathbf{b}_{dom}
```

The Gradient Reversal Layer is defined as:

```math
\mathrm{GRL}_{\lambda}(\mathbf{h}) = \mathbf{h}
```

during the forward pass, and:

```math
\frac{\partial \mathrm{GRL}_{\lambda}}{\partial \mathbf{h}}
= -\lambda \mathbf{I}
```

during the backward pass. In this study:

```math
\lambda = 0.1
```

The domain loss is:

```math
\mathcal{L}_{dom}
= -\sum_{k=1}^{K} d_k \log \hat{q}_k
```

where \(K\) is the number of domains and \(\hat{q}_k\) is the predicted domain
probability. The final GRL objective is:

```math
\mathcal{L}_{total}
= \mathcal{L}_{cls} + 0.1 \times \mathcal{L}_{dom}
```

Because the domain gradient is reversed before reaching the encoder, the shared
feature extractor is trained to reduce classification loss while suppressing
domain-specific information.

## Experimental Setup

Both models use the same tokenizer, data splits, optimizer, scheduler, and
training hyperparameters.

| Component | Value |
|---|---|
| Backbone | `distilroberta-base` |
| Tokenizer | `AutoTokenizer` |
| Maximum sequence length | 128 |
| Batch size | 16 |
| Epochs | 3 |
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Scheduler | Linear warmup |
| Primary loss | Cross-entropy |
| Model selection | Best validation macro F1 |
| Device | GPU if available, otherwise CPU |
| GRL lambda | 0.1 |
| GRL domain loss weight | 0.1 |

### Evaluation Metrics

The following metrics are computed on the held-out test split:

| Metric | Definition |
|---|---|
| Accuracy | Fraction of correctly classified examples |
| Precision | Weighted precision across classes |
| Recall | Weighted recall across classes |
| Macro F1 | Unweighted mean of per-class F1 scores |
| Weighted F1 | Support-weighted mean of per-class F1 scores |
| ROC AUC | Binary ROC AUC or multiclass one-vs-rest ROC AUC when defined |
| Confusion Matrix | Class-wise prediction error matrix |

Macro F1 is emphasized because mental health datasets are commonly imbalanced,
and macro averaging gives equal importance to each class.

## Results

### Main Comparison

| Model | Accuracy | Precision | Recall | Macro F1 | Weighted F1 | ROC AUC |
|---|---:|---:|---:|---:|---:|---:|
| DistilRoBERTa | TODO | TODO | TODO | TODO | TODO | TODO |
| DistilRoBERTa + GRL | TODO | TODO | TODO | TODO | TODO | TODO |

### Per-Class Classification Report: DistilRoBERTa

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Class 1 | TODO | TODO | TODO | TODO |
| Class 2 | TODO | TODO | TODO | TODO |
| Class 3 | TODO | TODO | TODO | TODO |
| Class 4 | TODO | TODO | TODO | TODO |

### Per-Class Classification Report: DistilRoBERTa + GRL

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Class 1 | TODO | TODO | TODO | TODO |
| Class 2 | TODO | TODO | TODO | TODO |
| Class 3 | TODO | TODO | TODO | TODO |
| Class 4 | TODO | TODO | TODO | TODO |

### Figures

The experimental pipeline generates the following publication-quality figures:

| Figure | File |
|---|---|
| Training loss curve | `outputs/figures/training_loss_curve.png` |
| Validation loss curve | `outputs/figures/validation_loss_curve.png` |
| Accuracy curve | `outputs/figures/accuracy_curve.png` |
| Macro F1 comparison | `outputs/figures/macro_f1_comparison.png` |
| Baseline confusion matrix | `outputs/figures/baseline_confusion_matrix.png` |
| GRL confusion matrix | `outputs/figures/grl_confusion_matrix.png` |
| ROC curve | `outputs/figures/roc_curve.png` |

## Discussion

The DistilRoBERTa baseline provides a strong supervised reference point because
it directly optimizes mental health classification performance using contextual
representations from a pretrained transformer. Its main advantage is simplicity:
the model has a single objective and directly aligns representation learning
with the target labels.

The GRL model introduces a second learning signal through the domain classifier.
This adversarial objective may improve robustness when domain labels capture
meaningful source variation. If the GRL model improves macro F1 or reduces
domain-specific error patterns, this suggests that the shared encoder has
learned more transferable mental health features. Conversely, if GRL degrades
performance, possible explanations include weak or noisy domain labels,
insufficient domain shift, an overly strong adversarial objective, or conflict
between domain invariance and class-discriminative information.

Confusion matrices are especially important for interpreting mental health
classification models. Aggregate metrics can hide systematic errors, such as
frequent confusion between semantically related conditions. ROC curves provide
an additional view of ranking behavior, although they should be interpreted
carefully in multiclass and imbalanced settings.

## Conclusion

This paper presents a reproducible framework for mental health classification
using DistilRoBERTa and a domain-adversarial DistilRoBERTa + GRL extension. The
baseline model fine-tunes a pretrained transformer for supervised
classification, while the GRL model encourages the shared encoder to learn
features that are predictive of mental health labels but less predictive of
domain identity. The project includes a complete training, evaluation, and
figure-generation pipeline suitable for research experimentation and reporting.

## Future Work

Future work should evaluate the GRL model using explicit and semantically
meaningful domain annotations, such as platform, subreddit, collection source,
or demographic grouping where ethically appropriate. Additional extensions
include:

1. Testing larger transformer backbones such as RoBERTa-base and DeBERTa.
2. Exploring class-balanced loss functions for imbalanced mental health labels.
3. Performing hyperparameter sweeps over GRL lambda and domain loss weight.
4. Evaluating calibration and uncertainty estimates for safer deployment.
5. Conducting external validation on datasets from different platforms.
6. Adding explainability methods such as attention analysis or attribution
   scores.
7. Studying fairness and subgroup robustness when reliable metadata is
   available and ethically permissible.

## References

[CITATION_MURARKA_DATASET] Murarka et al. Mental Health Classification Dataset.
Full bibliographic details to be added.

[CITATION_SURVEY_MENTAL_HEALTH_NLP] Placeholder for a survey on NLP for mental
health.

[CITATION_TRANSFORMERS] Placeholder for the Transformer architecture citation.

[CITATION_BERT] Placeholder for BERT citation.

[CITATION_ROBERTA] Placeholder for RoBERTa citation.

[CITATION_DISTILROBERTA] Placeholder for DistilRoBERTa citation.

[CITATION_DANN] Placeholder for Domain-Adversarial Neural Networks and Gradient
Reversal Layer citation.

[CITATION_DOMAIN_ADAPTATION_NLP] Placeholder for domain adaptation in NLP.

[CITATION_DEPRESSION_SOCIAL_MEDIA] Placeholder for depression detection from
social media.

[CITATION_REDDIT_MENTAL_HEALTH] Placeholder for Reddit mental health analysis.

[CITATION_CLASSICAL_MH_NLP] Placeholder for classical machine learning in mental
health NLP.

[CITATION_BERT_MH] Placeholder for BERT-based mental health classification.
