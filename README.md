# Joint-sentence-comment-model
Requires: > Python.3.10

**Dependencies**:
- numpy
- pandas
- math
- dataclasses
- torch
- sklearn
- transformers

**To run the pure-comment model**:
ALPHA_SENTENCE_LOSS = 0  

**To run the joint comment-sentence model**:
ALPHA_SENTENCE_LOSS = 1

**Note**: For legal reasons the models can be made public at this time. The annotated data will be made public upon publication of the main article.

# Repository Structure
```text
.
├── code/
│   └── Plot_creation.py                        # Creation of figures 2, 3, and 4 with Eval_epoch.csv
│   ├── Pure_joint.py                           # Running pure-comment and joint sentence-comment classification schemes
├── data/
│   ├── Eval_epoch.csv                          # Evaluation metrics over epochs (on validation data)
│   ├── test_metrics                            # Evaluation metrics of best performing model on held-out test set
├── plots/
│   ├── BERT.png                                # Figure 2
│   ├── RoBERTa.png                             # Figure 3
│   └── BERTweet.png                            # Figure 4
└── README.md
