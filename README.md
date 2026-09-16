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
ALPHA_SENTENCE_LOSS > 0

**Note**: For legal reasons the models can be made public at this time. The annotated data will be made public upon publication of the main article.

# Repository Structure
```text
.
├── code/
│   └── Pure_sentence.py                        # Rinning pure-sentence classification scheme
│   ├── Pure_comment_and_joint.py               # Running pure-comment and joint sentence-comment classification schemes
├── data/
│   ├── test_metrics_pure_comment_joint.csv     # Test metrics of best performing model on a held-out test set for the pure-comment and joint sentence-comment classification schemes
│   ├── test_metrics_pure_sentence.csv          # Test metrics of best performing model on a held-out test set for the pure-sentence classification scheme
└── README.md
