import gc
import re
import itertools
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

try:
    from google.colab import drive 
    IN_COLAB = True
except Exception:
    IN_COLAB = False

if IN_COLAB:
    drive.mount("/content/drive")

comment_df = pd.read_csv("/content/drive/MyDrive/comments.csv", encoding="latin-1")
#This dataset contains the following key variables:
#      - comment_id
#      - comment
#      - incivility
sentence_df = pd.read_csv("/content/drive/MyDrive/sentences.csv", encoding="latin-1")
#This dataset contains the following key variables:
#      - comment_id
#      - sentence_id
#      - comment
#      - sentence
#      - incivility

print("comment_df:", comment_df.shape)
#output: comment_df: (7941, 4)
print("sentence_df:", sentence_df.shape)
#output: sentence_df: (23840, 5)

def canonicalize_comment_id(x):
    """
    Ensuring that comment_id values comparable across comment_df, sentence_df,
    DataLoader batches, and prediction DataFrames.
    """
    if x is None:
        return None

    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            x = x.detach().cpu().item()
        else:
            x = x.detach().cpu().tolist()

    if isinstance(x, np.generic):
        x = x.item()

    if pd.isna(x):
        return None

    s = str(x).strip()

    tensor_match = re.fullmatch(r"tensor\(([-+]?[0-9]+(?:\.0)?)\)", s)
    if tensor_match:
        s = tensor_match.group(1)

    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except ValueError:
        pass

    return re.sub(r"\.0$", "", s)

comment_df["comment_id"] = comment_df["comment_id"].apply(canonicalize_comment_id)
sentence_df["comment_id"] = sentence_df["comment_id"].apply(canonicalize_comment_id)

comment_ids = set(comment_df["comment_id"])
sentence_ids = set(sentence_df["comment_id"])
overlap_ids = comment_ids & sentence_ids

print("unique comments in comment_df:", len(comment_ids))
#output: unique comments in comment_df: 7941
print("unique comments in sentence_df:", len(sentence_ids))
#output: unique comments in sentence_df: 7942
print("overlapping comment_ids:", len(overlap_ids))
#output: overlapping comment_ids: 7941

#Ensuring the correct comment_ids are recorded, as this was where the code previously failed
print("example comment_df ids:")
#output: example comment_df ids:
print(comment_df["comment_id"].head())
#output: 0    5272
#1    5982
#2    5759
#3    5401
#4    5924
#Name: comment_id, dtype: object

print("example sentence_df ids:")
#ouput: example sentence_df ids:
print(sentence_df["comment_id"].head())
#output: 0    5272
#1    5982
#2    5759
#3    5401
#4    5924
#Name: comment_id, dtype: object

comment_df = pd.read_csv(COMMENTS_CSV_PATH, encoding="latin-1")
sentence_df = pd.read_csv(SENTENCES_CSV_PATH, encoding="latin-1")

print("comment_df:", comment_df.shape)
#output: comment_df: (7941, 4)
print("sentence_df:", sentence_df.shape)
#output: sentence_df: (23840, 5)

required_comment_cols = {"comment_id", "Comment", "incivility"}
required_sentence_cols = {"comment_id", "sentence", "incivility"}

missing_comment_cols = required_comment_cols - set(comment_df.columns)
missing_sentence_cols = required_sentence_cols - set(sentence_df.columns)

if missing_comment_cols:
    raise ValueError(f"comment_df is missing required columns: {missing_comment_cols}")
if missing_sentence_cols:
    raise ValueError(f"sentence_df is missing required columns: {missing_sentence_cols}")

if "sentence_id" not in sentence_df.columns:
    sentence_df["sentence_id"] = sentence_df.groupby("comment_id").cumcount().astype(int)
else:
    sentence_df["sentence_id"] = sentence_df["sentence_id"].astype(str)

print("comment cols ok:", required_comment_cols.issubset(comment_df.columns))
#output: comment cols ok: True
print("sentence cols ok:", required_sentence_cols.issubset(sentence_df.columns))
#output: sentence cols ok: True

BASE_OUTPUT_DIR = Path("/content/drive/MyDrive/outputs").expanduser()
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "vinai/bertweet-large"

def seed_everything(seed: int) -> None:
    """Seeding Python, NumPy, and PyTorch to ensure a deterministic run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Seeding DataLoader workers, if num_workers > 0 is used later."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def make_splits(
    split_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Building the train/validation/test splits for a given seed such that the training data i 70%, validation data is 15% and the test data is 15%. Also, this prevents leakage as sentences from the same comment cannot appear in both the training and test data.
    """
    train_comments, tmp_comments = train_test_split(
        comment_df,
        test_size=0.30,
        random_state=split_seed,
        stratify=comment_df["incivility"],
    )

    val_comments, test_comments = train_test_split(
        tmp_comments,
        test_size=0.50,
        random_state=split_seed,
        stratify=tmp_comments["incivility"],
    )

    train_ids = set(train_comments["comment_id"])
    val_ids = set(val_comments["comment_id"])
    test_ids = set(test_comments["comment_id"])

    train_sentences = sentence_df[sentence_df["comment_id"].isin(train_ids)].copy()
    val_sentences = sentence_df[sentence_df["comment_id"].isin(val_ids)].copy()
    test_sentences = sentence_df[sentence_df["comment_id"].isin(test_ids)].copy()

    return train_comments.copy(), val_comments.copy(), test_comments.copy(), train_sentences, val_sentences, test_sentences

lass SentenceOnlyDataset(Dataset):
    """Sentence-level dataset that keeps parent comment_id for the max-one-rule aggregation."""

    def __init__(self, df: pd.DataFrame, tokenizer, text_col: str = "sentence", label_col: str = "incivility"):
        self.texts = df[text_col].astype(str).tolist()
        self.labels = df[label_col].astype(int).tolist()
        self.comment_ids = (
            df["comment_id"]
            .apply(canonicalize_comment_id)
            .astype(str)
            .tolist()
        )
        self.sentence_ids = df["sentence_id"].astype(str).tolist()
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        text = self.texts[idx]
        label = self.labels[idx]

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_SENTENCE_LEN,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
            "comment_id": self.comment_ids[idx],
            "sentence_id": self.sentence_ids[idx],
        }

class SentenceBertClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        num_labels: int = 2,
        dropout_prob: float = 0.1,
        gradient_checkpointing: bool = GRADIENT_CHECKPOINTING,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

        if gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()

        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

  def compute_binary_metrics(y_true: List[int], y_pred: List[int], prefix: str = "") -> Dict[str, float]:
    """Returning the exact metric set requested for binary labels 0 and 1 (update if working with a non-binary measure of incivility)"""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    precision_by_class, recall_by_class, f1_by_class, _ = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=[0, 1],
        average=None,
        zero_division=0,
    )

    p_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=[0, 1],
        average="weighted",
        zero_division=0,
    )

    p_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=[0, 1],
        average="macro",
        zero_division=0,
    )

    return {
        f"{prefix}accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        f"{prefix}precision_0": float(precision_by_class[0]),
        f"{prefix}precision_1": float(precision_by_class[1]),
        f"{prefix}recall_0": float(recall_by_class[0]),
        f"{prefix}recall_1": float(recall_by_class[1]),
        f"{prefix}f1_0": float(f1_by_class[0]),
        f"{prefix}f1_1": float(f1_by_class[1]),
        f"{prefix}p_weighted": float(p_weighted),
        f"{prefix}p_macro": float(p_macro),
        f"{prefix}recall_weighted": float(recall_weighted),
        f"{prefix}recall_macro": float(recall_macro),
        f"{prefix}f1_weighted": float(f1_weighted),
        f"{prefix}f1_macro": float(f1_macro),
    }

def make_class_weights(y: np.ndarray, num_classes: int = 2) -> torch.Tensor:
    """Balancing class weights, robust to edge cases where a class is absent."""
    y_arr = np.asarray(y, dtype=int)
    weights = np.ones(num_classes, dtype=np.float32)
    present_classes = np.unique(y_arr)

    if len(present_classes) > 0:
        computed = compute_class_weight(class_weight="balanced", classes=present_classes, y=y_arr)
        for cls, weight in zip(present_classes, computed):
            if 0 <= int(cls) < num_classes:
                weights[int(cls)] = float(weight)

    return torch.tensor(weights, dtype=torch.float)


@dataclass
class TrainConfig:
    device: torch.device
    class_weights: torch.Tensor
    label_smoothing: float
    use_amp: bool = True


def create_optimizer_and_scheduler(
    model: nn.Module,
    total_steps: int,
    lr: float,
    weight_decay: float,
    warmup_proportion: float,
):
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    decay_params, no_decay_params = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
    )

    warmup_steps = int(warmup_proportion * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler

def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer,
    scheduler,
    config: TrainConfig,
) -> float:
    model.train()
    device = config.device
    use_amp = config.use_amp and device.type == "cuda"

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    running_loss = 0.0
    n_batches = 0

    for batch in data_loader:
        optimizer.zero_grad(set_to_none=True)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(
                logits,
                labels,
                weight=config.class_weights.to(device),
                label_smoothing=config.label_smoothing,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += float(loss.detach().cpu())
        n_batches += 1

    return running_loss / max(n_batches, 1)


@torch.no_grad()
def predict_sentence_dataframe(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    """Predicting sentence labels and keep comment_id for max-one-rule aggregation."""
    model.eval()

    rows = []
    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].cpu().numpy().astype(int).tolist()

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs_uncivil = F.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
        preds = (probs_uncivil >= 0.5).astype(int)

        comment_ids = [str(x) for x in batch["comment_id"]]
        sentence_ids = [str(x) for x in batch["sentence_id"]]

        for cid, sid, y, pred, prob in zip(comment_ids, sentence_ids, labels, preds, probs_uncivil):
            rows.append(
                {
                    "comment_id": cid,
                    "sentence_id": sid,
                    "sentence_label": int(y),
                    "sentence_pred": int(pred),
                    "sentence_prob_uncivil": float(prob),
                }
            )

    return pd.DataFrame(rows)


def max_rule_from_sentence_predictions(
    sentence_predictions: pd.DataFrame,
    comments: pd.DataFrame,
    prefix: str,
    allow_missing_sentences: bool = ALLOW_MISSING_SENTENCES,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Aggregating sentence predictions to comment predictions by the max-one rule."""
    if sentence_predictions.empty:
        raise ValueError("No sentence predictions available for max-one-rule aggregation.")

    agg = (
        sentence_predictions.groupby("comment_id", as_index=False)
        .agg(
            comment_pred=("sentence_pred", "max"),
            comment_prob_uncivil=("sentence_prob_uncivil", "max"),
            n_sentences=("sentence_pred", "size"),
            n_predicted_uncivil_sentences=("sentence_pred", "sum"),
        )
    )

    gold = comments[["comment_id", "incivility"]].copy()
    gold["comment_id"] = gold["comment_id"].astype(str)
    gold["comment_label"] = gold["incivility"].astype(int)
    gold = gold.drop(columns=["incivility"])

    eval_df = gold.merge(agg, on="comment_id", how="left")
    missing_mask = eval_df["comment_pred"].isna()
    n_missing = int(missing_mask.sum())

    if n_missing > 0 and not allow_missing_sentences:
        missing_examples = eval_df.loc[missing_mask, "comment_id"].head(10).tolist()
        raise ValueError(
            f"{n_missing} comments have no sentence rows, so max-rule evaluation is undefined. "
            f"Examples: {missing_examples}. If you really want to count these as civil, "
            f"set ALLOW_MISSING_SENTENCES = True."
        )

    if n_missing > 0 and allow_missing_sentences:
        eval_df.loc[missing_mask, "comment_pred"] = 0
        eval_df.loc[missing_mask, "comment_prob_uncivil"] = 0.0
        eval_df.loc[missing_mask, "n_sentences"] = 0
        eval_df.loc[missing_mask, "n_predicted_uncivil_sentences"] = 0

    eval_df["comment_pred"] = eval_df["comment_pred"].astype(int)
    eval_df["comment_prob_uncivil"] = eval_df["comment_prob_uncivil"].astype(float)
    eval_df["n_sentences"] = eval_df["n_sentences"].astype(int)
    eval_df["n_predicted_uncivil_sentences"] = eval_df["n_predicted_uncivil_sentences"].astype(int)

    metrics = compute_binary_metrics(
        y_true=eval_df["comment_label"].astype(int).tolist(),
        y_pred=eval_df["comment_pred"].astype(int).tolist(),
        prefix=prefix,
    )
    metrics[f"{prefix}n_comments"] = int(len(eval_df))
    metrics[f"{prefix}n_missing_sentence_comments"] = int(n_missing)
    metrics[f"{prefix}mean_sentences_per_comment"] = float(eval_df["n_sentences"].mean())
    metrics[f"{prefix}share_predicted_uncivil"] = float(eval_df["comment_pred"].mean())

    return metrics, eval_df


def sentence_metrics_from_predictions(sentence_predictions: pd.DataFrame, prefix: str) -> Dict[str, float]:
    metrics = compute_binary_metrics(
        y_true=sentence_predictions["sentence_label"].astype(int).tolist(),
        y_pred=sentence_predictions["sentence_pred"].astype(int).tolist(),
        prefix=prefix,
    )
    metrics[f"{prefix}n_sentences"] = int(len(sentence_predictions))
    metrics[f"{prefix}share_predicted_uncivil"] = float(sentence_predictions["sentence_pred"].mean())
    return metrics

def make_sentence_loader(
    df: pd.DataFrame,
    tokenizer,
    batch_size: int,
    shuffle: bool,
    seed: Optional[int] = None,
) -> DataLoader:
    dataset = SentenceOnlyDataset(df, tokenizer=tokenizer)

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


def train_sentence_only_model(
    train_comments: pd.DataFrame,
    val_comments: pd.DataFrame,
    test_comments: pd.DataFrame,
    train_sentences: pd.DataFrame,
    val_sentences: pd.DataFrame,
    test_sentences: pd.DataFrame,
    seed: int,
    run_index: int,
    run_tag: str,
) -> Tuple[nn.Module, AutoTokenizer, pd.DataFrame, Dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Training on sentences and evaluate comments by max-one-rule."""
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    train_loader = make_sentence_loader(
        train_sentences,
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=seed,
    )
    train_eval_loader = make_sentence_loader(
        train_sentences,
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    val_loader = make_sentence_loader(
        val_sentences,
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_loader = make_sentence_loader(
        test_sentences,
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = SentenceBertClassifier(model_name=MODEL_NAME)
    model.to(device)

    class_weights = make_class_weights(train_sentences["incivility"].values)
    total_steps = EPOCHS * len(train_loader)

    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        total_steps=total_steps,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_proportion=WARMUP_PROPORTION,
    )

    train_cfg = TrainConfig(
        device=device,
        class_weights=class_weights,
        label_smoothing=LABEL_SMOOTHING,
        use_amp=True,
    )

    best_val_score = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    best_state_dict = None
    epoch_rows = []

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, train_cfg)

        # Sentence-level predictions.
        train_sentence_pred = predict_sentence_dataframe(model, train_eval_loader, device)
        val_sentence_pred = predict_sentence_dataframe(model, val_loader, device)

        # Direct sentence-level metrics.
        train_sentence_metrics = sentence_metrics_from_predictions(train_sentence_pred, prefix="train_sentence_")
        val_sentence_metrics = sentence_metrics_from_predictions(val_sentence_pred, prefix="val_sentence_")

        # Comment-level max-one-rule metrics.
        train_comment_metrics, _ = max_rule_from_sentence_predictions(
            train_sentence_pred,
            comments=train_comments,
            prefix="train_comment_",
        )
        val_comment_metrics, _ = max_rule_from_sentence_predictions(
            val_sentence_pred,
            comments=val_comments,
            prefix="val_comment_",
        )

        row = {
            "run_index": int(run_index),
            "run_tag": run_tag,
            "RNG_SEED": int(seed),
            "MODEL_NAME": MODEL_NAME,
            "MAX_SENTENCE_LEN": int(MAX_SENTENCE_LEN),
            "BATCH_SIZE": int(BATCH_SIZE),
            "LR": float(LR),
            "WEIGHT_DECAY": float(WEIGHT_DECAY),
            "WARMUP_PROPORTION": float(WARMUP_PROPORTION),
            "LABEL_SMOOTHING": float(LABEL_SMOOTHING),
            "epoch": int(epoch),
            "train_loss": float(train_loss),
        }
        row.update(train_sentence_metrics)
        row.update(train_comment_metrics)
        row.update(val_sentence_metrics)
        row.update(val_comment_metrics)

        val_score = float(row[EARLY_STOP_METRIC])
        row["val_score"] = val_score
        row["early_stop_metric"] = EARLY_STOP_METRIC
        epoch_rows.append(row)

        print(
            f"train_loss={train_loss:.5f} | "
            f"val_comment_accuracy={row['val_comment_accuracy']:.4f} | "
            f"val_comment_f1_weighted={row['val_comment_f1_weighted']:.4f} | "
            f"val_comment_f1_macro={row['val_comment_f1_macro']:.4f}"
        )

        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            epochs_no_improve = 0
            best_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
      
    test_sentence_pred = predict_sentence_dataframe(model, test_loader, device)
    test_sentence_metrics = sentence_metrics_from_predictions(test_sentence_pred, prefix="test_sentence_")
    test_comment_metrics, test_comment_pred = max_rule_from_sentence_predictions(
        test_sentence_pred,
        comments=test_comments,
        prefix="test_comment_",
    )

    test_metrics = {
        "run_index": int(run_index),
        "run_tag": run_tag,
        "RNG_SEED": int(seed),
        "MODEL_NAME": MODEL_NAME,
        "MAX_SENTENCE_LEN": int(MAX_SENTENCE_LEN),
        "BATCH_SIZE": int(BATCH_SIZE),
        "LR": float(LR),
        "WEIGHT_DECAY": float(WEIGHT_DECAY),
        "WARMUP_PROPORTION": float(WARMUP_PROPORTION),
        "LABEL_SMOOTHING": float(LABEL_SMOOTHING),
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_val_score),
        "early_stop_metric": EARLY_STOP_METRIC,
    }
    test_metrics.update(test_sentence_metrics)
    test_metrics.update(test_comment_metrics)

    epoch_df = pd.DataFrame(epoch_rows)
    return model, tokenizer, epoch_df, test_metrics, test_sentence_pred, test_comment_pred

SEEDS = [40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55]

#This example uses bertweet and this grid therefore takes the best performing hyperparameters for the bertweet model; update if/when another model is used.
grid = {
    "RNG_SEED": SEEDS,
    "LR": [1e-5],
    "BATCH_SIZE": [16],
    "WEIGHT_DECAY": [0.01],
    "WARMUP_PROPORTION": [0.01],
    "LABEL_SMOOTHING": [0.01]
}

EPOCHS_SEARCH = 30
PATIENCE_SEARCH = 30 #The training did not include early stopping, set to <30 if early stopping is desired.
EARLY_STOP_METRIC = "f1_weighted"

RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
EPOCH_RESULTS_CSV = BASE_OUTPUT_DIR / f"sentence_maxrule_bertweet_epoch_metrics_{RUN_TAG}.csv"
TEST_RESULTS_CSV = BASE_OUTPUT_DIR / f"sentence_maxrule_bertweet_test_metrics_{RUN_TAG}.csv"
TEST_SENTENCE_PREDS_CSV = BASE_OUTPUT_DIR / f"sentence_maxrule_bertweet_test_sentence_predictions_{RUN_TAG}.csv"
TEST_COMMENT_PREDS_CSV = BASE_OUTPUT_DIR / f"sentence_maxrule_bertweet_test_comment_predictions_{RUN_TAG}.csv"

keys = list(grid.keys())
configs = list(itertools.product(*[grid[k] for k in keys]))

print("Total configs:", len(configs))
print("Seeds:", SEEDS)
print("Epochs:", EPOCHS)
print("Patience:", PATIENCE)
print("Early-stop metric:", EARLY_STOP_METRIC)

all_epoch_results = []
all_test_results = []
all_test_sentence_predictions = []
all_test_comment_predictions = []

best = {"score": -1.0, "cfg": None}

for run_index, values in enumerate(configs, start=1):
    cfg = dict(zip(keys, values))

    print("\n" + "=" * 80)
    print(f"Run {run_index}/{len(configs)} | cfg = {cfg}")

    # Override globals read by training functions.
    RNG_SEED = int(cfg["RNG_SEED"])
    LR = float(cfg["LR"])
    BATCH_SIZE = int(cfg["BATCH_SIZE"])
    WEIGHT_DECAY = float(cfg["WEIGHT_DECAY"])
    WARMUP_PROPORTION = float(cfg["WARMUP_PROPORTION"])
    LABEL_SMOOTHING = float(cfg["LABEL_SMOOTHING"])

    train_comments, val_comments, test_comments, train_sentences, val_sentences, test_sentences = make_splits(RNG_SEED)

    print("train/val/test comments:", len(train_comments), len(val_comments), len(test_comments))
    print("train/val/test sentences:", len(train_sentences), len(val_sentences), len(test_sentences))

    model, tokenizer, epoch_df, test_metrics, test_sentence_pred, test_comment_pred = train_sentence_only_model(
        train_comments=train_comments,
        val_comments=val_comments,
        test_comments=test_comments,
        train_sentences=train_sentences,
        val_sentences=val_sentences,
        test_sentences=test_sentences,
        seed=RNG_SEED,
        run_index=run_index,
        run_tag=RUN_TAG,
    )

    all_epoch_results.append(epoch_df)
    all_test_results.append(test_metrics)

    test_sentence_pred = test_sentence_pred.assign(run_index=run_index, run_tag=RUN_TAG, RNG_SEED=RNG_SEED)
    test_comment_pred = test_comment_pred.assign(run_index=run_index, run_tag=RUN_TAG, RNG_SEED=RNG_SEED)
    all_test_sentence_predictions.append(test_sentence_pred)
    all_test_comment_predictions.append(test_comment_pred)

    if test_metrics["best_val_score"] > best["score"]:
        best = {"score": float(test_metrics["best_val_score"]), "cfg": cfg}
        print(">>> NEW BEST BY VALIDATION:", best)

    # Save after every run, so partial results survive interruptions.
    pd.concat(all_epoch_results, ignore_index=True).to_csv(EPOCH_RESULTS_CSV, index=False)
    pd.DataFrame(all_test_results).to_csv(TEST_RESULTS_CSV, index=False)

    if SAVE_TEST_PREDICTIONS:
        pd.concat(all_test_sentence_predictions, ignore_index=True).to_csv(TEST_SENTENCE_PREDS_CSV, index=False)
        pd.concat(all_test_comment_predictions, ignore_index=True).to_csv(TEST_COMMENT_PREDS_CSV, index=False)

    # Cleanup between runs.
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


epoch_results_df = pd.concat(all_epoch_results, ignore_index=True)
test_results_df = pd.DataFrame(all_test_results)

print("\nSaved epoch-level metrics to:", EPOCH_RESULTS_CSV)
print("Saved held-out test metrics to:", TEST_RESULTS_CSV)

if SAVE_TEST_PREDICTIONS:
    print("Saved test sentence predictions to:", TEST_SENTENCE_PREDS_CSV)
    print("Saved test comment max-rule predictions to:", TEST_COMMENT_PREDS_CSV)

print("\nTest comment-level max-rule results:")
cols_to_show = [
    "RNG_SEED",
    "best_epoch",
    "best_val_score",
    "test_comment_accuracy",
    "test_comment_precision_0",
    "test_comment_precision_1",
    "test_comment_recall_0",
    "test_comment_recall_1",
    "test_comment_f1_0",
    "test_comment_f1_1",
    "test_comment_p_weighted",
    "test_comment_p_macro",
    "test_comment_recall_weighted",
    "test_comment_recall_macro",
    "test_comment_f1_weighted",
    "test_comment_f1_macro",
]
print(test_results_df[cols_to_show])

print("\nMean test comment metrics across seeds:")
metric_cols = [c for c in test_results_df.columns if c.startswith("test_comment_") and test_results_df[c].dtype != object]
print(test_results_df[metric_cols].mean(numeric_only=True).sort_index())

print("\nBest validation config:", best["cfg"])
print("Best validation score:", best["score"])
