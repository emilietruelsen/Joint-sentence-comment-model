import gc
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
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
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
    drive.mount('/content/drive')

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

BASE_OUTPUT_DIR = Path("/content/drive/MyDrive/outputs").expanduser() #Outputs directory for evaluation metrics
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "roberta-large"

required_comment_cols = {"comment_id", "Comment", "incivility"}
required_sentence_cols = {"comment_id", "sentence", "incivility"}

missing_comment_cols = required_comment_cols - set(comment_df.columns)
missing_sentence_cols = required_sentence_cols - set(sentence_df.columns)

if missing_comment_cols:
    raise ValueError(f"comment_df is missing required columns: {missing_comment_cols}")
if missing_sentence_cols:
    raise ValueError(f"sentence_df is missing required columns: {missing_sentence_cols}")

print("comment cols ok:", required_comment_cols.issubset(comment_df.columns))
print("sentence cols ok:", required_sentence_cols.issubset(sentence_df.columns))

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
    """Building the train/validation/test splits for a given seed such that the training data i 70%, validation data is 15% and the test data is 15%."""
    df_train, df_tmp = train_test_split(
        comment_df,
        test_size=0.30,
        random_state=split_seed,
        stratify=comment_df["incivility"],
    )

    val_comments, test_comments = train_test_split(
        df_tmp,
        test_size=0.50,
        random_state=split_seed,
        stratify=df_tmp["incivility"],
    )

    train_comment_ids = set(df_train["comment_id"])
    val_comment_ids = set(val_comments["comment_id"])
    test_comment_ids = set(test_comments["comment_id"])

    train_sentences = sentence_df[sentence_df["comment_id"].isin(train_comment_ids)].copy()
    val_sentences = sentence_df[sentence_df["comment_id"].isin(val_comment_ids)].copy()
    test_sentences = sentence_df[sentence_df["comment_id"].isin(test_comment_ids)].copy()

    return df_train, val_comments, test_comments, train_sentences, val_sentences, test_sentences

def compute_binary_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """Returning the exact metric set requested for binary labels 0 and 1 (update if working with a non-binary measure of incivility)"""
    precision, recall, f1, _support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_0": float(precision[0]),
        "precision_1": float(precision[1]),
        "recall_0": float(recall[0]),
        "recall_1": float(recall[1]),
        "f1_0": float(f1[0]),
        "f1_1": float(f1[1]),
        "p_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "p_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def prefix_metrics(metrics: Dict[str, float], prefix: str) -> Dict[str, float]:
    return {f"{prefix}_{k}": float(v) for k, v in metrics.items()}

def build_sentence_map(
    sentences: pd.DataFrame,
    sentence_text_col: str,
    sentence_label_col: str,
) -> Dict[object, List[Tuple[object, str, int]]]:
    """Mapping comment_id between comments and sentences -> ordered list of (sentence_id, sentence_text, sentence_label)."""
    sort_cols = ["comment_id"]
    if "sentence_id" in sentences.columns:
        sort_cols.append("sentence_id")
        sentences = sentences.sort_values(sort_cols)

    sentence_map: Dict[object, List[Tuple[object, str, int]]] = {}
    for _, row in sentences.iterrows():
        cid = row["comment_id"]
        sentence_id = row["sentence_id"] if "sentence_id" in sentences.columns else len(sentence_map.get(cid, []))
        text = str(row[sentence_text_col])
        label = int(row[sentence_label_col])
        sentence_map.setdefault(cid, []).append((sentence_id, text, label))
    return sentence_map


def find_sentence_char_spans(
    comment_text: str,
    linked_sentences: List[Tuple[object, str, int]],
) -> Tuple[List[Tuple[object, int, int, int]], int]:
    """
    Locating sentence strings inside the full comment.

    Returns:
        spans: list of (sentence_id, start_char, end_char, label)
        skipped: number of linked sentences whose text could not be matched
    """
    spans: List[Tuple[object, int, int, int]] = []
    skipped = 0
    cursor = 0
    lower_comment = comment_text.lower()

    for sentence_id, sentence_text, label in linked_sentences:
        sent = str(sentence_text)
        if not sent.strip():
            skipped += 1
            continue

        start = comment_text.find(sent, cursor)
        if start < 0:
            start = lower_comment.find(sent.lower(), cursor)
        if start < 0:
            start = comment_text.find(sent)
        if start < 0:
            start = lower_comment.find(sent.lower())

        if start < 0:
            skipped += 1
            continue

        end = start + len(sent)
        spans.append((sentence_id, start, end, int(label)))
        cursor = end

    return spans, skipped

class ContextualizedCommentSentenceDataset(Dataset):
    """
    Here, one item is one full comment with token masks for its linked sentence spans.

    The sentence classifier receives pooled token states from inside the full
    contextualized comment representation, rather than separately encoded sentences.
    """

    def __init__(
        self,
        comments: pd.DataFrame,
        sentences: pd.DataFrame,
        tokenizer,
        comment_text_col: str,
        comment_label_col: str,
        sentence_text_col: str,
        sentence_label_col: str,
        split_name: str,
    ):
        self.comments = comments.reset_index(drop=True).copy()
        self.tokenizer = tokenizer
        self.comment_text_col = comment_text_col
        self.comment_label_col = comment_label_col
        self.sentence_map = build_sentence_map(sentences, sentence_text_col, sentence_label_col)

        self.samples: List[Dict[str, object]] = []
        total_sentences = 0
        skipped_sentences = 0

        for _, row in self.comments.iterrows():
            cid = row["comment_id"]
            comment_text = str(row[comment_text_col])
            linked_sentences = self.sentence_map.get(cid, [])
            spans, skipped = find_sentence_char_spans(comment_text, linked_sentences)
            total_sentences += len(linked_sentences)
            skipped_sentences += skipped
            self.samples.append(
                {
                    "comment_id": cid,
                    "comment_text": comment_text,
                    "comment_label": int(row[comment_label_col]),
                    "sentence_spans": spans,
                }
            )

        matched = total_sentences - skipped_sentences
        print(
            f"{split_name}: matched {matched}/{total_sentences} sentence spans "
            f"({skipped_sentences} skipped before tokenization)."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]
        comment_text = str(sample["comment_text"])
        comment_label = int(sample["comment_label"])

        enc = self.tokenizer(
            comment_text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        offsets = enc["offset_mapping"].squeeze(0).tolist()

        sentence_masks: List[torch.Tensor] = []
        sentence_labels: List[int] = []
        sentence_ids: List[object] = []

        for sentence_id, start_char, end_char, label in sample["sentence_spans"]:
            mask = torch.zeros(MAX_LEN, dtype=torch.bool)
            for token_idx, (tok_start, tok_end) in enumerate(offsets):
                # Special tokens and padding generally have (0, 0).
                if tok_start == tok_end:
                    continue
                # Token overlaps sentence character span.
                if tok_start < end_char and tok_end > start_char:
                    mask[token_idx] = True

            # Skip sentences that were truncated away or failed to map to tokens.
            if mask.any():
                sentence_masks.append(mask)
                sentence_labels.append(int(label))
                sentence_ids.append(sentence_id)

        if sentence_masks:
            sentence_masks_tensor = torch.stack(sentence_masks)
            sentence_labels_tensor = torch.tensor(sentence_labels, dtype=torch.long)
        else:
            sentence_masks_tensor = torch.empty((0, MAX_LEN), dtype=torch.bool)
            sentence_labels_tensor = torch.empty((0,), dtype=torch.long)

        return {
            "comment_id": sample["comment_id"],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "comment_label": torch.tensor(comment_label, dtype=torch.long),
            "sentence_ids": sentence_ids,
            "sentence_masks": sentence_masks_tensor,
            "sentence_labels": sentence_labels_tensor,
        }

class ContextualizedBatchCollator:
    """Collatting full comments and flatten contextualized sentence masks."""

    def __call__(self, samples: List[Dict[str, object]]) -> Dict[str, object]:
        input_ids = torch.stack([s["input_ids"] for s in samples])
        attention_mask = torch.stack([s["attention_mask"] for s in samples])
        comment_labels = torch.stack([s["comment_label"] for s in samples])
        comment_ids = [s["comment_id"] for s in samples]

        flat_sentence_masks: List[torch.Tensor] = []
        flat_sentence_labels: List[torch.Tensor] = []
        flat_sentence_comment_indices: List[int] = []
        flat_sentence_ids: List[object] = []

        for comment_idx, sample in enumerate(samples):
            masks = sample["sentence_masks"]
            labels = sample["sentence_labels"]
            ids = sample["sentence_ids"]
            for local_sentence_idx in range(masks.shape[0]):
                flat_sentence_masks.append(masks[local_sentence_idx])
                flat_sentence_labels.append(labels[local_sentence_idx])
                flat_sentence_comment_indices.append(comment_idx)
                flat_sentence_ids.append(ids[local_sentence_idx])

        if flat_sentence_masks:
            sentence_masks = torch.stack(flat_sentence_masks)
            sentence_labels = torch.stack(flat_sentence_labels).long()
            sentence_comment_indices = torch.tensor(flat_sentence_comment_indices, dtype=torch.long)
        else:
            sentence_masks = torch.empty((0, MAX_LEN), dtype=torch.bool)
            sentence_labels = torch.empty((0,), dtype=torch.long)
            sentence_comment_indices = torch.empty((0,), dtype=torch.long)

        return {
            "comment_ids": comment_ids,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "comment_labels": comment_labels,
            "sentence_ids": flat_sentence_ids,
            "sentence_masks": sentence_masks,
            "sentence_labels": sentence_labels,
            "sentence_comment_indices": sentence_comment_indices,
        }

  class ContextualizedSentenceBert(nn.Module):
    def __init__(self, model_name: str = MODEL_NAME, num_labels: int = 2, dropout_prob: float = 0.1): 
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout_prob)
        self.comment_classifier = nn.Linear(hidden_size, num_labels)
        self.sentence_classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sentence_masks: Optional[torch.Tensor] = None,
        sentence_comment_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        cls_vec = self.dropout(hidden[:, 0, :])
        comment_logits = self.comment_classifier(cls_vec)

        sentence_logits: Optional[torch.Tensor] = None
        if (
            sentence_masks is not None
            and sentence_comment_indices is not None
            and sentence_masks.shape[0] > 0
        ):
            sentence_masks = sentence_masks.to(hidden.device)
            sentence_comment_indices = sentence_comment_indices.to(hidden.device)
          
            parent_hidden = hidden.index_select(0, sentence_comment_indices)  
            mask = sentence_masks.unsqueeze(-1).to(dtype=parent_hidden.dtype)  

            summed = (parent_hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            sentence_vecs = summed / denom
            sentence_vecs = self.dropout(sentence_vecs)
            sentence_logits = self.sentence_classifier(sentence_vecs)

        return comment_logits, sentence_logits

def make_class_weights(y: np.ndarray, num_classes: int = 2) -> torch.Tensor:
    classes = np.arange(num_classes)
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) == num_classes:
        cw = compute_class_weight(class_weight="balanced", classes=classes, y=y)
        return torch.tensor(cw, dtype=torch.float)

    counts = np.bincount(y, minlength=num_classes)
    total = max(int(counts.sum()), 1)
    weights = np.ones(num_classes, dtype=np.float32)
    for c in classes:
        if counts[c] > 0:
            weights[c] = total / (num_classes * counts[c])
    return torch.tensor(weights, dtype=torch.float)

@dataclass
class TrainConfig:
    device: torch.device
    comment_class_weights: torch.Tensor
    sentence_class_weights: torch.Tensor
    alpha_sentence_loss: float
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


def aggregate_any_sentence_probability(
    p_sentence_uncivil: torch.Tensor,
    sentence_comment_indices: torch.Tensor,
    batch_size: int,
    eps: float = 1e-7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute 1 - prod_i(1 - p_ij) for linked sentences in each comment."""
    p_any = torch.zeros(batch_size, device=p_sentence_uncivil.device, dtype=p_sentence_uncivil.dtype)
    valid = torch.zeros(batch_size, device=p_sentence_uncivil.device, dtype=torch.bool)

    for comment_idx in range(batch_size):
        mask = sentence_comment_indices == comment_idx
        if mask.any():
            p = p_sentence_uncivil[mask].clamp(eps, 1.0 - eps)
            p_any[comment_idx] = 1.0 - torch.prod(1.0 - p)
            valid[comment_idx] = True

    return p_any.clamp(eps, 1.0 - eps), valid

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer,
    scheduler,
    config: TrainConfig,
) -> None:
    model.train()
    device = config.device
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp and device.type == "cuda")

    need_sentence_outputs = (config.alpha_sentence_loss > 0.0)

    for batch in train_loader:
        optimizer.zero_grad(set_to_none=True)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        comment_labels = batch["comment_labels"].to(device)

        sentence_masks = None
        sentence_comment_indices = None
        if need_sentence_outputs and batch["sentence_masks"].shape[0] > 0:
            sentence_masks = batch["sentence_masks"].to(device)
            sentence_comment_indices = batch["sentence_comment_indices"].to(device)

        with torch.cuda.amp.autocast(enabled=config.use_amp and device.type == "cuda"):
            # One encoder forward pass over the full comment.
            comment_logits, sentence_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                sentence_masks=sentence_masks,
                sentence_comment_indices=sentence_comment_indices,
            )

            loss_comment = F.cross_entropy(
                comment_logits,
                comment_labels,
                weight=config.comment_class_weights.to(device),
                label_smoothing=config.label_smoothing,
            )

            loss_sentence = comment_logits.sum() * 0.0

            if sentence_logits is not None:
                if config.alpha_sentence_loss > 0.0:
                    sentence_labels = batch["sentence_labels"].to(device)
                    loss_sentence = F.cross_entropy(
                        sentence_logits,
                        sentence_labels,
                        weight=config.sentence_class_weights.to(device),
                        label_smoothing=config.label_smoothing,
                    )

            loss = (
                loss_comment
                + config.alpha_sentence_loss * loss_sentence
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()


@torch.no_grad()
def evaluate_comment_level(model: nn.Module, data_loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Evaluate comment-level predictions only."""
    model.eval()
    all_labels: List[int] = []
    all_preds: List[int] = []

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        comment_labels = batch["comment_labels"].to(device)

        comment_logits, _sentence_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sentence_masks=None,
            sentence_comment_indices=None,
        )
        preds = torch.argmax(comment_logits, dim=1)

        all_labels.extend(comment_labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

    return compute_binary_metrics(all_labels, all_preds)


def make_loaders(
    tokenizer,
    train_comments_df: pd.DataFrame,
    train_sentences_df: pd.DataFrame,
    val_comments_df: pd.DataFrame,
    val_sentences_df: pd.DataFrame,
    test_comments_df: pd.DataFrame,
    test_sentences_df: pd.DataFrame,
    comment_text_col: str,
    comment_label_col: str,
    sentence_text_col: str,
    sentence_label_col: str,
    seed: int,
):
    collator = ContextualizedBatchCollator()

    train_dataset = ContextualizedCommentSentenceDataset(
        train_comments_df,
        train_sentences_df,
        tokenizer,
        comment_text_col,
        comment_label_col,
        sentence_text_col,
        sentence_label_col,
        split_name="train",
    )
    val_dataset = ContextualizedCommentSentenceDataset(
        val_comments_df,
        val_sentences_df,
        tokenizer,
        comment_text_col,
        comment_label_col,
        sentence_text_col,
        sentence_label_col,
        split_name="validation",
    )
    test_dataset = ContextualizedCommentSentenceDataset(
        test_comments_df,
        test_sentences_df,
        tokenizer,
        comment_text_col,
        comment_label_col,
        sentence_text_col,
        sentence_label_col,
        split_name="test",
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
        worker_init_fn=seed_worker,
        generator=train_generator,
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
    )

    return train_loader, train_eval_loader, val_loader, test_loader


def train_model(
    train_comments_df: pd.DataFrame,
    train_sentences_df: pd.DataFrame,
    val_comments_df: pd.DataFrame,
    val_sentences_df: pd.DataFrame,
    test_comments_df: pd.DataFrame,
    test_sentences_df: pd.DataFrame,
    comment_text_col: str,
    comment_label_col: str,
    sentence_text_col: str,
    sentence_label_col: str,
    seed: int,
    early_stop_metric: str = "f1_weighted",
):
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError(
            "Version 2 requires a fast tokenizer because it uses offset_mapping "
            "to map sentence character spans to token spans."
        )

    train_loader, train_eval_loader, val_loader, test_loader = make_loaders(
        tokenizer=tokenizer,
        train_comments_df=train_comments_df,
        train_sentences_df=train_sentences_df,
        val_comments_df=val_comments_df,
        val_sentences_df=val_sentences_df,
        test_comments_df=test_comments_df,
        test_sentences_df=test_sentences_df,
        comment_text_col=comment_text_col,
        comment_label_col=comment_label_col,
        sentence_text_col=sentence_text_col,
        sentence_label_col=sentence_label_col,
        seed=seed,
    )

    model = ContextualizedSentenceBert(model_name=MODEL_NAME)
    model.to(device)

    comment_class_weights = make_class_weights(train_comments_df[comment_label_col].values)
    sentence_class_weights = make_class_weights(train_sentences_df[sentence_label_col].values)

    total_steps = max(1, EPOCHS * len(train_loader))
    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        total_steps=total_steps,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_proportion=WARMUP_PROPORTION,
    )

    train_cfg = TrainConfig(
        device=device,
        comment_class_weights=comment_class_weights,
        sentence_class_weights=sentence_class_weights,
        alpha_sentence_loss=ALPHA_SENTENCE_LOSS,
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
        train_one_epoch(model, train_loader, optimizer, scheduler, train_cfg)

        train_metrics = evaluate_comment_level(model, train_eval_loader, device)
        val_metrics = evaluate_comment_level(model, val_loader, device)
        val_score = float(val_metrics[early_stop_metric])

        print("Train metrics:", train_metrics)
        print("Validation metrics:", val_metrics)
        print(f"Early-stop metric val_{early_stop_metric}: {val_score:.6f}")

        row = {
            "epoch": int(epoch),
            "val_score": val_score,
            **prefix_metrics(train_metrics, "train"),
            **prefix_metrics(val_metrics, "val"),
        }
        epoch_rows.append(row)

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

    test_metrics = evaluate_comment_level(model, test_loader, device)
    print("Test metrics using best validation checkpoint:", test_metrics)

    epoch_df = pd.DataFrame(epoch_rows)
    return model, tokenizer, float(best_val_score), int(best_epoch), epoch_df, test_metrics

#Setting the hyperparameters
SEEDS = [40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55]
ALPHA_SENTENCE_LOSS_VALUES = [0,0.1,0.25,0.5,1] #When alpha=0, it recovers the pure comment model

#the following grid corresponds to the best performing hyperparameters for roberta, change as needed
grid = {
    "RNG_SEED": SEEDS,
    "LR": [1e-5],
    "BATCH_SIZE": [8],
    "WEIGHT_DECAY": [0.01],
    "WARMUP_PROPORTION": [0.1],
    "LABEL_SMOOTHING": [0.1],
    "ALPHA_SENTENCE_LOSS": ALPHA_SENTENCE_LOSS_VALUES
}

EPOCHS_SEARCH = 30
PATIENCE_SEARCH = 30 #The training did not include early stopping, set to <30 if early stopping is desired.
EARLY_STOP_METRIC = "f1_weighted"

RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
SUMMARY_RESULTS_CSV = BASE_OUTPUT_DIR / f"onefp_test_roberta_{SEEDS}_{RUN_TAG}.csv"
EPOCH_RESULTS_CSV = BASE_OUTPUT_DIR / f"onefp_epoch_roberta_{SEEDS}_{RUN_TAG}.csv"

results = []
all_epoch_results = []
best = {"score": -1.0, "cfg": None}

keys = list(grid.keys())
raw_configs = list(itertools.product(*[grid[k] for k in keys]))

configs = []
for values in raw_configs:
    cfg_tmp = dict(zip(keys, values))

    configs.append(values)

print("Total configs:", len(configs))
print("Seeds:", SEEDS)
print("Alpha values:", ALPHA_SENTENCE_LOSS_VALUES)

for i, values in enumerate(configs, start=1):
    cfg = dict(zip(keys, values))

    print("\n" + "=" * 70)
    print(f"Run {i}/{len(configs)} | cfg = {cfg}")

    RNG_SEED = int(cfg["RNG_SEED"])
    LR = float(cfg["LR"])
    BATCH_SIZE = int(cfg["BATCH_SIZE"])
    WEIGHT_DECAY = float(cfg["WEIGHT_DECAY"])
    WARMUP_PROPORTION = float(cfg["WARMUP_PROPORTION"])
    LABEL_SMOOTHING = float(cfg["LABEL_SMOOTHING"])
    ALPHA_SENTENCE_LOSS = float(cfg["ALPHA_SENTENCE_LOSS"])

    EPOCHS = EPOCHS_SEARCH
    PATIENCE = PATIENCE_SEARCH

    df_train, val_comments, test_comments, train_sentences, val_sentences, test_sentences = make_splits(RNG_SEED)

    print("train/val/test comments:", len(df_train), len(val_comments), len(test_comments))
    print("train/val/test sentences:", len(train_sentences), len(val_sentences), len(test_sentences))

    model, tokenizer, best_val_score, best_epoch, epoch_df, test_metrics = train_model(
        train_comments_df=df_train,
        train_sentences_df=train_sentences,
        val_comments_df=val_comments,
        val_sentences_df=val_sentences,
        test_comments_df=test_comments,
        test_sentences_df=test_sentences,
        comment_text_col="Comment",
        comment_label_col="incivility",
        sentence_text_col="sentence",
        sentence_label_col="incivility",
        seed=RNG_SEED,
        early_stop_metric=EARLY_STOP_METRIC,
    )

    row = {
        **cfg,
        "run_index": int(i),
        "run_tag": RUN_TAG,
        "best_val_score": float(best_val_score),
        "best_epoch": int(best_epoch),
        **prefix_metrics(test_metrics, "test"),
    }
    results.append(row)

    epoch_df = epoch_df.assign(
        run_index=int(i),
        run_tag=RUN_TAG,
        RNG_SEED=RNG_SEED,
        LR=LR,
        BATCH_SIZE=BATCH_SIZE,
        WEIGHT_DECAY=WEIGHT_DECAY,
        WARMUP_PROPORTION=WARMUP_PROPORTION,
        LABEL_SMOOTHING=LABEL_SMOOTHING,
        ALPHA_SENTENCE_LOSS=ALPHA_SENTENCE_LOSS,
    )
    all_epoch_results.append(epoch_df)

    if best_val_score > best["score"]:
        best = {"score": float(best_val_score), "cfg": cfg}
        print(">>> NEW BEST:", best)

    pd.DataFrame(results).sort_values("best_val_score", ascending=False).to_csv(SUMMARY_RESULTS_CSV, index=False)
    pd.concat(all_epoch_results, ignore_index=True).to_csv(EPOCH_RESULTS_CSV, index=False)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

results_df = pd.DataFrame(results).sort_values("best_val_score", ascending=False)
epoch_results_df = pd.concat(all_epoch_results, ignore_index=True)

print("\nTop configs:")
print(results_df.head(20))

print("\nMean test f1_weighted by ALPHA_SENTENCE_LOSS")
print(
    results_df.groupby(["ALPHA_SENTENCE_LOSS"])["test_f1_weighted"]
    .agg(["mean", "std", "count"])
    .sort_values("mean", ascending=False)
)

print("\nSaved summary results to:", SUMMARY_RESULTS_CSV)
print("Saved epoch-level results to:", EPOCH_RESULTS_CSV)
print("\nBest config:", best["cfg"])
print("Best validation score:", best["score"])
