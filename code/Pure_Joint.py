#Loading the dependencies
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from sklearn.model_selection import train_test_split

comment_df = pd.read_csv("data/comments.csv", encoding='latin-1')
#This dataset contains the following key variables:
#      - comment_id
#      - comment
#      - incivility
sentence_df = pd.read_csv("data/sentence.csv", encoding='latin-1')
#This dataset contains the following key variables:
#      - comment_id
#      - sentence_id
#      - comment
#      - sentence
#      - incivility


MODEL_NAME = "google-bert/bert-large-cased" #Change to ""vinai/bertweet-large" or "roberta-large"

#Defining the hyperparameters
RNG_SEED = 42 #This is the random seed 
MAX_LEN = 512 #This is the maximum sequence length       
BATCH_SIZE = 16 #This is the batch size   
LR = 1e-5 #This is the learning rate        
WEIGHT_DECAY = 0.01 #This is thw weight decay
WARMUP_PROPORTION = 0.10 #This is the warm-up proportion
LABEL_SMOOTHING = 0.15 #This is the label smoothing
EPOCHS = 30 #This is the number of epochs
PATIENCE = 30 #This is a parameter for early stopping. Note: since it is said to 30, there was no early stopping.
MAX_GRAD_NORM = 1.0 #This is the gradient clipping

# The following parameter decides how loss is defined. If this is set to 0, the loss is defined entirely by the comment labels, i.e., the 'pure-comment model.' If, however, this is set to 1, the loss is defined jointly by the comment and sentence labels, i.e., the 'joint comment-sentence model.'
ALPHA_SENTENCE_LOSS = 0  

# Setting the random seed for all major sources of randonness, save dataset partioning (see below).
def seed_everything(seed: int = RNG_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Creating a PyTorch dataset for the comment labels
class CommentDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, text_col: str, label_col: str):
        self.texts = df[text_col].astype(str).tolist()
        self.labels = df[label_col].astype(int).tolist()
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }

# Creating a PyTorch dataset for the sentence labels
class SentenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, text_col: str, label_col: str):
        self.texts = df[text_col].astype(str).tolist()
        self.labels = df[label_col].astype(int).tolist()
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# Creating the model
class MultiTask(nn.Module):
    """
    Encoder with two task-specific classification heads:
    - comment head (main task)
    - sentence head (auxiliary task) (disregarded if ALPHA_SENTENCE_LOSS=0)
    """

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
        task: str,
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  
        pooled = self.dropout(pooled)

        if task == "comment":
            logits = self.comment_classifier(pooled)
        elif task == "sentence":
            logits = self.sentence_classifier(pooled)
        else:
            raise ValueError(f"Unknown task: {task}")

        return logits


# Correcting for label imbalance
def make_class_weights(y: np.ndarray, num_classes: int = 2) -> torch.Tensor:
    classes = np.arange(num_classes)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return torch.tensor(cw, dtype=torch.float)

# Packaging training-related settings and tensors needed across a given training loop. 
@dataclass
class TrainConfig:
    device: torch.device
    comment_class_weights: torch.Tensor
    sentence_class_weights: torch.Tensor
    use_amp: bool = True

# Optimization set-up.
def create_optimizer_and_scheduler(
    model: nn.Module,
    total_steps: int,
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
            {"params": decay_params, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=LR,
    )

    warmup_steps = int(WARMUP_PROPORTION * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    return optimizer, scheduler


# Training one batch of comments and correspondinng sentences 
def train_one_epoch(
    model: nn.Module,
    comment_loader: DataLoader,
    sentence_loader: DataLoader,
    optimizer,
    scheduler,
    config: TrainConfig,
):
    model.train()
    device = config.device

    scaler = torch.cuda.amp.GradScaler(
        enabled=config.use_amp and device.type == "cuda"
    )

    for batch_c, batch_s in zip(comment_loader, sentence_loader):
        optimizer.zero_grad(set_to_none=True)

        input_ids_c = batch_c["input_ids"].to(device)
        attention_mask_c = batch_c["attention_mask"].to(device)
        labels_c = batch_c["labels"].to(device)

        input_ids_s = batch_s["input_ids"].to(device)
        attention_mask_s = batch_s["attention_mask"].to(device)
        labels_s = batch_s["labels"].to(device)

        with torch.cuda.amp.autocast(enabled=config.use_amp and device.type == "cuda"):
            logits_c = model(input_ids_c, attention_mask_c, task="comment")
            loss_c = F.cross_entropy(
                logits_c,
                labels_c,
                weight=config.comment_class_weights.to(device),
                label_smoothing=LABEL_SMOOTHING,
            )

            logits_s = model(input_ids_s, attention_mask_s, task="sentence")
            loss_s = F.cross_entropy(
                logits_s,
                labels_s,
                weight=config.sentence_class_weights.to(device),
                label_smoothing=LABEL_SMOOTHING,
            )

            loss = loss_c + ALPHA_SENTENCE_LOSS * loss_s # i.e., if alpha_sentence_loss=0, then the loss is only a function of comment labels

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

# Evaluates the model on comment-level incivility
@torch.no_grad()
def evaluate_comment_level(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    all_labels: List[int] = []
    all_preds: List[int] = []

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask, task="comment")
        preds = torch.argmax(logits, dim=1)

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1_w = f1_score(all_labels, all_preds, average="weighted")
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    report = classification_report(
        all_labels,
        all_preds,
        target_names=["civil", "uncivil"],
        digits=4
    )

    print(report)

    return {"accuracy": acc, "f1_weighted": f1_w, "f1_macro": f1_macro}


def train_model(
    train_comments_df: pd.DataFrame,
    train_sentences_df: pd.DataFrame,
    val_comments_df: pd.DataFrame,
    comment_text_col: str,
    comment_label_col: str,
    sentence_text_col: str,
    sentence_label_col: str,
):
    """
    This is the high-level training entry point, where:
    - train_comments_df contains one comment per row and its corresponding label
    - train_sentences_df: contains one sentence per row and its corresponding label
    - val_comments_df: contains validation comments with comment-level labels only
    """
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    # Datasets / loaders
    train_comment_dataset = CommentDataset(
        train_comments_df, tokenizer, text_col=comment_text_col, label_col=comment_label_col
    )
    val_comment_dataset = CommentDataset(
        val_comments_df, tokenizer, text_col=comment_text_col, label_col=comment_label_col
    )
    train_sentence_dataset = SentenceDataset(
        train_sentences_df, tokenizer, text_col=sentence_text_col, label_col=sentence_label_col
    )

    train_comment_loader = DataLoader(
        train_comment_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    train_sentence_loader = DataLoader(
        train_sentence_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_comment_loader = DataLoader(
        val_comment_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    model = MultiTask(model_name=MODEL_NAME)
    model.to(device)

    comment_class_weights = make_class_weights(
        train_comments_df[comment_label_col].values
    )
    sentence_class_weights = make_class_weights(
        train_sentences_df[sentence_label_col].values
    )

    steps_per_epoch = min(len(train_comment_loader), len(train_sentence_loader))
    total_steps = EPOCHS * steps_per_epoch

    optimizer, scheduler = create_optimizer_and_scheduler(model, total_steps)

    train_cfg = TrainConfig(
        device=device,
        comment_class_weights=comment_class_weights,
        sentence_class_weights=sentence_class_weights,
        use_amp=True,
    )

    best_val_f1 = 0.0
    epochs_no_improve = 0
    best_state_dict = None

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_one_epoch(
            model=model,
            comment_loader=train_comment_loader,
            sentence_loader=train_sentence_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=train_cfg,
        )

        val_metrics = evaluate_comment_level(model, val_comment_loader, device)
        val_f1 = val_metrics["f1_weighted"]
        print(f"Validation metrics: {val_metrics}")

        # Early stopping possible if PATIENCE < EPOCHS
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    return model, tokenizer, best_val_f1

# Splitting dataset into a train (70%), validation (15%) and test split (15%).
df_train, df_tmp = train_test_split(
    comment_df,
    test_size=0.30,
    random_state=RNG_SEED, # Partioning is done according to the seed
    stratify=comment_df["incivility"]
)
val_comments, test_comments = train_test_split(
    df_tmp,
    test_size=0.50,
    random_state=RNG_SEED,
    stratify=df_tmp["incivility"]
)

print(len(df_train), len(val_comments), len(test_comments))

# Ensuring that the sentence supervision is partitioned consistently with the main task.
train_comment_ids = set(df_train["comment_id"])
val_comment_ids   = set(val_comments["comment_id"])
test_comment_ids  = set(test_comments["comment_id"])
train_sentences = sentence_df[sentence_df["comment_id"].isin(train_comment_ids)]
val_sentences   = sentence_df[sentence_df["comment_id"].isin(val_comment_ids)]

print(RNG_SEED)

# Running a given model-method-seed combination
model, tokenizer, best_val_f1 = train_model(
    train_comments_df=df_train,
    train_sentences_df=train_sentences,
    val_comments_df=val_comments,
    comment_text_col="Comment",      # column with full comment text
    comment_label_col="incivility",  # 0/1 label
    sentence_text_col="sentence",    # sentence text column
    sentence_label_col="incivility", # 0/1 sentence label
)

print("Best validation F1 (comments):", best_val_f1)

# Evaluating the best performing epoch on the test set
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

test_dataset = CommentDataset(
    test_comments,
    tokenizer,
    text_col="Comment",
    label_col="incivility",
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_metrics = evaluate_comment_level(model, test_loader, device)
print("Test metrics:", test_metrics)
