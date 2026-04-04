#!/usr/bin/env python3
"""Полный прогон HW13: датасет emotion, DistilBERT, артефакты в ./artifacts/.

Запуск из каталога homeworks/HW13:
  python3 hw13_run.py

Или из корня репозитория:
  python3 homeworks/HW13/hw13_run.py
"""
from __future__ import annotations

import inspect
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from datasets import load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

SEED = 42
MODEL_NAME = "distilbert-base-uncased"


def _hw13_root() -> Path:
    here = Path(__file__).resolve().parent
    if (here / "HW13.ipynb").exists():
        return here
    return Path.cwd()


def main() -> int:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    base = _hw13_root()
    artifact_dir = base / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "| BASE:", base.resolve())

    raw = load_dataset("emotion")
    label_names = list(raw["train"].features["label"].names)
    id2label = {i: name for i, name in enumerate(label_names)}
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(label_names)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    tokenized = raw.map(tokenize_fn, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized = tokenized.remove_columns(["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        }

    training_args = TrainingArguments(
        output_dir=str(artifact_dir / "trainer_checkpoints"),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        seed=SEED,
        logging_steps=50,
        report_to="none",
    )

    model_ft = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # transformers 5.x: `tokenizer` → `processing_class`; 4.x оставляет только `tokenizer`
    _trainer_kw = dict(
        model=model_ft,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        _trainer_kw["processing_class"] = tokenizer
    else:
        _trainer_kw["tokenizer"] = tokenizer
    trainer = Trainer(**_trainer_kw)

    trainer.train()

    pred_out = trainer.predict(tokenized["test"])
    logits_test = pred_out.predictions
    y_true = pred_out.label_ids
    y_pred = np.argmax(logits_test, axis=-1)

    test_acc = float(accuracy_score(y_true, y_pred))
    test_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    print("test_accuracy:", test_acc)
    print("test_f1_macro:", test_f1)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_xticks(range(num_labels))
    ax.set_yticks(range(num_labels))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)
    ax.set_ylabel("Истина")
    ax.set_xlabel("Предсказание")
    mx = cm.max() if cm.size else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="w" if cm[i, j] > mx / 2 else "black",
            )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    cm_path = artifact_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print("saved:", cm_path)

    probs_test = torch.softmax(torch.tensor(logits_test, dtype=torch.float32), dim=-1).numpy()
    conf = probs_test[np.arange(len(y_pred)), y_pred]
    test_texts = raw["test"]["text"]

    rows = []
    for i in range(len(y_true)):
        rows.append(
            {
                "text": test_texts[i],
                "true_label": id2label[int(y_true[i])],
                "pred_label": id2label[int(y_pred[i])],
                "confidence": float(conf[i]),
            }
        )

    pred_df = pd.DataFrame(rows)
    pred_path = artifact_dir / "sample_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print("saved:", pred_path, "rows:", len(pred_df))

    # Для заполнения report.md и проверки воспроизводимости
    meta = {
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
        "model_name": MODEL_NAME,
        "dataset": "emotion",
        "num_labels": num_labels,
        "label_names": label_names,
        "seed": SEED,
        "device": str(device),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
    }
    (artifact_dir / "hw13_metrics.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved:", artifact_dir / "hw13_metrics.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
