import os
import torch
import wandb
from torch.utils.data import DataLoader
from torch import nn, optim

from config import configs, DATASET_ROOT, PATIENCE
from dataset import load_datasets
from model import ResNetClassifier
from train_eval import train_one_epoch, evaluate, test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
datasets_dict = load_datasets(DATASET_ROOT)
class_names = datasets_dict["train"].classes
num_classes = len(class_names)

summary_results = []
best_overall = {"val_acc": 0.0, "config_idx": None, "run_idx": None, "model_path": None}

for i, config in enumerate(configs):
    print(f"\nConfig {i+1}: {config}")
    best_val_acc_cfg, best_model_path_cfg, best_test_acc_cfg, best_run_idx_cfg = 0.0, None, 0.0, -1

    for j in range(3):
        run_name = f"resnet18_cfg{i+1}_run{j+1}"
        print(f"\nRun {j+1}/3 - {run_name}")
        wandb.init(project="miniimagenet-resnet18", config=config, name=run_name, reinit=True)

        loaders = {
            "train": DataLoader(datasets_dict["train"], batch_size=config["batch_size"], shuffle=True),
            "val": DataLoader(datasets_dict["val"], batch_size=config["batch_size"], shuffle=False),
            "test": DataLoader(datasets_dict["test"], batch_size=config["batch_size"], shuffle=False),
        }

        model = ResNetClassifier(num_classes).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
        optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.05)

        best_val_acc_run = 0.0
        best_model_run_path = f"temp_best_cfg{i+1}_run{j+1}.pth"
        early_stop_counter = 0

        for epoch in range(config["epochs"]):
            train_loss, train_acc = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, loaders["val"], criterion, device)

            wandb.log({"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})
            print(f"[Epoch {epoch+1}] Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

            if val_acc > best_val_acc_run:
                best_val_acc_run = val_acc
                torch.save(model.state_dict(), best_model_run_path)
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= PATIENCE:
                    print("Early stopping triggered.")
                    break

        model.load_state_dict(torch.load(best_model_run_path))
        test_acc = test(model, loaders["test"], device)
        wandb.log({"test_acc": test_acc})

        if best_val_acc_run > best_val_acc_cfg:
            best_val_acc_cfg = best_val_acc_run
            best_model_path_cfg = f"best_model_cfg{i+1}.pth"
            best_run_idx_cfg = j + 1
            best_test_acc_cfg = test_acc
            os.replace(best_model_run_path, best_model_path_cfg)

        wandb.finish()

    for k in range(3):
        temp_path = f"temp_best_cfg{i+1}_run{k+1}.pth"
        if os.path.exists(temp_path) and (k + 1) != best_run_idx_cfg:
            os.remove(temp_path)

    summary_results.append({
        "config_idx": i + 1,
        "config": config,
        "val_acc": best_val_acc_cfg,
        "test_acc": best_test_acc_cfg,
        "run_idx": best_run_idx_cfg,
        "model_path": best_model_path_cfg
    })

    if best_val_acc_cfg > best_overall["val_acc"]:
        best_overall = {
            "val_acc": best_val_acc_cfg,
            "config_idx": i + 1,
            "run_idx": best_run_idx_cfg,
            "model_path": best_model_path_cfg
        }

print("\nTổng hợp kết quả:")
for result in summary_results:
    print(f"Config {result['config_idx']} | Run {result['run_idx']} → Val Acc: {result['val_acc']:.4f} | Test Acc: {result['test_acc']:.4f} | Model: {result['model_path']}")

print(f"\nMô hình tốt nhất toàn bộ:")
print(f"Config {best_overall['config_idx']} | Run {best_overall['run_idx']} → Val Acc: {best_overall['val_acc']:.4f} | Model: {best_overall['model_path']}")