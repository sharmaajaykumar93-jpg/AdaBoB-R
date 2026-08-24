import os
import random
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support


# ============================================================
# Publication-quality plot settings
# ============================================================

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 19,
    "axes.titlesize": 19,
    "xtick.labelsize": 14,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
    "lines.linewidth": 2,
    "lines.markersize": 6,
})


# ============================================================
# Reproducibility
# ============================================================

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# CIFAR-10 Data
# ============================================================

def get_cifar10_loaders(batch_size=128, num_workers=2):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    train_set = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform_train
    )

    test_set = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform_test
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


# ============================================================
# ResNet-18 for CIFAR-10
# ============================================================

def build_resnet18_cifar10(num_classes=10):
    model = models.resnet18(weights=None)

    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )

    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ============================================================
# AdaBoB-R Optimizer ONLY
# ============================================================

class AdaBoBR(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        final_lr=0.1,
        betas=(0.9, 0.999),
        gamma=1e-3,
        eps=1e-8,
        lam=0.5,
        warmup_steps=50,
        weight_decay=0.0
    ):
        defaults = dict(
            lr=lr,
            final_lr=final_lr,
            betas=betas,
            gamma=gamma,
            eps=eps,
            lam=lam,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            final_lr = group["final_lr"]
            beta1, beta2 = group["betas"]
            gamma = group["gamma"]
            eps = group["eps"]
            lam = group["lam"]
            warmup_steps = group["warmup_steps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                m = state["m"]
                v = state["v"]

                state["step"] += 1
                t = state["step"]

                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                b = grad - m
                v.mul_(beta2).addcmul_(b, b, value=1 - beta2)

                alpha_t = lr * np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

                alpha_l = final_lr * (1 - 1 / (gamma * t + 1))
                alpha_u = final_lr * (1 + 1 / (gamma * t))

                v_hat = v / (1 - beta2 ** t)

                outlier_score = torch.mean(
                    torch.abs(b) / (torch.sqrt(v_hat) + eps)
                )

                w_t = min(1.0, t / warmup_steps)
                robust_lower = w_t * alpha_l
                robust_upper = w_t * max(
                    alpha_l,
                    alpha_u / (1.0 + lam * float(outlier_score))
                )

                step_size = alpha_t / (torch.sqrt(v) + eps)
                step_size = torch.clamp(
                    step_size,
                    min=robust_lower,
                    max=robust_upper
                )

                # Keep this consistent with the supplied experimental code.
                # For the theoretical version, uncomment the next line:
                # step_size = step_size / np.sqrt(t)

                p.addcmul_(m, step_size, value=-1.0)

        return loss


# ============================================================
# Evaluation
# ============================================================

def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds) * 100.0

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets,
        all_preds,
        average="macro",
        zero_division=0
    )

    return (
        avg_loss,
        acc,
        precision * 100.0,
        recall * 100.0,
        f1 * 100.0
    )


# ============================================================
# One AdaBoB-R run
# ============================================================

def train_adabobr(
    train_loader,
    test_loader,
    epochs,
    lam,
    warmup_steps,
    gamma,
    final_lr,
    lr=1e-3,
    weight_decay=5e-4,
    seed=SEED,
    run_name="run"
):
    # Reset seed so each hyperparameter setting starts from the same
    # model initialization and random-number state as closely as possible.
    set_seed(seed)

    model = build_resnet18_cifar10(num_classes=10).to(DEVICE)
    optimizer = AdaBoBR(
        model.parameters(),
        lr=lr,
        final_lr=final_lr,
        gamma=gamma,
        lam=lam,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_preds = []
        train_targets = []

        for x, y in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            train_preds.extend(preds.detach().cpu().numpy())
            train_targets.extend(y.detach().cpu().numpy())

        train_loss = train_loss_sum / len(train_loader.dataset)
        train_acc = accuracy_score(train_targets, train_preds) * 100.0

        test_loss, test_acc, precision, recall, f1 = evaluate(
            model, test_loader, criterion
        )

        history.append({
            "run_name": run_name,
            "epoch": epoch,
            "lambda": lam,
            "T_w": warmup_steps,
            "gamma": gamma,
            "alpha_f": final_lr,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": f1
        })

        print(
            f"{run_name:28s} | epoch {epoch:03d} | "
            f"test acc {test_acc:6.2f}% | F1 {f1:6.2f}% | "
            f"test loss {test_loss:.4f}"
        )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pd.DataFrame(history)


# ============================================================
# Plot helpers
# ============================================================

def parameter_display_name(parameter):
    names = {
        "lambda": r"$\lambda$",
        "T_w": r"$T_w$",
        "gamma": r"$\gamma$",
        "alpha_f": r"$\alpha_f$",
    }
    return names.get(parameter, parameter)


def value_display_name(parameter, value):
    if parameter == "gamma":
        return rf"$\gamma={value:.0e}$"
    if parameter == "lambda":
        return rf"$\lambda={value:g}$"
    if parameter == "T_w":
        return rf"$T_w={int(value)}$"
    if parameter == "alpha_f":
        return rf"$\alpha_f={value:g}$"
    return f"{parameter}={value}"


def plot_epoch_sensitivity(all_results, parameter, metric, out_dir):
    """Plot metric versus epoch for every tested value of one parameter."""
    sub = all_results[
        (all_results["study_type"] == "1D") &
        (all_results["study_parameter"] == parameter)
    ].copy()

    if sub.empty:
        return

    plt.figure(figsize=(9, 6))

    values = sorted(sub["parameter_numeric"].dropna().unique())
    for value in values:
        curve = sub[sub["parameter_numeric"] == value].sort_values("epoch")
        plt.plot(
            curve["epoch"],
            curve[metric],
            label=value_display_name(parameter, value)
        )

    plt.xlabel("Epoch")
    plt.ylabel(metric.replace("_", " ").title().replace("Acc", "Accuracy (%)"))
    plt.legend(loc="best", frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"epoch_vs_{metric}_{parameter}.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# def plot_1d_sensitivity(summary_df, parameter, metric, out_dir):
#     sub = summary_df[summary_df["study_parameter"] == parameter].copy()
#     if sub.empty:
#         return
#
#     sub = sub.sort_values("parameter_numeric")
#
#     plt.figure(figsize=(8, 5.5))
#     plt.plot(sub["parameter_numeric"], sub[metric], marker="o")
#
#     if parameter in {"gamma", "alpha_f"}:
#         plt.xscale("log")
#
#     plt.xlabel(parameter_display_name(parameter))
#     plt.ylabel(metric.replace("_", " ").title())
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(
#         os.path.join(out_dir, f"sensitivity_{parameter}_{metric}.png"),
#         dpi=300,
#         bbox_inches="tight"
#     )
#     plt.close()


def plot_heatmap(df, row_param, col_param, metric, out_dir):
    pivot = df.pivot(index=row_param, columns=col_param, values=metric)
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)

    values = pivot.values.astype(float)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    image = ax.imshow(values, aspect="auto")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))

    ax.set_xticklabels([str(v) for v in pivot.columns])
    ax.set_yticklabels([str(v) for v in pivot.index])

    ax.set_xlabel(parameter_display_name(col_param))
    ax.set_ylabel(parameter_display_name(row_param))

    fig.colorbar(image, ax=ax)

    midpoint = np.nanmean(values)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            text_color = "white" if value < midpoint else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color)

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            out_dir,
            f"heatmap_{row_param}_vs_{col_param}_{metric}.png"
        ),
        dpi=300
    )
    plt.close()


# ============================================================
# Sensitivity experiment
# ============================================================

def main():
    EPOCHS = 30
    BATCH_SIZE = 128
    NUM_WORKERS = 2
    OUT_DIR = "cifar10_resnet18_adabobr_sensitivity"
    os.makedirs(OUT_DIR, exist_ok=True)

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    # --------------------------------------------------------
    # Baseline/default AdaBoB-R hyperparameters
    # --------------------------------------------------------
    baseline = {
        "lambda": 0.5,
        "T_w": 50,
        "gamma": 1e-3,
        "alpha_f": 0.1
    }

    # --------------------------------------------------------
    # One-factor-at-a-time sensitivity ranges
    # Each parameter varies while the other three remain fixed.
    # --------------------------------------------------------
    sensitivity_values = {
        "lambda": [0.0, 0.25, 0.5, 1.0, 2.0],
        "T_w": [10, 25, 50, 100, 200],
        "gamma": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        "alpha_f": [0.025, 0.05, 0.1, 0.2, 0.4]
    }

    all_histories = []
    one_d_final_rows = []

    print("\n========================================================")
    print("AdaBoB-R One-Dimensional Sensitivity Analysis")
    print("========================================================\n")

    for parameter, values in sensitivity_values.items():
        for value in values:
            cfg = baseline.copy()
            cfg[parameter] = value

            run_name = f"1D_{parameter}_{value}"
            history = train_adabobr(
                train_loader=train_loader,
                test_loader=test_loader,
                epochs=EPOCHS,
                lam=cfg["lambda"],
                warmup_steps=int(cfg["T_w"]),
                gamma=cfg["gamma"],
                final_lr=cfg["alpha_f"],
                run_name=run_name
            )
            history["study_type"] = "1D"
            history["study_parameter"] = parameter
            history["parameter_value"] = value
            history["parameter_numeric"] = float(value)
            all_histories.append(history)

            final_row = history.iloc[-1].to_dict()
            final_row["study_parameter"] = parameter
            final_row["parameter_value"] = value
            final_row["parameter_numeric"] = float(value)
            one_d_final_rows.append(final_row)

    one_d_summary = pd.DataFrame(one_d_final_rows)
    one_d_summary.to_csv(
        os.path.join(OUT_DIR, "one_dimensional_sensitivity_summary.csv"),
        index=False
    )

    # # Plot final-epoch performance versus each parameter.
    # for parameter in sensitivity_values:
    #     for metric in ["test_acc", "macro_f1", "test_loss"]:
    #         plot_1d_sensitivity(one_d_summary, parameter, metric, OUT_DIR)

    # --------------------------------------------------------
    # Convergence sensitivity: Epoch vs Test Accuracy
    # One curve for each tested value of the parameter.
    # --------------------------------------------------------
    one_d_epoch_results = pd.concat(all_histories, ignore_index=True)
    for parameter in sensitivity_values:
        plot_epoch_sensitivity(
            one_d_epoch_results,
            parameter=parameter,
            metric="test_acc",
            out_dir=OUT_DIR
        )

    # --------------------------------------------------------
    # Pairwise heatmaps
    # Covers all four sensitivity parameters with two focused maps.
    # Other parameters remain at baseline values.
    # --------------------------------------------------------
    heatmap_grids = [
        ("lambda", [0.0, 0.5, 1.0, 2.0], "T_w", [10, 50, 100, 200]),
        ("gamma", [1e-4, 1e-3, 5e-3, 1e-2], "alpha_f", [0.025, 0.05, 0.1, 0.2])
    ]

    heatmap_final_rows = []

    print("\n========================================================")
    print("AdaBoB-R Pairwise Sensitivity Heatmaps")
    print("========================================================\n")

    for p1, values1, p2, values2 in heatmap_grids:
        grid_rows = []

        for v1, v2 in itertools.product(values1, values2):
            cfg = baseline.copy()
            cfg[p1] = v1
            cfg[p2] = v2

            run_name = f"HM_{p1}_{v1}_{p2}_{v2}"
            history = train_adabobr(
                train_loader=train_loader,
                test_loader=test_loader,
                epochs=EPOCHS,
                lam=cfg["lambda"],
                warmup_steps=int(cfg["T_w"]),
                gamma=cfg["gamma"],
                final_lr=cfg["alpha_f"],
                run_name=run_name
            )
            history["study_type"] = "heatmap"
            history["study_parameter"] = f"{p1}_vs_{p2}"
            history["parameter_value"] = np.nan
            history["parameter_numeric"] = np.nan
            all_histories.append(history)

            final_row = history.iloc[-1].to_dict()
            final_row["heatmap_pair"] = f"{p1}_vs_{p2}"
            final_row[p1] = v1
            final_row[p2] = v2
            grid_rows.append(final_row)
            heatmap_final_rows.append(final_row)

        grid_df = pd.DataFrame(grid_rows)
        grid_df.to_csv(
            os.path.join(OUT_DIR, f"heatmap_data_{p1}_vs_{p2}.csv"),
            index=False
        )

        for metric in ["test_acc", "test_loss"]:
            plot_heatmap(grid_df, p1, p2, metric, OUT_DIR)

    # --------------------------------------------------------
    # Save all epoch-level results and best settings
    # --------------------------------------------------------
    all_results = pd.concat(all_histories, ignore_index=True)
    all_results.to_csv(
        os.path.join(OUT_DIR, "all_sensitivity_epoch_results.csv"),
        index=False
    )

    heatmap_summary = pd.DataFrame(heatmap_final_rows)
    heatmap_summary.to_csv(
        os.path.join(OUT_DIR, "pairwise_heatmap_summary.csv"),
        index=False
    )

    best_1d = (
        one_d_summary.sort_values("test_acc", ascending=False)
        .groupby("study_parameter", as_index=False)
        .first()
    )
    best_1d.to_csv(
        os.path.join(OUT_DIR, "best_one_dimensional_settings.csv"),
        index=False
    )

    print("\n========================================================")
    print("Best one-dimensional settings by final test accuracy")
    print("========================================================")
    print(
        best_1d[[
            "study_parameter",
            "parameter_value",
            "test_acc",
            "macro_f1",
            "test_loss"
        ]].to_string(index=False)
    )

    print(f"\nSaved all sensitivity results to: {OUT_DIR}")


if __name__ == "__main__":
    main()