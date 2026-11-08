# compare_optimizers_tinyimagenet_resnet18_train_val_auto_download.py

import os
import random
import shutil
import zipfile
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

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

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Automatic Tiny ImageNet Download
# ============================================================

def download_tiny_imagenet(data_dir="./data"):
    """
    Downloads and extracts Tiny ImageNet automatically.

    Final folder:
        ./data/tiny-imagenet-200/
    """

    os.makedirs(data_dir, exist_ok=True)

    dataset_dir = os.path.join(data_dir, "tiny-imagenet-200")
    zip_path = os.path.join(data_dir, "tiny-imagenet-200.zip")

    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

    if os.path.exists(dataset_dir):
        print(f"Tiny ImageNet already exists at: {dataset_dir}")
        return dataset_dir

    if not os.path.exists(zip_path):
        print("Downloading Tiny ImageNet...")
        print(f"URL: {url}")
        print(f"Saving to: {zip_path}")

        urllib.request.urlretrieve(url, zip_path)

        print("Download finished.")
    else:
        print(f"Zip file already exists at: {zip_path}")

    print("Extracting Tiny ImageNet...")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    print(f"Extraction finished. Dataset folder: {dataset_dir}")

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(
            f"Extraction failed. Could not find: {dataset_dir}"
        )

    return dataset_dir


# ============================================================
# Prepare Tiny ImageNet Validation Folder
# ============================================================

def prepare_tiny_imagenet_val_folder(data_root):
    """
    Tiny ImageNet validation images originally have this structure:

        val/images/*.JPEG
        val/val_annotations.txt

    ImageFolder requires this structure:

        val/class_name/*.JPEG

    This function converts val/ into ImageFolder format.
    It is safe to run multiple times.
    """

    val_dir = os.path.join(data_root, "val")
    val_images_dir = os.path.join(val_dir, "images")
    annotations_file = os.path.join(val_dir, "val_annotations.txt")

    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Validation folder not found: {val_dir}")

    if not os.path.exists(annotations_file):
        print("val_annotations.txt not found. Validation folder may already be prepared.")
        return

    if not os.path.exists(val_images_dir):
        print("val/images folder not found. Validation folder may already be prepared.")
        return

    image_to_class = {}

    with open(annotations_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            image_name = parts[0]
            class_name = parts[1]
            image_to_class[image_name] = class_name

    for image_name, class_name in image_to_class.items():
        class_dir = os.path.join(val_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        src = os.path.join(val_images_dir, image_name)
        dst = os.path.join(class_dir, image_name)

        if os.path.exists(src) and not os.path.exists(dst):
            shutil.move(src, dst)

    if os.path.exists(val_images_dir) and len(os.listdir(val_images_dir)) == 0:
        shutil.rmtree(val_images_dir)

    print("Tiny ImageNet validation folder prepared for ImageFolder.")


# ============================================================
# Tiny ImageNet Data
# ============================================================

def get_tinyimagenet_loaders(
    data_root="./data/tiny-imagenet-200",
    batch_size=128,
    num_workers=2,
    auto_download=True
):
    if not os.path.exists(data_root):
        if auto_download:
            data_dir = os.path.dirname(data_root)

            if data_dir == "":
                data_dir = "./data"

            data_root = download_tiny_imagenet(data_dir=data_dir)
        else:
            raise FileNotFoundError(
                f"Tiny ImageNet folder not found: {data_root}\n"
                f"Expected path: ./data/tiny-imagenet-200"
            )

    prepare_tiny_imagenet_val_folder(data_root)

    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4802, 0.4481, 0.3975),
            std=(0.2302, 0.2265, 0.2262)
        )
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4802, 0.4481, 0.3975),
            std=(0.2302, 0.2265, 0.2262)
        )
    ])

    train_set = ImageFolder(
        root=train_dir,
        transform=transform_train
    )

    val_set = ImageFolder(
        root=val_dir,
        transform=transform_val
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Training images: {len(train_set)}")
    print(f"Validation images: {len(val_set)}")
    print(f"Number of classes: {len(train_set.classes)}")

    return train_loader, val_loader


# ============================================================
# ResNet-18 for Tiny ImageNet
# ============================================================

def build_resnet18_tinyimagenet(num_classes=200):
    model = models.resnet18(weights=None)

    # Tiny ImageNet images are 64x64.
    # This CIFAR-style first layer keeps more spatial detail.
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
# AdaBound Optimizer
# ============================================================

class AdaBound(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        final_lr=0.1,
        betas=(0.9, 0.999),
        gamma=1e-3,
        eps=1e-8,
        weight_decay=0.0
    ):
        defaults = dict(
            lr=lr,
            final_lr=final_lr,
            betas=betas,
            gamma=gamma,
            eps=eps,
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
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                alpha_t = lr * np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

                lower = final_lr * (1 - 1 / (gamma * t + 1))
                upper = final_lr * (1 + 1 / (gamma * t))

                step_size = alpha_t / (torch.sqrt(v) + eps)
                step_size = torch.clamp(step_size, min=lower, max=upper)
                # step_size = step_size / np.sqrt(t)
                step_size = step_size

                p.addcmul_(m, step_size, value=-1.0)

        return loss


# ============================================================
# AdaBelief Optimizer
# ============================================================

class AdaBelief(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
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
            beta1, beta2 = group["betas"]
            eps = group["eps"]
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

                belief_residual = grad - m

                v.mul_(beta2).addcmul_(
                    belief_residual,
                    belief_residual,
                    value=1 - beta2
                )

                alpha_t = lr * np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

                update = m / (torch.sqrt(v) + eps)
                p.add_(update, alpha=-alpha_t)

        return loss


# ============================================================
# AdaBoB Optimizer
# AdaBoB = AdaBelief variance + AdaBound bounds
# ============================================================

class AdaBoB(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        final_lr=0.1,
        betas=(0.9, 0.999),
        gamma=1e-3,
        eps=1e-8,
        weight_decay=0.0
    ):
        defaults = dict(
            lr=lr,
            final_lr=final_lr,
            betas=betas,
            gamma=gamma,
            eps=eps,
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

                belief_residual = grad - m

                v.mul_(beta2).addcmul_(
                    belief_residual,
                    belief_residual,
                    value=1 - beta2
                )

                alpha_t = lr * np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

                lower = final_lr * (1 - 1 / (gamma * t + 1))
                upper = final_lr * (1 + 1 / (gamma * t))

                step_size = alpha_t / (torch.sqrt(v) + eps)
                step_size = torch.clamp(step_size, min=lower, max=upper)
                # step_size = step_size / np.sqrt(t)
                step_size = step_size

                p.addcmul_(m, step_size, value=-1.0)

        return loss


class AdaDB(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        final_lr=0.1,
        betas=(0.9, 0.999),
        gamma=1e-3,
        weight_decay=0.0
    ):
        defaults = dict(
            lr=lr,
            final_lr=final_lr,
            betas=betas,
            gamma=gamma,
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

                # m_t
                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                # v_t
                v.mul_(beta2).addcmul_(
                    grad,
                    grad,
                    value=1 - beta2
                )

                # r_t
                m_inf = torch.max(torch.abs(m))

                r_t = torch.abs(m) / (
                    m_inf * gamma * t
                )

                # eta_l(t), eta_u(t)
                eta_l = final_lr
                eta_u = final_lr + r_t

                # Bias-corrected moments
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                # alpha_t / sqrt(v_hat)
                step_size = lr / torch.sqrt(v_hat)

                # max{alpha_t/sqrt(v_hat), eta_l(t)}
                step_size = torch.maximum(
                    step_size,
                    torch.full_like(step_size, eta_l)
                )

                # min{..., eta_u(t)}
                step_size = torch.minimum(
                    step_size,
                    eta_u
                )

                # Parameter update
                p.addcmul_(
                    m_hat,
                    step_size,
                    value=-1.0
                )

        return loss

# ============================================================
# AdaBoB-R Optimizer
# Robust AdaBoB with residual outlier control
# ============================================================

class AdaBoBR(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        final_lr=0.1,
        betas=(0.9, 0.999),
        gamma=1e-4,
        eps=1e-8,
        lam=0.5,
        warmup_steps=10,
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

                # step_size = step_size / np.sqrt(t)
                step_size = step_size

                p.addcmul_(m, step_size, value=-1.0)

        return loss


# ============================================================
# Optimizer Factory
# ============================================================

def make_optimizer(name, model):
    name = name.lower()

    weight_decay = 5e-4

    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=weight_decay
        )

    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=1e-3,
            weight_decay=weight_decay
        )

    if name == "radam":
        return torch.optim.RAdam(
            model.parameters(),
            lr=1e-3,
            weight_decay=weight_decay
        )

    if name == "adabound":
        return AdaBound(
            model.parameters(),
            lr=1e-3,
            final_lr=0.1,
            gamma=1e-3,
            weight_decay=weight_decay
        )

    if name == "adabelief":
        return AdaBelief(
            model.parameters(),
            lr=1e-3,
            weight_decay=weight_decay
        )

    if name == "adabob":
        return AdaBoB(
            model.parameters(),
            lr=1e-3,
            final_lr=0.1,
            gamma=1e-3,
            weight_decay=weight_decay
        )

    if name == "adadb":
        return AdaDB(
            model.parameters(),
            lr=1e-3,
            final_lr=0.1,
            betas=(0.9, 0.999),
            gamma=1e-4,
            weight_decay=weight_decay
        )

    if name == "adabob-r":
        return AdaBoBR(
            model.parameters(),
            lr=1e-6,
            final_lr=0.1,
            gamma=1e-3,
            lam=0.5,
            warmup_steps=500,
            weight_decay=weight_decay
        )

    raise ValueError(f"Unknown optimizer: {name}")


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

    precision *= 100.0
    recall *= 100.0
    f1 *= 100.0

    return avg_loss, acc, precision, recall, f1


# ============================================================
# Training
# ============================================================

def train_one_optimizer(
    optimizer_name,
    train_loader,
    val_loader,
    epochs=30
):
    model = build_resnet18_tinyimagenet(num_classes=200).to(DEVICE)
    optimizer = make_optimizer(optimizer_name, model)
    criterion = nn.CrossEntropyLoss()

    scheduler = None

    if optimizer_name.lower() == "sgd":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[25, 40],
            gamma=0.1
        )

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

        if scheduler is not None:
            scheduler.step()

        train_loss = train_loss_sum / len(train_loader.dataset)
        train_acc = accuracy_score(train_targets, train_preds) * 100.0

        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate(
            model,
            val_loader,
            criterion
        )

        history.append({
            "optimizer": optimizer_name,
            "epoch": epoch,

            "train_loss": train_loss,
            "val_loss": val_loss,

            "train_acc": train_acc,
            "val_acc": val_acc,

            "val_macro_precision": val_precision,
            "val_macro_recall": val_recall,
            "val_macro_f1": val_f1
        })

        print(
            f"{optimizer_name:10s} | "
            f"epoch {epoch:03d} | "
            f"Train Loss {train_loss:.4f} | "
            f"Val Loss {val_loss:.4f} | "
            f"Train Accuracy {train_acc:.2f}% | "
            f"Val Accuracy {val_acc:.2f}% | "
            f"Val Macro P {val_precision:.2f}% | "
            f"Val Macro R {val_recall:.2f}% | "
            f"Val Macro F1 {val_f1:.2f}%"
        )

    return pd.DataFrame(history)


# ============================================================
# Plotting
# ============================================================

def plot_metric(df, metric, ylabel, save_path):
    plt.figure(figsize=(9, 6))

    for opt_name in df["optimizer"].unique():
        sub = df[df["optimizer"] == opt_name]
        plt.plot(sub["epoch"], sub[metric], label=opt_name)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)

    # No plot title
    # plt.title(...)

    # Automatic y-axis scaling

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_final_macro_metrics(df, save_path):
    final_df = (
        df.sort_values("epoch")
        .groupby("optimizer")
        .tail(1)
        .reset_index(drop=True)
    )

    optimizer_order = [
        "AdaBoB-R",
        "AdaBound",
        "AdaBelief",
        "AdaBoB",
        "AdaDB",
        "SGD",
        "Adam",
        "RAdam"
    ]

    final_df["optimizer"] = pd.Categorical(
        final_df["optimizer"],
        categories=optimizer_order,
        ordered=True
    )

    final_df = final_df.sort_values("optimizer")

    x = np.arange(len(final_df))
    width = 0.25

    plt.figure(figsize=(11, 6))

    plt.bar(
        x - width,
        final_df["val_macro_precision"],
        width,
        label="Precision"
    )

    plt.bar(
        x,
        final_df["val_macro_recall"],
        width,
        label="Recall"
    )

    plt.bar(
        x + width,
        final_df["val_macro_f1"],
        width,
        label="F1-score"
    )

    plt.xlabel("Optimizer")
    plt.ylabel("Score (%)")

    plt.xticks(
        x,
        final_df["optimizer"],
        rotation=25,
        ha="right"
    )

    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()

def make_all_plots(results_df, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    plot_metric(
        results_df,
        "train_loss",
        "Train Loss",
        os.path.join(out_dir, "train_loss.png")
    )

    plot_metric(
        results_df,
        "val_loss",
        "Val Loss",
        os.path.join(out_dir, "val_loss.png")
    )

    plot_metric(
        results_df,
        "train_acc",
        "Train Accuracy (%)",
        os.path.join(out_dir, "train_accuracy.png")
    )

    plot_metric(
        results_df,
        "val_acc",
        "Validation Accuracy (%)",
        os.path.join(out_dir, "val_accuracy.png")
    )

    # Final-epoch grouped bar plot:
    # Macro Precision, Macro Recall, and Macro F1
    plot_final_macro_metrics(
        results_df,
        os.path.join(out_dir, "final_macro_metrics.png")
    )


# ============================================================
# Final Summary Table
# ============================================================

def final_epoch_summary(df):
    final_df = (
        df.sort_values("epoch")
        .groupby("optimizer")
        .tail(1)
        .reset_index(drop=True)
    )

    cols = [
        "optimizer",
        "train_loss",
        "val_loss",
        "train_acc",
        "val_acc",
        "val_macro_precision",
        "val_macro_recall",
        "val_macro_f1"
    ]

    return final_df[cols].sort_values("val_macro_f1", ascending=False)


# ============================================================
# Main
# ============================================================

def main():
    EPOCHS = 40
    BATCH_SIZE = 128

    # On Windows/PyCharm, num_workers=0 is safer.
    # If your system works well, you can change it to 2 or 4.
    NUM_WORKERS = 2

    DATA_ROOT = "./data/tiny-imagenet-200"
    OUT_DIR = "tinyimagenet_resnet18_train_val_results"

    train_loader, val_loader = get_tinyimagenet_loaders(
        data_root=DATA_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        auto_download=True
    )

    optimizers = [
        "AdaBoB-R",
        "AdaBound",
        "AdaBelief",
        "AdaBoB",
        "AdaDB",
        "SGD",
        "Adam",
        "RAdam"
    ]

    results = []

    print("\n==========================================")
    print("Tiny ImageNet ResNet-18 Optimizer Comparison")
    print("Train and Validation Evaluation")
    print("==========================================\n")

    for opt_name in optimizers:
        df = train_one_optimizer(
            opt_name,
            train_loader,
            val_loader,
            epochs=EPOCHS
        )
        results.append(df)

    results_df = pd.concat(results, ignore_index=True)

    os.makedirs(OUT_DIR, exist_ok=True)

    results_df.to_csv(
        os.path.join(OUT_DIR, "results.csv"),
        index=False
    )

    make_all_plots(results_df, OUT_DIR)

    summary = final_epoch_summary(results_df)

    summary.to_csv(
        os.path.join(OUT_DIR, "final_summary.csv"),
        index=False
    )

    print("\n==========================================")
    print("Final Tiny ImageNet ResNet-18 Summary")
    print("==========================================")
    print(summary.to_string(index=False))

    print(f"\nSaved all plots and CSV files to: {OUT_DIR}")


if __name__ == "__main__":
    main()