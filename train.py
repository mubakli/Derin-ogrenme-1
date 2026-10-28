"""MNIST üzerinde sıfırdan MLP eğitimi; test kümesi yalnızca seçimden sonra okunur."""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def make_model(hidden):
    layers = [nn.Flatten()]
    previous = 784
    for width in hidden:
        layers.extend([nn.Linear(previous, width), nn.ReLU()])
        previous = width
    layers.append(nn.Linear(previous, 10))
    # CrossEntropyLoss logit bekler; burada softmax kullanılmaz.
    return nn.Sequential(*layers)


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    loss_sum = correct = count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            count += labels.numel()
            loss_sum += loss.item() * labels.numel()
            correct += (logits.argmax(1) == labels).sum().item()
    return loss_sum / count, correct / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch-size ve lr pozitif olmalıdır")
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    full = datasets.MNIST("data/mnist", train=True, download=True, transform=transform)
    train, val = random_split(full, [50000, 10000], generator=torch.Generator().manual_seed(args.seed))
    results = []
    configurations = {"baseline": [256, 128], "compact": [128, 64]}
    for name, hidden in configurations.items():
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                                  generator=torch.Generator().manual_seed(args.seed))
        val_loader = DataLoader(val, batch_size=512)
        model = make_model(hidden).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        history, best_loss, best_epoch = [], float("inf"), 0
        start = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            train_loss, train_accuracy = run_epoch(model, train_loader, device, optimizer)
            val_loss, val_accuracy = run_epoch(model, val_loader, device)
            row = dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                       train_accuracy=train_accuracy, val_accuracy=val_accuracy)
            history.append(row)
            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch
                torch.save(dict(hidden=hidden, state_dict=model.cpu().state_dict()), out / f"{name}.pt")
                model.to(device)
            print(f"{name} {epoch:02d}: train={train_accuracy:.4f}, val={val_accuracy:.4f}", flush=True)
        seconds = time.perf_counter() - start
        with (out / f"{name}_history.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, metric in zip(axes, ["loss", "accuracy"]):
            for split in ["train", "val"]:
                ax.plot(range(1, args.epochs + 1), [r[f"{split}_{metric}"] for r in history], label=split)
            ax.set(xlabel="Epoch", ylabel=metric, title=name)
            ax.legend()
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / f"{name}_curves.png", dpi=150)
        plt.close(fig)
        results.append(dict(name=name, hidden=hidden, best_epoch=best_epoch,
                            best_epoch_metrics=history[best_epoch - 1], final_epoch_metrics=history[-1],
                            training_seconds=seconds))
    # Yapılandırma ve checkpoint seçimi sadece validation loss ile yapılır.
    selected = min(results, key=lambda r: r["best_epoch_metrics"]["val_loss"])
    checkpoint = torch.load(out / f"{selected['name']}.pt", map_location="cpu", weights_only=True)
    model = make_model(checkpoint["hidden"])
    model.load_state_dict(checkpoint["state_dict"])
    test = datasets.MNIST("data/mnist", train=False, download=True, transform=transform)
    test_loss, test_accuracy = run_epoch(model.to(device), DataLoader(test, batch_size=512), device)
    report = dict(parameters=vars(args), device=str(device), split=dict(train=50000, validation=10000, test=10000),
                  normalization=dict(mean=0.1307, std=0.3081), optimizer="Adam", loss="CrossEntropyLoss",
                  experiments=results, selected=selected["name"], test_loss=test_loss, test_accuracy=test_accuracy)
    (out / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Deney sonuçları", "", f"Cihaz: {device}; seed: {args.seed}. Süreler checkpoint kaydını içerir, grafik çizimini içermez.", "",
             "| Model | Katmanlar | Seçilen epoch | Train loss | Val loss | Train accuracy | Val accuracy | Süre (sn) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        m = r["best_epoch_metrics"]
        lines.append(f"| {r['name']} | {r['hidden']} | {r['best_epoch']} | {m['train_loss']:.4f} | {m['val_loss']:.4f} | {m['train_accuracy']:.2%} | {m['val_accuracy']:.2%} | {r['training_seconds']:.1f} |")
    lines.extend(["", f"Validation loss ile seçilen model: **{selected['name']}**.",
                  f"Bu modelin nihai test doğruluğu: **{test_accuracy:.2%}**; test loss: {test_loss:.4f}.", "",
                  "Train metrikleri epoch boyunca değişen ağırlıklarla, validation metrikleri epoch sonunda hesaplanır.",
                  "İki yapılandırma aynı veri ayrımı ve seed ile karşılaştırılmıştır. Tek seed, istatistiksel üstünlük kanıtlamaz.",
                  "El yazısı sonuçları ayrıca handwriting.md dosyasına yazılır; bu görüntüler eğitim veya model seçiminde kullanılmaz."])
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Seçilen model: {selected['name']}; test accuracy: {test_accuracy:.2%}")


if __name__ == "__main__":
    main()
