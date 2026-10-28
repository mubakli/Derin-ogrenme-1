"""Kullanıcının ayrı kırptığı rakamları MNIST formatına çevirip değerlendirir."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

from train import make_model


def preprocess(path):
    image = Image.open(path).convert("L")
    # Yerel arka plan kestirimi, fotoğraftaki yavaş değişen gölgeleri azaltır.
    image.thumbnail((1000, 1000))
    gray = np.asarray(image, dtype=np.float32)
    background = np.asarray(image.filter(ImageFilter.GaussianBlur(max(image.size) / 15)), dtype=np.float32)
    ink = np.clip(background - gray, 0, None)
    peak = float(ink.max())
    if peak < 5:
        raise ValueError(f"Rakam bulunamadı: {path}")
    ink[ink < max(5, peak * 0.15)] = 0
    ink = np.uint8(np.clip(ink / peak * 255, 0, 255))
    digit = Image.fromarray(ink)
    box = digit.getbbox()
    if box is None:
        raise ValueError(f"Boş görüntü: {path}")
    digit = digit.crop(box)
    digit.thumbnail((20, 20), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28))
    canvas.paste(digit, ((28 - digit.width) // 2, (28 - digit.height) // 2))
    # MNIST benzeri kütle merkezi hizalaması; kaydırmada sarma yapılmaz.
    array = np.asarray(canvas, dtype=np.float32)
    yy, xx = np.indices(array.shape)
    mass = array.sum()
    dx = round(13.5 - (xx * array).sum() / mass)
    dy = round(13.5 - (yy * array).sum() / mass)
    centered = Image.new("L", (28, 28))
    centered.paste(canvas, (dx, dy))
    tensor = torch.from_numpy(np.array(centered, dtype=np.float32) / 255)
    return centered, ((tensor - 0.1307) / 0.3081).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--digits", default="data/handwriting/digits")
    parser.add_argument("--labels", default="23120205038")
    args = parser.parse_args()
    files = sorted(p for p in Path(args.digits).glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if len(files) != len(args.labels) or not args.labels.isdigit():
        parser.error("Her rakam için bir görüntü gerekli: 00.png ... 10.png; etiket sayısı dosya sayısına eşit olmalı.")
    out = Path("outputs")
    results = json.loads((out / "results.json").read_text())
    checkpoint = torch.load(out / f"{results['selected']}.pt", map_location="cpu", weights_only=True)
    model = make_model(checkpoint["hidden"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    processed = out / "handwriting"
    processed.mkdir(exist_ok=True)
    rows, predictions = [], []
    montage = Image.new("L", (28 * len(files), 28))
    for i, (path, label) in enumerate(zip(files, args.labels)):
        image, tensor = preprocess(path)
        image.save(processed / f"{i:02d}.png")
        montage.paste(image, (28 * i, 0))
        with torch.no_grad():
            probabilities = model(tensor.unsqueeze(0)).softmax(1)[0]
        prediction = int(probabilities.argmax())
        predictions.append(str(prediction))
        rows.append(f"| {i + 1} | {label} | {prediction} | {probabilities[prediction].item():.2%} |")
    montage.resize((28 * len(files) * 4, 112)).save(processed / "montage.png")
    correct = sum(a == b for a, b in zip(args.labels, predictions))
    text = ["# Kendi el yazısı değerlendirmesi", "", f"Gerçek: `{args.labels}`; tahmin: `{''.join(predictions)}`.",
            f"Rakam bazında doğruluk: {correct}/{len(files)} = {correct / len(files):.2%}.", "",
            "| Sıra | Gerçek | Tahmin | Softmax skoru |", "|---|---|---|---|"] + rows
    text += ["", "![Ön işlenen rakamlar](handwriting/montage.png)", "",
             "Softmax skorları kalibre edilmiş güven ölçümü değildir. Tek öğrenci numarası genel başarıyı göstermez.",
             "Fotoğraftaki ince çizgiler, ışık/gölge, farklı rakam boyutları ve yazım biçimi MNIST'ten farklıdır; hatalar bu dağılım farkı ve ön işleme kayıplarıyla ilişkili olabilir.",
             "Bu örnek model eğitimine veya model seçimine dahil edilmemiştir. Sonuçlar görüldükten sonra ön işleme değiştirilirse bu ayrıca raporlanmalıdır."]
    (out / "handwriting.md").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text[:4]))


if __name__ == "__main__":
    main()
