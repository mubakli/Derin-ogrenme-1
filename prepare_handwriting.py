"""Gönderilen tam fotoğraf için elle belirlenen kutularla kayıpsız kırpma.

Kutular yalnızca görüntüdeki rakam sınırlarına göre belirlenmiştir;
model tahminleri veya başarı sonuçları kullanılmamıştır.
"""
import json
from pathlib import Path

from PIL import Image, ImageOps


# 2000 x 1500 referans fotoğrafındaki (sol, üst, sağ, alt) koordinatları.
BOXES = [
    (35, 435, 300, 650),
    (300, 450, 485, 625),
    (525, 430, 680, 610),
    (745, 440, 940, 585),
    (950, 440, 1095, 575),
    (1100, 445, 1238, 575),
    (1240, 435, 1375, 580),
    (1395, 450, 1480, 550),
    (1480, 440, 1590, 550),
    (1645, 435, 1735, 530),
    (1735, 430, 1810, 520),
]


def main():
    source = Path("data/handwriting/student_number.png")
    image = ImageOps.exif_transpose(Image.open(source)).convert("RGB")
    destination = source.parent / "digits"
    destination.mkdir(exist_ok=True)
    boxes = []
    for i, box in enumerate(BOXES):
        scaled = tuple(round(value * image.size[j % 2] / (2000, 1500)[j % 2])
                       for j, value in enumerate(box))
        target = destination / f"{i:02d}.png"
        if target.exists():
            raise FileExistsError(f"Mevcut kırpımın üzerine yazılmadı: {target}")
        image.crop(scaled).save(target)
        boxes.append(dict(file=target.name, box=scaled))
    (source.parent / "crop_boxes.json").write_text(
        json.dumps(dict(source=source.name, size=image.size, crops=boxes), indent=2), encoding="utf-8")
    print(f"{len(boxes)} rakam özgün fotoğraftan kırpıldı.")


if __name__ == "__main__":
    main()
