# Deney sonuçları

Cihaz: cpu; seed: 42. Süreler checkpoint kaydını içerir, grafik çizimini içermez.

| Model | Katmanlar | Seçilen epoch | Train loss | Val loss | Train accuracy | Val accuracy | Süre (sn) |
|---|---|---|---|---|---|---|---|
| baseline | [256, 128] | 5 | 0.0426 | 0.0893 | 98.66% | 97.62% | 82.7 |
| compact | [128, 64] | 6 | 0.0456 | 0.0981 | 98.58% | 97.12% | 79.8 |

Validation loss ile seçilen model: **baseline**.
Bu modelin nihai test doğruluğu: **97.59%**; test loss: 0.0841.

Train metrikleri epoch boyunca değişen ağırlıklarla, validation metrikleri epoch sonunda hesaplanır.
İki yapılandırma aynı veri ayrımı ve seed ile karşılaştırılmıştır. Tek seed, istatistiksel üstünlük kanıtlamaz.
El yazısı sonuçları ayrıca handwriting.md dosyasına yazılır; bu görüntüler eğitim veya model seçiminde kullanılmaz.