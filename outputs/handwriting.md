# Kendi el yazısı değerlendirmesi

Gerçek: `23120205038`; tahmin: `55555505232`.
Rakam bazında doğruluk: 3/11 = 27.27%.

| Sıra | Gerçek | Tahmin | Softmax skoru |
|---|---|---|---|
| 1 | 2 | 5 | 99.09% |
| 2 | 3 | 5 | 77.93% |
| 3 | 1 | 5 | 79.23% |
| 4 | 2 | 5 | 89.52% |
| 5 | 0 | 5 | 62.89% |
| 6 | 2 | 5 | 72.93% |
| 7 | 0 | 0 | 48.98% |
| 8 | 5 | 5 | 99.07% |
| 9 | 0 | 2 | 37.62% |
| 10 | 3 | 3 | 92.21% |
| 11 | 8 | 2 | 78.06% |

![Ön işlenen rakamlar](handwriting/montage.png)

Softmax skorları kalibre edilmiş güven ölçümü değildir. Tek öğrenci numarası genel başarıyı göstermez.
Fotoğraftaki ince çizgiler, ışık/gölge, farklı rakam boyutları ve yazım biçimi MNIST'ten farklıdır; hatalar bu dağılım farkı ve ön işleme kayıplarıyla ilişkili olabilir.
Bu örnek model eğitimine veya model seçimine dahil edilmemiştir. Sonuçlar görüldükten sonra ön işleme değiştirilirse bu ayrıca raporlanmalıdır.