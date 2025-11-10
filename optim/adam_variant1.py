import numpy as np
from dataclasses import dataclass

"""

Bu varyant, Adam optimizasyon algoritmasinin başlangiçtaki bias düzeltmesini kaldirir. Normalde Adam, hareketli ortalamalarin (m ve v) başlangiçta 0 olmasindan kaynaklanan bias'i düzeltmek için m_hat ve v_hat hesaplar. Ancak bu varyantta, m ve v'nin başlangiç değerleri ilk gradyan ve gradyanin karesi ile doğrudan başlatilir, böylece bias düzeltme adimlarina gerek kalmaz. BU YÜZDEN MUTLAKA GRADYANIN BÜYÜK, GÜRÜLTÜNÜN KÜÇÜK OLDUĞU NOKTADAN BAŞLANMALI.

"""

@dataclass
class AdamVariant1Config:
    learning_rate: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    max_iters: int = 1000

class AdamVariant1:
    def __init__(self, dim: int, config: AdamVariant1Config | None = None):
        self.config = config or AdamVariant1Config()
        self.m = np.zeros(dim, dtype=float)
        self.v = np.zeros(dim, dtype=float)
        self.t = 0

    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        b1, b2 = self.config.beta1, self.config.beta2
        # değişikliği yapildiği kisim burasi, m0 ve v0 0 ile başlatilsa bile burada değişecek.
        if self.t == 1:
            # İlk adimda, geçmiş (beta*m) olmadiği için m ve v'yi doğrudan başlat.
            self.m = (1.0 - b1) * grad
            self.v = (1.0 - b2) * (grad * grad)
        else:
            # Sonraki adimlarda normal EMA (Exponential Moving Average) güncellemesi
            self.m = b1 * self.m + (1.0 - b1) * grad
            self.v = b2 * self.v + (1.0 - b2) * (grad * grad)

        # Bias correction (m_hat, v_hat) adimlari TAMAMEN KALDIRILDI.Çünkü o adimlar m0 ve v0 0 olarak başlatilirsa anlamli oluyordu.
        update = self.config.learning_rate * self.m / (np.sqrt(self.v) + self.config.epsilon)
        
        return x - update


"""

Hangi senaryoda nasil adim atilmali:

"1. Büyük Gradyan, Düşük Gürültü",""Hizli ve Emin"" -> BÜYÜK ADIM,Yokuş dik ve yönümüzden eminiz (sinyal güçlü). Minimuma hizla yaklaşmak için tam gaz ilerlemeliyiz.
"2. Küçük Gradyan, Düşük Gürültü",""Yavaş ve Emin"" -> KÜÇÜK ADIM,"Yokuş düzleşti ve yönümüz hala tutarli. Bu, minimumun dibine yaklaştiğimizi gösterir. ""Aşiriya kaçmamak"" (overshoot) için yavaşlamaliyiz."
"3. Büyük Gradyan, Yüksek Gürültü",""Hizli ve Kararsiz"" -> KÜÇÜK ADIM,"Yokuş dik, AMA yönümüz sürekli değişiyor. Bu, gürültülü bir yüzeyde (Rastrigin gibi) olduğumuzu gösterir. Büyük bir adim atarsak, yanliş yöne ""ziplayabiliriz"". Gürültünün ortalamasini almak için yavaşlamaliyiz."
"4. Küçük Gradyan, Yüksek Gürültü",""Yavaş ve Kararsiz"" -> ÇOK KÜÇÜK ADIM,"Yokuş düz VE yönümüz belirsiz. Bu, gürültülü bir düzlükte veya eyer noktasinda (saddle point) olabileceğimizi gösterir. Sinyal yok, bu yüzden hareket etmemeliyiz."

"""


""""
Çarpan ->  self.m / (np.sqrt(self.v) + self.config.epsilon)

Varsayilan b1=0.9, b2=0.999 değerleriyle Çarpan ~~3.16'dir. Bu, ilk adimin standart Adam'dan 3 kat daha büyük olacaği anlamina gelir.

Senaryo,AdamWarmStart Davranişi,Etki
"1. Büyük G, Düşük N",AŞIRI BÜYÜK ADIM. Standart Adam'in LR'lik adiminin ≈3.16 katini atar.,"Agresif. Hizli bir başlangiç yapar, ancak minimuma çok yaklaşirsa ""overshoot"" yapabilir (fazla ileri atlayabilir)."
"2. Küçük G, Düşük N",AŞIRI BÜYÜK ADIM. Gradyan küçük olsa bile ≈3.16 * LR büyüklüğünde bir adim atar.,KÖTÜ. Minimumun dibindeyken (Sphere'in merkezi gibi) onu çok uzağa firlatir. Yakinsamayi bozar.
"3. Büyük G, Yüksek N",AŞIRI BÜYÜK VE GÜRÜLTÜLÜ ADIM. Gürültüden kaynaklanan büyük gradyani alir ve onu ≈3.16 ile çarparak yanliş yöne dev bir adim atar.,"FELAKET. Rastrigin gibi gürültülü bir yüzeyde, algoritmayi aninda çok kötü bir bölgeye ""işinlayabilir""."
"4. Küçük G, Yüksek N","AŞIRI BÜYÜK VE GÜRÜLTÜLÜ ADIM. Ayni şekilde, gürültülü yönde ≈3.16 * LR'lik bir adim atar.",FELAKET. Sinyal olmayan yerde dev adimlar atar.

Sonuç: Bu varyant, bias düzeltmesinin ne kadar kritik olduğunu gösterir. O düzeltme olmadan, ilk adimlar b1 ve b2'nin seçimine aşiri duyarli hale gelir ve genellikle çok agresif olur.

"""
