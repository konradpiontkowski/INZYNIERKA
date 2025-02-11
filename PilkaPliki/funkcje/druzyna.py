from typing import Generator, Iterable, List, TypeVar

import numpy as np
import supervision as sv
import torch
import umap
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import AutoProcessor, SiglipVisionModel

V = TypeVar("V")

SCIEZKA_MODELU_SIGLIP = 'google/siglip-base-patch16-224'


def stworz_paczki(
    sekwencja: Iterable[V], rozmiar_paczki: int
) -> Generator[List[V], None, None]:
    rozmiar_paczki = max(rozmiar_paczki, 1)
    aktualna_paczka = []
    for element in sekwencja:
        if len(aktualna_paczka) == rozmiar_paczki:
            yield aktualna_paczka
            aktualna_paczka = []
        aktualna_paczka.append(element)
    if aktualna_paczka:
        yield aktualna_paczka


class KlasyfikatorDruzyn:
    def __init__(self, urzadzenie: str = 'cpu', rozmiar_paczki: int = 32):
        self.urzadzenie = urzadzenie
        self.rozmiar_paczki = rozmiar_paczki
        self.model_cech = SiglipVisionModel.from_pretrained(
            SCIEZKA_MODELU_SIGLIP).to(urzadzenie)
        self.procesor = AutoProcessor.from_pretrained(SCIEZKA_MODELU_SIGLIP)
        self.reducer = umap.UMAP(n_components=3)
        self.model_klastrujacy = KMeans(n_clusters=2)

    def wyodrebnij_cechy(self, crops: List[np.ndarray]) -> np.ndarray:
        crops = [sv.cv2_to_pillow(crops) for crops in crops]
        paczki = stworz_paczki(crops, self.rozmiar_paczki)
        dane = []
        with torch.no_grad():
            for paczka in tqdm(paczki, desc='Ekstrakcja osadzeń'):
                wektory = self.procesor(
                    images=paczka, return_tensors="pt").to(self.urzadzenie)
                wyniki = self.model_cech(**wektory)
                osadzenia = torch.mean(wyniki.last_hidden_state, dim=1).cpu().numpy()
                dane.append(osadzenia)

        return np.concatenate(dane)

    def dopasuj(self, crops: List[np.ndarray]) -> None:
        dane = self.wyodrebnij_cechy(crops)
        projekcje = self.reducer.fit_transform(dane)
        self.model_klastrujacy.fit(projekcje)

    def przewiduj(self, crops: List[np.ndarray]) -> np.ndarray:
        if len(crops) == 0:
            return np.array([])

        dane = self.wyodrebnij_cechy(crops)
        projekcje = self.reducer.transform(dane)
        return self.model_klastrujacy.predict(projekcje)
