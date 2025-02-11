from collections import deque

import cv2
import numpy as np
import supervision as sv


class OznacznikPilki:
   
    def __init__(self, promien: int, rozmiar_bufora: int = 5, grubosc: int = 2):
        self.paleta_kolorow = sv.ColorPalette.from_matplotlib('jet', rozmiar_bufora)
        self.bufor = deque(maxlen=rozmiar_bufora)
        self.promien = promien
        self.grubosc = grubosc

    def interpoluj_promien(self, i: int, max_i: int) -> int:
        if max_i == 1:
            return self.promien
        return int(1 + i * (self.promien - 1) / (max_i - 1))

    def oznacz(self, ramka: np.ndarray, detekcje: sv.Detections) -> np.ndarray:
        xy = detekcje.get_anchors_coordinates(sv.Position.BOTTOM_CENTER).astype(int)
        self.bufor.append(xy)
        for i, xy in enumerate(self.bufor):
            kolor = self.paleta_kolorow.by_idx(i)
            interpolowany_promien = self.interpoluj_promien(i, len(self.bufor))
            for srodek in xy:
                ramka = cv2.circle(
                    img=ramka,
                    center=tuple(srodek),
                    radius=interpolowany_promien,
                    color=kolor.as_bgr(),
                    thickness=self.grubosc
                )
        return ramka


class SledzeniePilki:
   
    def __init__(self, rozmiar_bufora: int = 10):
        self.bufor = deque(maxlen=rozmiar_bufora)

    def aktualizuj(self, detekcje: sv.Detections) -> sv.Detections:
        xy = detekcje.get_anchors_coordinates(sv.Position.CENTER)
        self.bufor.append(xy)

        if len(detekcje) == 0:
            return detekcje

        centroid = np.mean(np.concatenate(self.bufor), axis=0)
        odleglosci = np.linalg.norm(xy - centroid, axis=1)
        indeks = np.argmin(odleglosci)
        return detekcje[[indeks]]
