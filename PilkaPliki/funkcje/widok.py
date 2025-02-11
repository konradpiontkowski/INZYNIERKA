from typing import Tuple
import cv2
import numpy as np
import numpy.typing as npt


class TransformatorWidoku:
    def __init__(
            self,
            zrodlo: npt.NDArray[np.float32],
            cel: npt.NDArray[np.float32]
    ) -> None:
        if zrodlo.shape != cel.shape:
            raise ValueError("Źródło i cel muszą mieć ten sam kształt.")
        if zrodlo.shape[1] != 2:
            raise ValueError("Źródło i punkty docelowe muszą być współrzędnymi 2D.")

        zrodlo = zrodlo.astype(np.float32)
        cel = cel.astype(np.float32)
        self.m, _ = cv2.findHomography(zrodlo, cel)
        if self.m is None:
            raise ValueError("Macierz homografii nie mogła zostać obliczona.")

    def transformuj_punkty(
            self,
            punkty: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        if punkty.size == 0:
            return punkty

        if punkty.shape[1] != 2:
            raise ValueError("Punkty muszą być współrzędnymi 2D.")

        przeksztalcone_punkty = punkty.reshape(-1, 1, 2).astype(np.float32)
        punkty_transformed = cv2.perspectiveTransform(przeksztalcone_punkty, self.m)
        return punkty_transformed.reshape(-1, 2).astype(np.float32)

    def transformuj_obraz(
            self,
            obraz: npt.NDArray[np.uint8],
            rozdzielczosc_wh: Tuple[int, int]
    ) -> npt.NDArray[np.uint8]:
        if len(obraz.shape) not in {2, 3}:
            raise ValueError("Obraz musi być w skali szarości lub kolorowy.")
        return cv2.warpPerspective(obraz, self.m, rozdzielczosc_wh)