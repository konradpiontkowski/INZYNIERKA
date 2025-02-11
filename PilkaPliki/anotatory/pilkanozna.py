from typing import Optional, List

import cv2
import supervision as sv
import numpy as np

from PilkaPliki.konfiguracja.pilkakonfig import KonfiguracjaBoiska


def rysuj_boisko(
    konfiguracja: KonfiguracjaBoiska,
    kolor_tla: sv.Color = sv.Color(11, 214, 11),
    kolor_linii: sv.Color = sv.Color(255, 255, 255),
    margines: int = 50,
    grubosc_linii: int = 4,
    promien_punktu: int = 8,
    skalowanie: float = 0.1
) -> np.ndarray:
    szerokosc_skalowana = int(konfiguracja.szerokosc * skalowanie)
    dlugosc_skalowana = int(konfiguracja.dlugosc * skalowanie)
    promien_kola_centrowego_skalowany = int(konfiguracja.promien_kola_srodkowego * skalowanie)
    dystans_punktu_karnego_skalowany = int(konfiguracja.odleglosc_od_punktu_karnego * skalowanie)

    obraz_boiska = np.ones(
        (szerokosc_skalowana + 2 * margines,
         dlugosc_skalowana + 2 * margines, 3),
        dtype=np.uint8
    ) * np.array(kolor_tla.as_bgr(), dtype=np.uint8)

    for poczatek, koniec in konfiguracja.krawedzie:
        punkt1 = (int(konfiguracja.wierzcholki[poczatek - 1][0] * skalowanie) + margines,
                  int(konfiguracja.wierzcholki[poczatek - 1][1] * skalowanie) + margines)
        punkt2 = (int(konfiguracja.wierzcholki[koniec - 1][0] * skalowanie) + margines,
                  int(konfiguracja.wierzcholki[koniec - 1][1] * skalowanie) + margines)
        cv2.line(
            img=obraz_boiska,
            pt1=punkt1,
            pt2=punkt2,
            color=kolor_linii.as_bgr(),
            thickness=grubosc_linii
        )

    srodek_kola_centrowego = (
        dlugosc_skalowana // 2 + margines,
        szerokosc_skalowana // 2 + margines
    )
    cv2.circle(
        img=obraz_boiska,
        center=srodek_kola_centrowego,
        radius=promien_kola_centrowego_skalowany,
        color=kolor_linii.as_bgr(),
        thickness=grubosc_linii
    )

    punkty_karne = [
        (
            dystans_punktu_karnego_skalowany + margines,
            szerokosc_skalowana // 2 + margines
        ),
        (
            dlugosc_skalowana - dystans_punktu_karnego_skalowany + margines,
            szerokosc_skalowana // 2 + margines
        )
    ]
    for punkt in punkty_karne:
        cv2.circle(
            img=obraz_boiska,
            center=punkt,
            radius=promien_punktu,
            color=kolor_linii.as_bgr(),
            thickness=-1
        )

    return obraz_boiska


def rysuj_punkty_na_boisku(
    konfiguracja: KonfiguracjaBoiska,
    xy: np.ndarray,
    kolor_wypelnienia: sv.Color = sv.Color.RED,
    kolor_obrysu: sv.Color = sv.Color.BLACK,
    promien: int = 10,
    grubosc: int = 2,
    margines: int = 50,
    skalowanie: float = 0.1,
    boisko: Optional[np.ndarray] = None
) -> np.ndarray:
   
    if boisko is None:
        boisko = rysuj_boisko(
            konfiguracja=konfiguracja,
            margines=margines,
            skalowanie=skalowanie
        )

    for punkt in xy:
        punkt_skalowany = (
            int(punkt[0] * skalowanie) + margines,
            int(punkt[1] * skalowanie) + margines
        )
        cv2.circle(
            img=boisko,
            center=punkt_skalowany,
            radius=promien,
            color=kolor_wypelnienia.as_bgr(),
            thickness=-1
        )
        cv2.circle(
            img=boisko,
            center=punkt_skalowany,
            radius=promien,
            color=kolor_obrysu.as_bgr(),
            thickness=grubosc
        )

    return boisko


def rysuj_sciezki_na_boisku(
    konfiguracja: KonfiguracjaBoiska,
    sciezki: List[np.ndarray],
    kolor: sv.Color = sv.Color.RED,
    grubosc: int = 3,
    margines: int = 50,
    skalowanie: float = 0.1,
    boisko: Optional[np.ndarray] = None
) -> np.ndarray:
    
    if boisko is None:
        boisko = rysuj_boisko(
            konfiguracja=konfiguracja,
            margines=margines,
            skalowanie=skalowanie
        )

    for sciezka in sciezki:
        sciezka_skalowana = [
            (
                int(punkt[0] * skalowanie) + margines,
                int(punkt[1] * skalowanie) + margines
            )
            for punkt in sciezka if punkt.size > 0
        ]

        if len(sciezka_skalowana) < 2:
            continue

        for i in range(len(sciezka_skalowana) - 1):
            cv2.line(
                img=boisko,
                pt1=sciezka_skalowana[i],
                pt2=sciezka_skalowana[i + 1],
                color=kolor.as_bgr(),
                thickness=grubosc
            )

        return boisko


def rysuj_diagram_voronoia(
    konfiguracja: KonfiguracjaBoiska,
    team_1_xy: np.ndarray,
    team_2_xy: np.ndarray,
    kolor_druzyny_1: sv.Color = sv.Color.RED,
    kolor_druzyny_2: sv.Color = sv.Color.WHITE,
    przezroczystosc: float = 0.5,
    margines: int = 50,
    skalowanie: float = 0.1,
    boisko: Optional[np.ndarray] = None
) -> np.ndarray:
   
    if boisko is None:
        boisko = rysuj_boisko(
            konfiguracja=konfiguracja,
            margines=margines,
            skalowanie=skalowanie
        )

    szerokosc_skalowana = int(konfiguracja.szerokosc * skalowanie)
    dlugosc_skalowana = int(konfiguracja.dlugosc * skalowanie)

    voronoi = np.zeros_like(boisko, dtype=np.uint8)

    kolor_druzyny_1_bgr = np.array(kolor_druzyny_1.as_bgr(), dtype=np.uint8)
    kolor_druzyny_2_bgr = np.array(kolor_druzyny_2.as_bgr(), dtype=np.uint8)

    wsp_y, wsp_x = np.indices((
        szerokosc_skalowana + 2 * margines,
        dlugosc_skalowana + 2 * margines
    ))

    wsp_y -= margines
    wsp_x -= margines

    def oblicz_dystanse(xy, wsp_x, wsp_y):
        return np.sqrt((xy[:, 0][:, None, None] * skalowanie - wsp_x) ** 2 +
                       (xy[:, 1][:, None, None] * skalowanie - wsp_y) ** 2)

    dystanse_druzyny_1 = oblicz_dystanse(team_1_xy, wsp_x, wsp_y)
    dystanse_druzyny_2 = oblicz_dystanse(team_2_xy, wsp_x, wsp_y)
    min_dystanse_druzyny_1 = np.min(dystanse_druzyny_1, axis=0)
    min_dystanse_druzyny_2 = np.min(dystanse_druzyny_2, axis=0)
    maska_kontroli = min_dystanse_druzyny_1 < min_dystanse_druzyny_2

    voronoi[maska_kontroli] = kolor_druzyny_1_bgr
    voronoi[~maska_kontroli] = kolor_druzyny_2_bgr

    nalozenie = cv2.addWeighted(voronoi, przezroczystosc, boisko, 1 - przezroczystosc, 0)

    return nalozenie
