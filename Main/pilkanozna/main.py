import argparse
from enum import Enum
from typing import Iterator, List

import os
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from PilkaPliki.anotatory.pilkanozna import rysuj_boisko, rysuj_punkty_na_boisku
from PilkaPliki.funkcje.pilka import SledzeniePilki, OznacznikPilki
from PilkaPliki.funkcje.druzyna import KlasyfikatorDruzyn
from PilkaPliki.funkcje.widok import TransformatorWidoku
from PilkaPliki.konfiguracja.pilkakonfig import KonfiguracjaBoiska

GLOWNY_KATALOG = os.path.dirname(os.path.abspath(__file__))
SCIEZKA_MODELU_DETEKCJI_GRACZA = os.path.join(GLOWNY_KATALOG, 'data/football-player-detection.pt')
SCIEZKA_MODELU_DETEKCJI_BOISKA = os.path.join(GLOWNY_KATALOG, 'data/football-pitch-detection.pt')
SCIEZKA_MODELU_DETEKCJI_PILKI = os.path.join(GLOWNY_KATALOG, 'data/football-ball-detection.pt')

KLASA_PILKI = 0
KLASA_BRAMKARZA = 1
KLASA_GRACZA = 2
KLASA_SEDZIEGO = 3

KROK = 60
KONFIGURACJA = KonfiguracjaBoiska()

KOLORY = ['#F5AE67', '#9306C7', '#FA2407', '#FFD700']
OZNACZNIK_ETYKIET_WIERZCHOŁKÓW = sv.VertexLabelAnnotator(
    color=[sv.Color.from_hex(color) for color in KONFIGURACJA.kolory],
    text_color=sv.Color.from_hex('#FFFFFF'),
    border_radius=5,
    text_thickness=1,
    text_scale=0.5,
    text_padding=5,
)
OZNACZNIK_KRAWĘDZI = sv.EdgeAnnotator(
    color=sv.Color.from_hex('#F5AE67'),
    thickness=2,
    edges=KONFIGURACJA.krawedzie,
)
OZNACZNIK_TROJKATA = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#F5AE67'),
    base=20,
    height=15,
)
OZNACZNIK_RAMKI = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(KOLORY),
    thickness=2
)
OZNACZNIK_ELIPSY = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(KOLORY),
    thickness=2
)
OZNACZNIK_ETYKIET_RAMKI = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(KOLORY),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
)
OZNACZNIK_ETYKIET_ELIPSY = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(KOLORY),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)


class Tryb(Enum):
    DETEKCJA_BOISKA = 'DETEKCJA_BOISKA'
    DETEKCJA_GRACZA = 'DETEKCJA_GRACZA'
    DETEKCJA_PILKI = 'DETEKCJA_PILKI'
    SLEDZENIE_GRACZA = 'SLEDZENIE_GRACZA'
    KLASYFIKACJA_DRUZYNY = 'KLASYFIKACJA_DRUZYNY'
    RADAR = 'RADAR'


def uzyskaj_crops(ramka: np.ndarray, detekcje: sv.Detections) -> List[np.ndarray]:
    return [sv.crop_image(ramka, xyxy) for xyxy in detekcje.xyxy]


def okresl_id_druzyny_bramkarzy(
    gracze: sv.Detections,
    id_druzyny_graczy: np.array,
    bramkarze: sv.Detections
) -> np.ndarray:
    bramkarze_xy = bramkarze.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    gracze_xy = gracze.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    centroid_druzyny_0 = gracze_xy[id_druzyny_graczy == 0].mean(axis=0)
    centroid_druzyny_1 = gracze_xy[id_druzyny_graczy == 1].mean(axis=0)
    id_druzyny_bramkarzy = []
    for bramkarz_xy in bramkarze_xy:
        dystans_0 = np.linalg.norm(bramkarz_xy - centroid_druzyny_0)
        dystans_1 = np.linalg.norm(bramkarz_xy - centroid_druzyny_1)
        id_druzyny_bramkarzy.append(0 if dystans_0 < dystans_1 else 1)
    return np.array(id_druzyny_bramkarzy)


def renderuj_radar(
    detekcje: sv.Detections,
    punkty_kluczowe: sv.KeyPoints,
    lookup_kolor: np.ndarray
) -> np.ndarray:
    maska = (punkty_kluczowe.xy[0][:, 0] > 1) & (punkty_kluczowe.xy[0][:, 1] > 1)
    transformator = TransformatorWidoku(
        zrodlo=punkty_kluczowe.xy[0][maska].astype(np.float32),
        cel=np.array(KONFIGURACJA.wierzcholki)[maska].astype(np.float32)
    )
    xy = detekcje.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    przeksztalcone_xy = transformator.transformuj_punkty(points=xy)

    radar = rysuj_boisko(config=KONFIGURACJA)
    radar = rysuj_punkty_na_boisku(
        config=KONFIGURACJA, xy=przeksztalcone_xy[lookup_kolor == 0],
        face_color=sv.Color.from_hex(KOLORY[0]), radius=20, pitch=radar)
    radar = rysuj_punkty_na_boisku(
        config=KONFIGURACJA, xy=przeksztalcone_xy[lookup_kolor == 1],
        face_color=sv.Color.from_hex(KOLORY[1]), radius=20, pitch=radar)
    radar = rysuj_punkty_na_boisku(
        config=KONFIGURACJA, xy=przeksztalcone_xy[lookup_kolor == 2],
        face_color=sv.Color.from_hex(KOLORY[2]), radius=20, pitch=radar)
    radar = rysuj_punkty_na_boisku(
        config=KONFIGURACJA, xy=przeksztalcone_xy[lookup_kolor == 3],
        face_color=sv.Color.from_hex(KOLORY[3]), radius=20, pitch=radar)
    return radar


def uruchom_detekcje_boiska(sciezka_wideo: str, urzadzenie: str) -> Iterator[np.ndarray]:
    model_detekcji_boiska = YOLO(SCIEZKA_MODELU_DETEKCJI_BOISKA).to(urzadzenie=urzadzenie)
    generator_ramki = sv.get_video_frames_generator(source_path=sciezka_wideo)
    for ramka in generator_ramki:
        wynik = model_detekcji_boiska(ramka, verbose=False)[0]
        punkty_kluczowe = sv.KeyPoints.from_ultralytics(wynik)

        ramka_oznaczona = ramka.copy()
        ramka_oznaczona = OZNACZNIK_ETYKIET_WIERZCHOŁKÓW.annotate(
            ramka_oznaczona, punkty_kluczowe, KONFIGURACJA.etykiety)
        yield ramka_oznaczona


def uruchom_detekcje_gracza(sciezka_wideo: str, urzadzenie: str) -> Iterator[np.ndarray]:
    model_detekcji_gracza = YOLO(SCIEZKA_MODELU_DETEKCJI_GRACZA).to(urzadzenie=urzadzenie)
    generator_ramki = sv.get_video_frames_generator(source_path=sciezka_wideo)
    for ramka in generator_ramki:
        wynik = model_detekcji_gracza(ramka, imgsz=1280, verbose=False)[0]
        detekcje = sv.Detections.from_ultralytics(wynik)

        ramka_oznaczona = ramka.copy()
        ramka_oznaczona = OZNACZNIK_RAMKI.annotate(ramka_oznaczona, detekcje)
        ramka_oznaczona = OZNACZNIK_ETYKIET_RAMKI.annotate(ramka_oznaczona, detekcje)
        yield ramka_oznaczona

def uruchom_detekcje_pilki(sciezka_wideo: str, urzadzenie: str) -> Iterator[np.ndarray]:
    model_detekcji_pilki = YOLO(SCIEZKA_MODELU_DETEKCJI_PILKI).to(urzadzenie=urzadzenie)
    generator_ramki = sv.get_video_frames_generator(source_path=sciezka_wideo)
    sledzenie_pilki = SledzeniePilki(buffer_size=20)
    oznacznik_pilki = OznacznikPilki(radius=6, buffer_size=10)

    def funkcja_pomocnicza(fragment_obrazu: np.ndarray) -> sv.Detections:
        wynik = model_detekcji_pilki(fragment_obrazu, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(wynik)

    slicer = sv.InferenceSlicer(
        callback=funkcja_pomocnicza,
        overlap_filter_strategy=sv.OverlapFilter.NONE,
        slice_wh=(640, 640),
    )

    for ramka in generator_ramki:
        detekcje = slicer(ramka).with_nms(threshold=0.1)
        detekcje = sledzenie_pilki.aktualizuj(detekcje)
        ramka_oznaczona = ramka.copy()
        ramka_oznaczona = oznacznik_pilki.oznacz(ramka_oznaczona, detekcje)
        yield ramka_oznaczona

def uruchom_sledzenie_graczy(sciezka_wideo: str, urzadzenie: str) -> Iterator[np.ndarray]:
    model_detekcji_gracza = YOLO(SCIEZKA_MODELU_DETEKCJI_GRACZA).to(urzadzenie=urzadzenie)
    generator_ramki = sv.get_video_frames_generator(source_path=sciezka_wideo)
    sledzenie = sv.ByteTrack(minimum_consecutive_frames=3)
    for ramka in generator_ramki:
        wynik = model_detekcji_gracza(ramka, imgsz=1280, verbose=False)[0]
        detekcje = sv.Detections.from_ultralytics(wynik)
        detekcje = sledzenie.update_with_detections(detekcje)

        etykiety = [str(tracker_id) for tracker_id in detekcje.tracker_id]

        ramka_oznaczona = ramka.copy()
        ramka_oznaczona = OZNACZNIK_ELIPSY.annotate(ramka_oznaczona, detekcje)
        ramka_oznaczona = OZNACZNIK_ETYKIET_ELIPSY.annotate(
            ramka_oznaczona, detekcje, labels=etykiety)
        yield ramka_oznaczona


def uruchom_klasyfikacje_druzyn(sciezka_wideo: str, urzadzenie: str) -> Iterator[np.ndarray]:
    model_detekcji_gracza = YOLO(SCIEZKA_MODELU_DETEKCJI_GRACZA).to(urzadzenie=urzadzenie)
    generator_ramki = sv.get_video_frames_generator(
        source_path=sciezka_wideo, stride=KROK)

    crops = []
    for ramka in tqdm(generator_ramki, desc='zbieranie wyciętych piłkarzy'):
        wynik = model_detekcji_gracza(ramka, imgsz=1280, verbose=False)[0]
        detekcje = sv.Detections.from_ultralytics(wynik)
        crops += uzyskaj_crops(ramka, detekcje[detekcje.class_id == KLASA_GRACZA])

    klasyfikator_druzyn = KlasyfikatorDruzyn(urzadzenie=urzadzenie)
    klasyfikator_druzyn.dopasuj(crops)

    generator_ramki = sv.get_video_frames_generator(source_path=sciezka_wideo)
    sledzenie = sv.ByteTrack(minimum_consecutive_frames=3)
    for ramka in generator_ramki:
        wynik = model_detekcji_gracza(ramka, imgsz=1280, verbose=False)[0]
        detekcje = sv.Detections.from_ultralytics(wynik)
        detekcje = sledzenie.update_with_detections(detekcje)

        gracze = detekcje[detekcje.class_id == KLASA_GRACZA]
        crops = uzyskaj_crops(ramka, gracze)
        id_druzyny_graczy = klasyfikator_druzyn.przewiduj(crops)

        bramkarze = detekcje[detekcje.class_id == KLASA_BRAMKARZA]
        id_druzyny_bramkarzy = okresl_id_druzyny_bramkarzy(
            gracze, id_druzyny_graczy, bramkarze)

        sedziowie = detekcje[detekcje.class_id == KLASA_SEDZIEGO]

        detekcje = sv.Detections.merge([gracze, bramkarze, sedziowie])
        lookup_kolor = np.array(
                id_druzyny_graczy.tolist() +
                id_druzyny_bramkarzy.tolist() +
                [KLASA_SEDZIEGO] * len(sedziowie)
        )
        etykiety = [str(tracker_id) for tracker_id in detekcje.tracker_id]

        ramka_oznaczona = ramka.copy()
        ramka_oznaczona = OZNACZNIK_ELIPSY.annotate(
            ramka_oznaczona, detekcje, custom_color_lookup=lookup_kolor)
        ramka_oznaczona = OZNACZNIK_ETYKIET_ELIPSY.annotate(
            ramka_oznaczona, detekcje, etykiety, custom_color_lookup=lookup_kolor)
        yield ramka_oznaczona


def uruchom_radar(sciezka_wideo: str, urzadzenie: str) -> Iterator[np.ndarray]:
    model_detekcji_gracza = YOLO(SCIEZKA_MODELU_DETEKCJI_GRACZA).to(urzadzenie=urzadzenie)
    model_detekcji_boiska = YOLO(SCIEZKA_MODELU_DETEKCJI_BOISKA).to(urzadzenie=urzadzenie)
    generator_ramki = sv.get_video_frames_generator(
        source_path=sciezka_wideo, stride=KROK)

    crops = []
    for ramka in tqdm(generator_ramki, desc='zbieranie wyciętych piłkarzy'):
        wynik = model_detekcji_gracza(ramka, imgsz=1280, verbose=False)[0]
        detekcje = sv.Detections.from_ultralytics(wynik)
        crops += uzyskaj_crops(ramka, detekcje[detekcje.class_id == KLASA_GRACZA])

    klasyfikator_druzyn = KlasyfikatorDruzyn(urzadzenie=urzadzenie)
    klasyfikator_druzyn.dopasuj(crops)

    generator_ramki = sv.get_video_frames_generator(source_path=sciezka_wideo)
    sledzenie = sv.ByteTrack(minimum_consecutive_frames=3)
    for ramka in generator_ramki:
        wynik = model_detekcji_boiska(ramka, verbose=False)[0]
        punkty_kluczowe = sv.KeyPoints.from_ultralytics(wynik)
        wynik = model_detekcji_gracza(ramka, imgsz=1280, verbose=False)[0]
        detekcje = sv.Detections.from_ultralytics(wynik)
        detekcje = sledzenie.update_with_detections(detekcje)

        gracze = detekcje[detekcje.class_id == KLASA_GRACZA]
        crops = uzyskaj_crops(ramka, gracze)
        id_druzyny_graczy = klasyfikator_druzyn.przewiduj(crops)

        bramkarze = detekcje[detekcje.class_id == KLASA_BRAMKARZA]
        id_druzyny_bramkarzy = okresl_id_druzyny_bramkarzy(
            gracze, id_druzyny_graczy, bramkarze)

        sedziowie = detekcje[detekcje.class_id == KLASA_SEDZIEGO]

        detekcje = sv.Detections.merge([gracze, bramkarze, sedziowie])
        lookup_kolor = np.array(
            id_druzyny_graczy.tolist() +
            id_druzyny_bramkarzy.tolist() +
            [KLASA_SEDZIEGO] * len(sedziowie)
        )
        etykiety = [str(tracker_id) for tracker_id in detekcje.tracker_id]

        ramka_oznaczona = ramka.copy()
        ramka_oznaczona = OZNACZNIK_ELIPSY.annotate(
            ramka_oznaczona, detekcje, custom_color_lookup=lookup_kolor)
        ramka_oznaczona = OZNACZNIK_ETYKIET_ELIPSY.annotate(
            ramka_oznaczona, detekcje, etykiety,
            custom_color_lookup=lookup_kolor)

        h, w, _ = ramka.shape
        radar = renderuj_radar(detekcje, punkty_kluczowe, lookup_kolor)
        radar = sv.resize_image(radar, (w // 2, h // 2))
        radar_h, radar_w, _ = radar.shape
        prostokat = sv.Rect(
            x=w // 2 - radar_w // 2,
            y=h - radar_h,
            width=radar_w,
            height=radar_h
        )
        ramka_oznaczona = sv.draw_image(ramka_oznaczona, radar, opacity=0.5, rect=prostokat)
        yield ramka_oznaczona


def main(sciezka_wideo_zrodlowa: str, sciezka_wideo_docelowa: str, urzadzenie: str, tryb: Tryb) -> None:
    if tryb == Tryb.DETEKCJA_BOISKA:
        generator_ramki = uruchom_detekcje_boiska(
            sciezka_wideo_zrodlowa=sciezka_wideo_zrodlowa, urzadzenie=urzadzenie)
    elif tryb == Tryb.DETEKCJA_GRACZA:
        generator_ramki = uruchom_detekcje_gracza(
            sciezka_wideo_zrodlowa=sciezka_wideo_zrodlowa, urzadzenie=urzadzenie)
    elif tryb == Tryb.DETEKCJA_PILKI:
        generator_ramki = uruchom_detekcje_pilki(
            sciezka_wideo_zrodlowa=sciezka_wideo_zrodlowa, urzadzenie=urzadzenie)
    elif tryb == Tryb.SLEDZENIE_GRACZA:
        generator_ramki = uruchom_sledzenie_graczy(
            sciezka_wideo_zrodlowa=sciezka_wideo_zrodlowa, urzadzenie=urzadzenie)
    elif tryb == Tryb.KLASYFIKACJA_DRUZYNY:
        generator_ramki = uruchom_klasyfikacje_druzyn(
            sciezka_wideo_zrodlowa=sciezka_wideo_zrodlowa, urzadzenie=urzadzenie)
    elif tryb == Tryb.RADAR:
        generator_ramki = uruchom_radar(
            sciezka_wideo_zrodlowa=sciezka_wideo_zrodlowa, urzadzenie=urzadzenie)
    else:
        raise NotImplementedError(f"Tryb {tryb} nie jest zaimplementowany.")

    informacje_o_wideo = sv.VideoInfo.from_video_path(sciezka_wideo_zrodlowa)
    with sv.VideoSink(sciezka_wideo_docelowa, informacje_o_wideo) as sink:
        for ramka in generator_ramki:
            sink.write_frame(ramka)

            cv2.imshow("ramka", ramka)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--sciezka_wideo_zrodlowa', type=str, required=True)
    parser.add_argument('--sciezka_wideo_docelowa', type=str, required=True)
    parser.add_argument('--urzadzenie', type=str, default='cpu')
    parser.add_argument('--tryb', type=Tryb, default=Tryb.DETEKCJA_GRACZA)
    args = parser.parse_args()
    main(
        sciezka_wideo_zrodlowa=args.sciezka_wideo_zrodlowa,
        sciezka_wideo_docelowa=args.sciezka_wideo_docelowa,
        urzadzenie=args.urzadzenie,
        tryb=args.tryb
    )