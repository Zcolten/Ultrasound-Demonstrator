"""Demonstrador de ecografia B-mode com cálculo acústico 2D e geometria 3D.

Os cortes XY e XZ são propagados separadamente. O modo B usa a sonda
do corte XY, com composição de várias emissões inclinadas.
"""

import os
import sys
import traceback
from copy import copy
from dataclasses import dataclass
from subprocess import run, TimeoutExpired
from time import monotonic

import h5py
from tempfile import TemporaryDirectory, gettempdir

import numpy as np
from PySide6.QtCore import QObject, QRectF, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QVector3D
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QSplitter,
    QMenu,
)
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.signal import butter, hilbert, sosfiltfilt
from skimage.measure import find_contours, marching_cubes

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.checks import check_stability
from kwave.utils.pml import get_optimal_pml_size


if os.environ.get("USERNAME"):
    os.environ["USERNAME"] = (
        os.environ["USERNAME"].encode("ascii", "ignore").decode("ascii")
    )

N = 64
DX = 0.1e-3
CENTRO = (N // 2, N // 2, N // 2)
# A origem é o vértice da sonda no primeiro vóxel da pele.
X_PELE = 12
FONTE = (X_PELE, CENTRO[1], CENTRO[2])

C_PELE = 1540.0
RHO_PELE = 1050.0
ALPHA_PELE = 0.5
ALPHA_POWER = 1.1

MATERIAIS = {
    "Testículo (modelo didático)": {"speed": 1560, "density": 1055, "absorption": 0.50, "dispersao": 0.7},
    "Cisto (água)": {"speed": 1480, "density": 998, "absorption": 0.0022, "dispersao": 0.0},
    "Gordura": {"speed": 1450, "density": 920, "absorption": 0.60, "dispersao": 1.5},
    "Músculo": {"speed": 1580, "density": 1050, "absorption": 1.09, "dispersao": 1.0},
    "Tecido mole": {"speed": 1540, "density": 1000, "absorption": 0.50, "dispersao": 1.0},
    "Osso cortical": {"speed": 3500, "density": 1900, "absorption": 20.0, "dispersao": 0.0},
    "Sangue": {"speed": 1570, "density": 1060, "absorption": 0.18, "dispersao": 0.0},
    "Aço": {"speed": 5900, "density": 7850, "absorption": 0.10, "dispersao": 0.0},
}


def mascara_elipsoide(dx, dy, dz, a, b, c):
    return (dx / a)**2 + (dy / b)**2 + (dz / c)**2 <= 1


def mascara_cilindro(dx, dy, dz, a, b, c):
    dentro_da_base = (dx / a)**2 + (dy / b)**2 <= 1
    dentro_da_altura = np.abs(dz) <= c
    return dentro_da_base & dentro_da_altura


def mascara_caixa(dx, dy, dz, a, b, c):
    return (np.abs(dx) <= a) & (np.abs(dy) <= b) & (np.abs(dz) <= c)


def mascara_toro(dx, dy, dz, a, b, c):
    distancia_ao_eixo = np.sqrt(dx**2 + dy**2)
    return ((distancia_ao_eixo - a) / b)**2 + (dz / c)**2 <= 1


FORMATOS = {
    "Elipsoide": mascara_elipsoide,
    "Cilindro": mascara_cilindro,
    "Caixa": mascara_caixa,
    "Toro": mascara_toro,
}


def avaliar_formato(formato, dx, dy, dz, a, b, c, rotacao_y, alongamento_x, alongamento_y):
    angulo = np.deg2rad(rotacao_y)
    cos_a, sin_a = np.cos(angulo), np.sin(angulo)
    x = cos_a * dx - sin_a * dz
    z = sin_a * dx + cos_a * dz
    y = dy
    if formato == "Toro":
        x = x / alongamento_x
        y = y / alongamento_y
    return FORMATOS[formato](x, y, z, a, b, c)


def criar_mascara(formato, a, b, c, rotacao_y, alongamento_x, alongamento_y, n, centro=None):
    x, y, z = np.ogrid[:n, :n, :n]
    cx, cy, cz = (n // 2,) * 3 if centro is None else centro
    return avaliar_formato(formato, x - cx, y - cy, z - cz, a, b, c,
                           rotacao_y, alongamento_x, alongamento_y)


def especificacao_segundo_objeto(dados):
    raio = dados["raio"]
    formato = dados["formato"]
    a, b, c = (raio * dados[chave] for chave in ("a", "b", "c"))
    eixos = (.7 * a, .3 * b, .3 * c) if formato == "Toro" else (a, b, c)
    return formato, eixos


def mascara_segundo_objeto(n, dados):
    formato, eixos = especificacao_segundo_objeto(dados)
    return criar_mascara(formato, *eixos, 0., 1., 1., n,
                         centro=(dados["x"], n // 2, n // 2))


C_GEL = 1500.0
RHO_GEL = 1000.0
ALPHA_GEL = 0.05


def perfil_contacto(n, x_pele, raio, abertura, altura, gel):
    """Geometria comum à vista estática e aos cortes acústicos."""
    centro = n // 2
    if not 1 <= abertura < min(centro, n - centro):
        raise ValueError("A meia-abertura tem de caber na grelha.")
    if raio and raio <= abertura:
        raise ValueError("Numa sonda curva, o raio tem de exceder a meia-abertura.")
    if not 0 <= altura < min(centro, n - centro):
        raise ValueError("A meia-altura da sonda tem de caber na grelha.")
    y = np.clip(np.arange(n) - centro, -abertura, abertura)
    sag = np.rint(raio - np.sqrt(raio**2 - y**2)).astype(int) if raio else 0
    limite = np.broadcast_to(x_pele - sag, (n,))
    if gel < 1 or limite.min() - gel < 1 or x_pele >= n - 3:
        raise ValueError("Deixa espaço para a face e o gel antes da pele, e tecido à frente.")
    yy = np.arange(centro - abertura, centro + abertura + 1)
    zz = np.arange(centro - altura, centro + altura + 1)
    return limite, yy, zz


@dataclass(frozen=True)
class Acoplamento:
    origem: tuple[int, int, int]
    fonte: np.ndarray
    limite_pele: np.ndarray
    gel: np.ndarray


def geometria_acoplamento(n, x_pele, raio_sonda, abertura_sonda, altura_sonda, espessura_gel=3):
    limite, yy, zz = perfil_contacto(
        n, x_pele, raio_sonda, abertura_sonda, altura_sonda, espessura_gel,
    )
    xx = limite[yy] - espessura_gel
    fonte = np.zeros((n, n, n), dtype=bool)
    fonte[xx[:, None], yy[:, None], zz[None, :]] = True
    gel = np.zeros_like(fonte)
    x = np.arange(n)[:, None, None]
    gel[:, yy[0]:yy[-1] + 1, zz[0]:zz[-1] + 1] = (
        (x >= xx[None, :, None]) & (x < limite[yy][None, :, None])
    )
    return Acoplamento((x_pele - espessura_gel, n // 2, n // 2), fonte, limite, gel)



@dataclass(frozen=True)
class Ecos:
    lateral: np.ndarray
    eixo_profundidade: np.ndarray
    imagem_db: np.ndarray
    mascara_setor: np.ndarray
    numero_disparos: int
    solicitados: int
    frequencia: float
    passo_acustico: float
    duracao: float
    piso_db: float = -55.0


@dataclass(frozen=True)
class Simulacao:
    tempos: np.ndarray
    cortes: tuple[np.ndarray, np.ndarray]
    limite: float
    tempos_sondas: np.ndarray
    sondas: np.ndarray
    posicoes_sondas: tuple[tuple[int, int, int], ...]
    ecos: Ecos
    dx: float


@dataclass(frozen=True)
class Corte:
    x: np.ndarray
    y: np.ndarray
    c: np.ndarray
    rho: np.ndarray
    alpha: np.ndarray
    fonte: tuple[np.ndarray, np.ndarray]
    passo: float
    pele: float
    gel: float


def _ruido(forma, semente):
    valores = np.random.default_rng(semente).standard_normal(forma).astype(np.float32)
    gaussian_filter(valores, .65, output=valores)
    return np.clip(valores / max(float(valores.std()), 1e-8), -3, 3)


def _corte_acustico(
    n, dx, passo, eixo, x_pele, espessura_gel, raio_sonda,
    abertura, altura, fundo, gel, objetos, dispersao_pele, semente, clinico,
):
    """Avalia diretamente a geometria 3D num corte, na malha acústica."""
    centro = n // 2
    limite, _, _ = perfil_contacto(n, x_pele, raio_sonda, abertura, altura, espessura_gel)
    x_min = max(0., (float(limite.min()) - espessura_gel) * dx - 5 * passo)
    x_max = (n - 1) * dx
    if clinico:
        posterior = max(
            (pos + np.hypot(a, c)) * dx if ang else (pos + a) * dx
            for _, (a, _, c), ang, _, _, pos, _, _ in objetos
        )
        x_max = min(x_max, max(x_pele * dx + .030, posterior + .004))

    x = x_min + passo * np.arange(int(np.ceil((x_max - x_min) / passo)) + 1)
    y = passo * np.arange(int(np.ceil((n - 1) * dx / passo)) + 1)

    u = x[:, None] / dx
    v = y[None, :] / dx - centro
    lateral = np.clip(v, -abertura, abertura) if eixo == 1 else np.zeros_like(v)
    sag = raio_sonda - np.sqrt(raio_sonda**2 - lateral**2) if raio_sonda else 0.
    fronteira = x_pele - .5 - sag
    tecido = u >= fronteira
    chao = [np.where(tecido, f, g).astype(np.float32) for f, g in zip(fundo, gel)]
    chao = [np.broadcast_to(p, (len(x), len(y))).copy() for p in chao]
    c, rho, alpha = chao
    if clinico:
        parede = tecido & ((u - fronteira) * dx < .001)
        parede = np.broadcast_to(parede, c.shape)
        c[parede], rho[parede], alpha[parede] = 1580, 1080, .65

    if dispersao_pele:
        ruido = _ruido(c.shape, np.random.SeedSequence([semente, eixo, 0]))
        ruido *= np.broadcast_to(tecido, c.shape) * dispersao_pele / 100
        rho *= 1 + ruido
        c *= 1 + .25 * ruido

    for i, (formato, eixos, ang, ax, ay, pos, propriedades, dispersao) in enumerate(objetos, 1):
        yy, zz = (v, 0.) if eixo == 1 else (0., v)
        mascara = avaliar_formato(formato, u - pos, yy, zz, *eixos, ang, ax, ay)
        if mascara.shape != c.shape:
            mascara = np.broadcast_to(mascara, c.shape)
        if np.any(mascara & ~tecido):
            raise ValueError(f"O objeto {i} invade o gel. Avança a posição X ou reduz o raio.")
        for mapa, valor in zip(chao, propriedades):
            mapa[mascara] = valor
        if dispersao and mascara.any():
            ruido = _ruido(c.shape, np.random.SeedSequence([semente, eixo, i]))
            rho[mascara] *= 1 + (dispersao / 100) * ruido[mascara]
            c[mascara] *= 1 + (.30 * dispersao / 100) * ruido[mascara]
        if clinico and i == 1 and formato == "Elipsoide":
            a, b, h = (s * dx for s in eixos)
            # Estrutura fina posterior, com maior impedância no fantoma.
            profundidade = x[:, None] - pos * dx
            lateral_m = y[None, :] - centro * dx
            extensao = b if eixo == 1 else h
            faixa = ((profundidade - .55 * a) / .00038)**2 + (
                lateral_m / (.43 * extensao)
            )**2 <= 1
            faixa &= mascara
            c[faixa], rho[faixa] = 1590, 1080

    meio_lateral = abertura if eixo == 1 else altura
    iy = np.flatnonzero(np.abs(v.ravel()) <= meio_lateral)
    if not iy.size:
        iy = np.array([int(np.argmin(np.abs(v.ravel())))])
    face = (fronteira - espessura_gel + .5) * dx
    fx = np.broadcast_to(face, (1, len(y))).ravel()[iy]
    ix = np.clip(np.rint((fx - x_min) / passo).astype(int), 0, len(x) - 1)
    return Corte(x, y, c, rho, alpha, (ix, iy), passo, (x_pele - .5) * dx, espessura_gel * dx)


def _angulos(numero, proximo_limite=12.):
    """Ordem progressiva que mantém os pares de ângulos balanceados."""
    valores = np.linspace(-proximo_limite, proximo_limite, numero)
    grupos = [[0.]] if numero % 2 else []
    positivos = list(valores[valores > 1e-9])
    usados = [0.]
    while positivos:
        valor = max(positivos, key=lambda v: min(abs(v - u) for u in usados))
        positivos.remove(valor)
        usados.append(valor)
        grupos.append([-valor, valor])
    return grupos


def _rasters(corte, p, indices, quadros, amostras, n, dx):
    ix, iy = amostras
    mapas = p[quadros[:, None, None], indices[ix[:, None], iy[None, :]]]
    px = np.arange(n) * dx
    xi = np.interp(px, corte.x[ix], np.arange(len(ix)), left=-1, right=len(ix))
    yi = np.interp(px, corte.y[iy], np.arange(len(iy)), left=0, right=len(iy) - 1)
    grade = np.stack(np.meshgrid(xi, yi, indexing="ij"))
    return np.stack([
        map_coordinates(mapa, grade, order=1, mode="constant", cval=0., prefilter=False)
        for mapa in mapas
    ])


def _projetar(corte, recus, rf, dt, frequencia, q_gel, q_fundo, p,
              atraso_origem, centro_pulso, lateral, profundidades, c_fundo, c_gel):
    """Soma coerente em receção com abertura dependente da profundidade."""
    x_rec, y_rec = recus
    grade_y, grade_d = lateral[:, None], profundidades[None, :]
    y_entrada = grade_y - p * (grade_d / q_fundo + corte.gel / q_gel)
    folga = corte.passo
    iluminado = (y_entrada >= y_rec.min() - folga) & (y_entrada <= y_rec.max() + folga)
    ida = atraso_origem + centro_pulso + p * grade_y + q_gel * corte.gel + q_fundo * grade_d

    sos = butter(3, [.50 * frequencia, 1.50 * frequencia], fs=1 / dt,
                 btype="bandpass", output="sos")
    analitico = hilbert(sosfiltfilt(sos, rf, axis=0), axis=0).astype(np.complex64)
    x_imagem = corte.pele + profundidades
    meia = np.clip(profundidades / 3, 2 * corte.passo, .5 * np.ptp(y_rec) + corte.passo)
    soma = np.zeros(ida.shape, dtype=np.complex64)
    pesos = np.zeros(ida.shape, dtype=np.float32)
    for j in range(len(y_rec)):
        primeiro = np.searchsorted(lateral, y_rec[j] - meia.max())
        ultimo = np.searchsorted(lateral, y_rec[j] + meia.max(), side="right")
        faixa = slice(primeiro, ultimo)
        dy = lateral[faixa, None] - y_rec[j]
        frac = np.minimum(np.abs(dy) / meia, 1.)
        peso = np.where(frac < 1, .5 * (1 + np.cos(np.pi * frac)), 0.)
        # Percurso de retorno reto, corrigido pelo tempo no gel.
        ax = x_imagem - x_rec[j]
        dist = np.hypot(ax, dy)
        comprimento_gel = dist * np.clip((corte.pele - x_rec[j]) / ax, 0, 1)
        volta = dist / c_fundo + comprimento_gel * (1 / c_gel - 1 / c_fundo)
        amostra = (ida[faixa] + volta) / dt
        recebido = np.interp(amostra.ravel(), np.arange(len(rf)), analitico[:, j],
                              left=0., right=0.).reshape(amostra.shape)
        soma[faixa] += recebido * peso
        pesos[faixa] += peso

    valido = iluminado & (pesos > 1e-7)
    envelope = np.divide(abs(soma), pesos, out=np.zeros_like(pesos), where=valido)
    return envelope**2, valido


def simular_kwave(
    raio, c, rho, material, frequencia, amplitude, alpha,
    a_raio, b_raio, c_raio, formato, c_ambiente, rho_ambiente,
    alpha_ambiente, angulo_y, alongamento_x, alongamento_y, N, objeto_2=None,
    *, dx=DX, x_pele=X_PELE, raio_sonda=12, abertura_sonda=10, altura_sonda=2,
    dispersao_pele=0.0, semente_pele=7, x_objeto=None, notificar=print,
    espessura_gel=3, c_gel=C_GEL, rho_gel=RHO_GEL, alpha_gel=ALPHA_GEL,
    dispersao_objeto=None, numero_disparos=9,
):
    inicio = monotonic()
    prazo = inicio + 690.
    if not isinstance(numero_disparos, (int, np.integer)) or not 2 <= numero_disparos <= 21:
        raise ValueError("Seleciona entre 2 e 21 disparos.")
    fundo = (c_ambiente, rho_ambiente, alpha_ambiente)
    gel = (c_gel, rho_gel, alpha_gel)
    disp = MATERIAIS[material]["dispersao"] if dispersao_objeto is None else dispersao_objeto
    objetos = [(
        formato, (a_raio, b_raio, c_raio), angulo_y, alongamento_x, alongamento_y,
        N // 2 if x_objeto is None else x_objeto, (c, rho, alpha), disp,
    )]
    if objeto_2 is not None:
        mat = MATERIAIS[objeto_2["material"]]
        forma_2, eixos_2 = especificacao_segundo_objeto(objeto_2)
        disp_2 = objeto_2.get("dispersao")
        objetos.append((
            forma_2, eixos_2, 0., 1., 1., objeto_2["x"],
            (objeto_2.get("velocidade", mat["speed"]), objeto_2.get("densidade", mat["density"]),
             objeto_2.get("atenuacao", mat["absorption"])),
            mat["dispersao"] if disp_2 is None else disp_2,
        ))
    meios = [fundo, gel, *(o[6] for o in objetos)]
    valores = [dx, frequencia, amplitude, *np.ravel(meios)]
    if not np.isfinite(valores).all() or min(valores[:3]) <= 0 or any(
        min(v, r) <= 0 or a < 0 for v, r, a in meios
    ):
        raise ValueError("Revê os valores acústicos e o passo espacial.")
    if not all(0 <= d <= 5 for d in (dispersao_pele, *(o[7] for o in objetos))):
        raise ValueError("A dispersão deve estar entre 0% e 5%.")

    c_min = min(v for v, _, _ in meios)
    c_max = max(v for v, _, _ in meios)
    passo = min(dx, c_min / (4.2 * frequencia))
    clinico = material.startswith("Testículo") and max(a_raio, b_raio, c_raio) * dx >= .008
    duracao_pulso = 2 / frequencia
    pasta_temp = gettempdir()
    if os.name == "nt" and not pasta_temp.isascii():
        pasta_temp = os.path.join(os.environ.get("SystemDrive", "C:"), "\\Temp")
        os.makedirs(pasta_temp, exist_ok=True)
    notificar("A preparar os cortes acústicos XY e XZ…")
    cortes = tuple(_corte_acustico(
        N, dx, passo, eixo, x_pele, espessura_gel, raio_sonda,
        abertura_sonda, altura_sonda, fundo, gel, objetos, dispersao_pele, semente_pele, clinico,
    ) for eixo in (1, 2))
    t_final = max(
        (2.12 * (p.x[-1] - p.x[0]) + .23 * np.ptp(p.y)) / c_min
        for p in cortes
    ) + 3 * duracao_pulso
    dt = .24 * passo / (c_max * 1.06)
    meios_kwave = tuple(kWaveMedium(
        sound_speed=p.c, density=p.rho, alpha_coeff=p.alpha, alpha_power=ALPHA_POWER,
        sound_speed_ref=max(c_max, float(p.c.max())),
    ) for p in cortes)
    for p, meio in zip(cortes, meios_kwave):
        try:
            dt = min(dt, .94 * check_stability(kWaveGrid(p.c.shape, (passo, passo)), meio))
        except RuntimeError as exc:
            raise ValueError("Não se conseguiu estabelecer um passo temporal estável neste meio.") from exc
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Passo temporal inválido. Revê os materiais.")
    nt = int(np.ceil(t_final / dt)) + 1
    tempos = np.arange(nt) * dt
    quadros = np.unique(np.linspace(0, nt - 1, min(nt, 100)).astype(int))
    pml = tuple(get_optimal_pml_size(p.c.shape, [10, 23]) for p in cortes)
    n_threads = 1 if cortes[0].c.size < 35_000 else min(2, os.cpu_count() or 1)
    execucao = SimulationExecutionOptions(
        is_gpu_simulation=False, num_threads=n_threads, show_sim_log=False,
    )

    def adquirir(indice, angulo, guardar_corte=False):
        corte, meio = cortes[indice], meios_kwave[indice]
        ix, iy = corte.fonte
        p = np.sin(np.deg2rad(angulo)) / c_ambiente
        q_gel = np.sqrt(1. / c_gel**2 - p**2)
        q_fundo = np.sqrt(1. / c_ambiente**2 - p**2)
        local_y = corte.y[iy] - (N // 2) * dx
        face_x = corte.x[ix]
        fase = local_y * p + (face_x - corte.pele + corte.gel) * q_gel
        atraso = fase - fase.min()
        ts = np.arange(int(np.ceil((duracao_pulso + atraso.max()) / dt)) + 1) * dt
        t = ts[None, :] - atraso[:, None]
        onda = np.sin(2 * np.pi * frequencia * t) * np.sin(np.pi * t / duracao_pulso)**2
        janela = np.hanning(len(iy) + 2)[1:-1]**.35
        onda *= janela[:, None] * amplitude * ((t >= 0) & (t < duracao_pulso))
        fonte = kSource()
        fonte.p_mask = np.zeros(corte.c.shape, dtype=bool)
        fonte.p_mask[ix, iy] = True
        fonte.p_mode = "additive"
        # Ordem nativa dos pontos da máscara, X varia primeiro.
        fonte.p = np.ascontiguousarray(onda[np.argsort(ix + len(corte.x) * iy)])
        sensor = fonte.p_mask.copy()
        amostras = None
        pos = None
        if guardar_corte:
            tam = max(16, min(62, int(np.sqrt(48 * 1024**2 / (4 * nt)))))
            amostras = tuple(np.unique(np.linspace(0, k - 1, min(k, tam)).astype(int))
                             for k in corte.c.shape)
            sensor[np.ix_(*amostras)] = True
            pos_x = np.clip(np.rint((np.array([corte.pele + .001, .5 * (corte.pele + corte.x[-1]),
                                             corte.x[-1] - passo]) - corte.x[0]) / passo).astype(int),
                            0, len(corte.x) - 1)
            pos_y = int(np.argmin(abs(corte.y - (N // 2) * dx)))
            pos = (pos_x, np.full(3, pos_y))
            sensor[pos] = True

        ns = int(sensor.sum())
        grelha = kWaveGrid(corte.c.shape, (passo, passo))
        grelha.setTime(nt, dt)
        with TemporaryDirectory(prefix="mim-2d-", dir=pasta_temp) as pasta:
            sensores = kSensor(mask=sensor.copy(), record=["p"])
            opcoes = SimulationOptions(
                pml_inside=False, pml_size=pml[indice], data_cast="single",
                save_to_disk=True, save_to_disk_exit=True, data_path=pasta,
                input_filename="entrada.h5", output_filename="saida.h5",
            )
            kspaceFirstOrder2DC(
                kgrid=grelha, medium=copy(meio), source=fonte,
                sensor=sensores, simulation_options=opcoes, execution_options=execucao,
            )
            comando = [
                str(execucao.binary_path), "-i", opcoes.input_filename,
                "-o", opcoes.output_filename, *execucao.as_list(sensores),
            ]
            ambiente = os.environ.copy()
            ambiente["OMP_NUM_THREADS"] = str(n_threads)
            ambiente["OMP_THREAD_LIMIT"] = str(n_threads)
            termino = run(comando, env=ambiente, capture_output=True,
                          timeout=max(.05, prazo - monotonic()))
            if termino.returncode:
                erro = (termino.stderr or termino.stdout).decode(errors="replace").strip()
                raise ValueError(erro[-1500:] or "O executável k-Wave terminou com erro.")
            with h5py.File(opcoes.output_filename, "r") as saida:
                bruto = saida["p"][()].squeeze()
        bruto = np.asarray(bruto).reshape(nt, 1) if ns == 1 else np.asarray(bruto)
        if bruto.shape != (nt, ns) and bruto.shape == (ns, nt):
            bruto = bruto.T
        if bruto.shape != (nt, ns) or not np.isfinite(bruto).all():
            raise ValueError(f"Registo acústico inválido: {bruto.shape}.")
        indices = np.full(sensor.shape, -1, dtype=np.int32, order="F")
        indices.ravel(order="F")[sensor.ravel(order="F")] = np.arange(ns)
        rf = bruto[:, indices[ix, iy]]
        if guardar_corte:
            filme = _rasters(corte, bruto, indices, quadros, amostras, N, dx)
            sondas = bruto[:, indices[pos]].T.copy()
        else:
            filme, sondas = None, None
        return rf, (face_x, local_y), (q_gel, q_fundo, p, -float(fase.min())), filme, sondas

    notificar("Corte XZ · disparo axial…")
    try:
        _, _, _, filme_xz, _ = adquirir(1, 0., True)
    except TimeoutExpired as exc:
        raise ValueError("O corte XZ excedeu 180 segundos. Reduz o campo ou a frequência.") from exc
    corte = cortes[0]
    salto = max(corte.passo, .00012)
    lateral = np.arange(0., corte.y[-1] + salto / 2, salto) - (N // 2) * dx
    profundidades = np.arange(max(.5 * passo, .48 * c_ambiente / frequencia),
                              corte.x[-1] - corte.pele, salto)
    if len(profundidades) < 2:
        raise ValueError("Falta profundidade útil para formar a imagem.")
    potencia = np.zeros((len(lateral), len(profundidades)))
    contagem = np.zeros(potencia.shape, dtype=np.uint8)
    completados = 0
    cronometros = []
    filme_xy = sondas = None
    limite_angular = min(12., np.rad2deg(np.arcsin(min(1., .9 * c_ambiente / c_gel))))
    for grupo in _angulos(numero_disparos, limite_angular):
        if completados >= 2 and monotonic() + len(grupo) * np.median(cronometros) > prazo - 6.:
            break
        acumulados = []
        for angulo in grupo:
            instante = monotonic()
            numero = completados + len(acumulados) + 1
            notificar(f"Modo B · disparo {numero}/{numero_disparos} · {angulo:+.1f}°")
            try:
                rf, rec, parametros, filme, sinais = adquirir(0, angulo, numero == 1)
            except TimeoutExpired:
                if completados < 2:
                    raise ValueError(
                        "O tempo terminou antes de completar várias emissões. "
                        "Reduz o campo físico ou a frequência."
                    ) from None
                break
            if filme is not None:
                filme_xy, sondas = filme, sinais
            energia, valido = _projetar(
                corte, rec, rf, dt, frequencia, *parametros, .5 * duracao_pulso,
                lateral, profundidades, c_ambiente, c_gel,
            )
            acumulados.append((energia, valido))
            cronometros.append(monotonic() - instante)
        if len(acumulados) < len(grupo):
            break
        for energia, valido in acumulados:
            potencia[valido] += energia[valido]
            contagem[valido] += 1
        completados += len(acumulados)
    validos = (contagem > 0) & (potencia > 0)
    imagem_db = np.full(potencia.shape, -np.inf, dtype=np.float32)
    if validos.any():
        energia = potencia[validos] / contagem[validos]
        referencia = max(float(np.percentile(energia, 99.)), (amplitude * 2e-6)**2)
        imagem_db[validos] = 10 * np.log10(energia / referencia)
    ecos = Ecos(lateral, profundidades, imagem_db, validos, completados, numero_disparos,
                frequencia, passo, monotonic() - inicio)
    limite = max(1e-9, *(float(np.percentile(np.abs(mapa), 99.7))
                         for mapa in (filme_xy, filme_xz)))
    sondas_pos = tuple((i, N // 2, N // 2) for i in (x_pele, N // 2, N - 2))
    return Simulacao(tempos[quadros], (filme_xy, filme_xz), limite, tempos,
                      sondas, sondas_pos, ecos, dx)


class TrabalhadorSimulacao(QObject):
    resultado = Signal(object)
    erro = Signal(str)
    progresso = Signal(str)

    def __init__(self, parametros):
        super().__init__()
        self.parametros = parametros

    @Slot()
    def correr(self):
        try:
            resultado = simular_kwave(**self.parametros, notificar=self.progresso.emit)
        except ValueError as exc:
            self.erro.emit(str(exc))
        except Exception:
            self.erro.emit(traceback.format_exc())
        else:
            self.resultado.emit(resultado)


class Demonstrador(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ecografia em modo B · Ondas em 2D")
        self.resize(1250, 800)
        self.simulacao = None
        self.thread = None
        self.fechar_apos_calculo = False

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        visualizacao = QVBoxLayout()
        layout.addLayout(visualizacao, stretch=1)

        self.gl_view = gl.GLViewWidget()
        self.gl_view.setCameraPosition(
            pos=QVector3D(*CENTRO), distance=220, elevation=16, azimuth=-85
        )
        # Barra arrastável entre a vista 3D e as abas.
        divisor_vistas = QSplitter(Qt.Orientation.Vertical)
        divisor_vistas.setChildrenCollapsible(False)
        divisor_vistas.setHandleWidth(8)
        divisor_vistas.addWidget(self.gl_view)
        visualizacao.addWidget(divisor_vistas, stretch=1)

        self.cubo = gl.GLLinePlotItem(
            color=(0.43, 0.54, 0.60, 0.65), width=1, mode="lines", antialias=True
        )
        self.gl_view.addItem(self.cubo)
        self.n_desenhado = None
        self.escala_desenhada = None
        self.contacto_valido = False
        self.gl_view.setToolTip("Geometria 3D: roda com o rato. A animação está em Cortes 2D.")
        self.interface_pele_3d = gl.GLLinePlotItem(
            color=(0.96, 0.76, 0.61, 0.8), width=1, mode="lines", antialias=True,
        )
        self.gl_view.addItem(self.interface_pele_3d)
        self.gel_3d = gl.GLMeshItem(
            color=(0.29, 0.66, 0.95, 0.27), shader="shaded", glOptions="translucent",
        )
        self.gl_view.addItem(self.gel_3d)

        self.objeto = gl.GLMeshItem(
            meshdata=gl.MeshData.sphere(rows=24, cols=32, radius=1.0),
            color=(0.02, 0.72, 0.77, 0.47),
            shader="shaded",
            glOptions="translucent",
        )
        self.gl_view.addItem(self.objeto)
        self.objeto_2 = gl.GLMeshItem(
            color=(0.60, 0.38, 0.88, 0.50), shader="shaded", glOptions="translucent"
        )
        self.objeto_2.hide()
        self.gl_view.addItem(self.objeto_2)
        self.objeto_2.setDepthValue(-2)
        self.objeto.setDepthValue(-1)

        self.marcador_fonte = gl.GLScatterPlotItem(
            pos=np.array([FONTE], dtype=np.float32),
            size=11, color=(1.0, 0.87, 0.25, 1.0), pxMode=True,
        )
        self.gl_view.addItem(self.marcador_fonte)

        abas = QTabWidget()
        divisor_vistas.addWidget(abas)
        divisor_vistas.setSizes([350, 400])
        pagina_cortes = QWidget()
        layout_cortes = QVBoxLayout(pagina_cortes)
        layout_cortes.setContentsMargins(0, 0, 0, 0)
        self.mostrar_propagacao = QCheckBox("Mostrar propagação")
        self.mostrar_propagacao.setChecked(True)
        layout_cortes.addWidget(self.mostrar_propagacao)
        linha_cortes = QHBoxLayout()
        layout_cortes.addLayout(linha_cortes, stretch=1)
        linha_cortes.setContentsMargins(0, 0, 0, 0)
        self.imagens = []
        self.contornos = []
        self.contornos_2 = []
        self.contornos_pele = []
        self.contornos_gel = []
        self.plots_cortes = []
        self.imagem_vazia = np.zeros((N, N), dtype=np.float32)
        lut = pg.ColorMap(
            [0, 0.5, 1], [(82, 168, 255), (27, 31, 35), (255, 138, 79)]
        ).getLookupTable(nPts=256)

        for plano, eixo_x, eixo_y in (
            ("XY", "x", "y"),
            ("XZ", "x", "z"),
        ):
            plot = pg.PlotWidget(title=plano, background=(15, 17, 21))
            self.fixar_vista(plot)
            plot.setAspectLocked(True)
            plot.setLabel("bottom", eixo_x, units="mm")
            plot.setLabel("left", eixo_y, units="mm")
            for eixo in ("bottom", "left"):
                plot.getAxis(eixo).enableAutoSIPrefix(False)

            imagem = pg.ImageItem(self.imagem_vazia, axisOrder="col-major")
            imagem.setLookupTable(lut)
            imagem.setLevels((-1, 1))
            imagem.setRect(QRectF(-0.5, -0.5, N, N))
            plot.addItem(imagem)
            self.contornos.append(plot.plot(pen=pg.mkPen("#57eee3", width=1.5)))
            self.contornos_2.append(plot.plot(pen=pg.mkPen("#c1a1ff", width=1.5)))
            interface = plot.plot(pen=pg.mkPen("#f4c29c", width=1.4, style=Qt.PenStyle.DashLine))
            interface.setZValue(2)
            self.contornos_pele.append(interface)
            linha_gel = plot.plot(pen=pg.mkPen("#70bfff", width=1.2))
            linha_gel.setZValue(2)
            self.contornos_gel.append(linha_gel)
            self.plots_cortes.append(plot)
            plot.setRange(xRange=(-0.5, N - 0.5), yRange=(-0.5, N - 0.5), padding=0)
            self.imagens.append(imagem)
            linha_cortes.addWidget(plot)

        self.mostrar_propagacao.toggled.connect(self.alternar_propagacao)
        abas.addTab(pagina_cortes, "Cortes 2D")

        self.plot_pressao = pg.PlotWidget(title="Pressão nas sondas")
        self.fixar_vista(self.plot_pressao)
        self.plot_pressao.setLabel("bottom", "Tempo", units="µs")
        self.plot_pressao.setLabel("left", "Pressão", units="Pa")
        self.plot_pressao.getAxis("bottom").enableAutoSIPrefix(False)
        self.plot_pressao.getAxis("left").enableAutoSIPrefix(False)
        self.plot_pressao.addLegend()
        self.plot_pressao.showGrid(x=True, y=True, alpha=0.2)
        abas.addTab(self.plot_pressao, "Sondas de pressão")
        self.criar_visualizacao_ecos(abas)
        self.grafico_ecos.setAspectLocked(True)

        linha_tempo = QHBoxLayout()
        self.tempo = QLabel("0,00 µs")
        self.tempo.setFixedWidth(85)
        self.slider_tempo = QSlider(Qt.Orientation.Horizontal)
        self.slider_tempo.setRange(0, 0)
        self.slider_tempo.setEnabled(False)
        linha_tempo.addWidget(self.slider_tempo, stretch=1)
        linha_tempo.addWidget(self.tempo)
        visualizacao.addLayout(linha_tempo)
        legenda_contacto = QLabel("Face ativa a amarelo · Gel a azul claro · Superfície da pele a bege")
        legenda_contacto.setWordWrap(True)
        visualizacao.addWidget(legenda_contacto)

        conteudo_painel = QWidget()
        painel = QVBoxLayout(conteudo_painel)
        area_painel = QScrollArea()
        area_painel.setWidgetResizable(True)
        area_painel.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        area_painel.setFixedWidth(370)
        area_painel.setWidget(conteudo_painel)
        # ALTERADO — o cálculo e a reprodução ficam à vista ao percorrer os parâmetros.
        coluna_painel = QVBoxLayout()
        coluna_painel.addWidget(area_painel, stretch=1)
        layout.addLayout(coluna_painel)

        titulo = QLabel("Biblioteca de Objetos")
        titulo.setStyleSheet("font-size: 18px; font-weight: bold")
        painel.addWidget(titulo)

        ajustes = QGroupBox("Simulação")
        formulario = QFormLayout(ajustes)
        painel.addWidget(ajustes)

        self.botao_preset = QPushButton("Escolher preset")
        menu_preset = QMenu(self.botao_preset)

        acao_testiculo = menu_preset.addAction("Exame testicular")
        acao_testiculo.triggered.connect(self.aplicar_predefinicao_testiculo)

        acao_tibia = menu_preset.addAction("Tíbia em corte transversal")
        acao_tibia.triggered.connect(self.aplicar_predefinicao_tibia)

        acao_vesicula = menu_preset.addAction("Vesícula biliar com cálculo")
        acao_vesicula.triggered.connect(self.aplicar_predefinicao_vesicula)

        self.botao_preset.setMenu(menu_preset)
        formulario.addRow(self.botao_preset)

        grupo_1 = QGroupBox("Objeto 1")
        formulario_1 = QFormLayout(grupo_1)

        grupo_2 = QGroupBox("Objeto 2")
        formulario_2 = QFormLayout(grupo_2)

        grupo_sonda = QGroupBox("Sonda e contacto")
        formulario_sonda = QFormLayout(grupo_sonda)
        painel.addWidget(grupo_sonda)

        grupo_pele = QGroupBox("Pele e tecido de fundo")
        formulario_ambiente = QFormLayout(grupo_pele)
        painel.addWidget(grupo_pele)

        painel.addWidget(grupo_1)
        painel.addWidget(grupo_2)

        self.grelha = QSpinBox()
        self.grelha.setRange(48, 167)
        self.grelha.setValue(N)
        formulario.addRow("Vóxeis por eixo", self.grelha)
        self.passo_espacial = QDoubleSpinBox()
        self.passo_espacial.setDecimals(3)
        self.passo_espacial.setRange(0.010, 0.500)
        self.passo_espacial.setSingleStep(0.010)
        self.passo_espacial.setValue(0.100)
        self.passo_espacial.setSuffix(" mm")
        self.passo_espacial.setToolTip(
            "Define as dimensões físicas na pré-visualização. A grelha acústica é ajustada automaticamente."
        )
        formulario.addRow("Passo da geometria Δx", self.passo_espacial)

        self.x_pele = QSpinBox()
        self.raio_sonda = QSpinBox()
        self.abertura_sonda = QSpinBox()
        self.altura_sonda = QSpinBox()
        for nome, campo, minimo, maximo, valor in (
            ("Superfície da pele X", self.x_pele, 1, N - 2, X_PELE),
            ("Raio de curvatura (0 = linear)", self.raio_sonda, 0, 250, 12),
            ("Meia abertura Y", self.abertura_sonda, 1, N // 2 - 1, 10),
            ("Meia altura Z", self.altura_sonda, 0, N // 2 - 1, 2),
        ):
            campo.setRange(minimo, maximo)
            campo.setValue(valor)
            campo.setSuffix(" vóxeis")
            formulario_sonda.addRow(nome, campo)
        self.raio_sonda.setToolTip("Usa 0 para uma sonda linear e um raio positivo para uma sonda convexa.")
        self.info_sonda = QLabel()
        self.info_sonda.setWordWrap(True)
        formulario_sonda.addRow(self.info_sonda)

        # ALTERADO — o gel ocupa a lâmina entre a face ativa e a pele.
        grupo_gel = QGroupBox("Gel de acoplamento")
        grupo_gel.setToolTip(
            "A vista 3D mostra o gel na zona de contacto. "
            "No cálculo 2D, o gel é prolongado até à camada absorvente."
        )
        formulario_gel = QFormLayout(grupo_gel)
        painel.insertWidget(painel.indexOf(grupo_sonda) + 1, grupo_gel)
        self.espessura_gel = QSpinBox()
        self.espessura_gel.setRange(1, 30)
        self.espessura_gel.setValue(3)
        self.espessura_gel.setSuffix(" vóxeis")
        formulario_gel.addRow("Espessura", self.espessura_gel)
        self.c_gel = QDoubleSpinBox()
        self.rho_gel = QDoubleSpinBox()
        self.alpha_gel = QDoubleSpinBox()
        for nome, campo, casas, minimo, maximo, valor, unidade in (
            ("Velocidade c", self.c_gel, 1, 800.0, 3000.0, C_GEL, " m/s"),
            ("Densidade ρ", self.rho_gel, 1, 300.0, 4000.0, RHO_GEL, " kg/m³"),
            ("Atenuação α", self.alpha_gel, 3, 0.0, 30.0, ALPHA_GEL, " dB/cm/MHz^1,1"),
        ):
            campo.setDecimals(casas)
            campo.setRange(minimo, maximo)
            campo.setValue(valor)
            campo.setSuffix(unidade)
            formulario_gel.addRow(nome, campo)

        self.formato = QComboBox()
        self.formato.addItems(FORMATOS)
        formulario_1.addRow("Formato", self.formato)

        self.opacidade_objeto_1 = QSpinBox()
        self.opacidade_objeto_1.setRange(0, 100)
        self.opacidade_objeto_1.setValue(47)
        self.opacidade_objeto_1.setSuffix(" %")
        formulario_1.addRow("Opacidade 3D", self.opacidade_objeto_1)
        self.opacidade_objeto_1.valueChanged.connect(self.atualizar_opacidade_objeto_1)

        self.material = QComboBox()
        self.material.addItems(MATERIAIS)
        self.material.setCurrentText("Osso cortical")
        formulario_1.addRow("Objeto", self.material)

        self.rotacao_y = QDoubleSpinBox()
        self.rotacao_y.setRange(-180.0, 180.0)
        self.rotacao_y.setSingleStep(5.0)
        self.rotacao_y.setSuffix(" °")
        formulario_1.addRow("Rotação", self.rotacao_y)

        self.raio = QSpinBox()
        self.raio.setRange(4, 22)
        self.raio.setValue(8)
        self.raio.setSuffix(" px")
        formulario_1.addRow("Raio", self.raio)
        # ALTERADO — a posição axial do primeiro objeto pode ser ajustada.
        self.posicao_x_1 = QSpinBox()
        self.posicao_x_1.setRange(1, N - 2)
        self.posicao_x_1.setValue(N // 2)
        self.posicao_x_1.setSuffix(" vóxeis")
        formulario_1.addRow("Posição X", self.posicao_x_1)

        self.segundo_objeto = QCheckBox("Ativar segundo objeto")
        formulario_2.addRow(self.segundo_objeto)

        self.formato_2 = QComboBox()
        self.formato_2.addItems(FORMATOS)
        formulario_2.addRow("Formato 2", self.formato_2)

        self.material_2 = QComboBox()
        self.material_2.addItems(MATERIAIS)
        self.material_2.setCurrentText("Osso cortical")
        formulario_2.addRow("Material 2", self.material_2)

        self.raio_2 = QSpinBox()
        self.raio_2.setRange(2, 22)
        self.raio_2.setValue(7)
        self.raio_2.setSuffix(" px")
        formulario_2.addRow("Raio 2", self.raio_2)

        self.a_raio_2 = QDoubleSpinBox()
        self.b_raio_2 = QDoubleSpinBox()
        self.c_raio_2 = QDoubleSpinBox()
        for nome, campo in (
            ("A 2", self.a_raio_2),
            ("B 2", self.b_raio_2),
            ("C 2", self.c_raio_2),
        ):
            campo.setRange(0.1, 7.0)
            campo.setSingleStep(0.1)
            campo.setValue(1.0)
            campo.setSuffix(" ×")
            formulario_2.addRow(nome, campo)

        self.posicao_x_2 = QSpinBox()
        self.posicao_x_2.setRange(9, N - 2)
        self.posicao_x_2.setValue(int(0.8 * N))
        self.posicao_x_2.setSuffix(" px")
        formulario_2.addRow("Posição X 2", self.posicao_x_2)

        # ALTERADO — propriedades acústicas próprias, com valores iniciais do material.
        self.velocidade_2 = QDoubleSpinBox()
        self.densidade_2 = QDoubleSpinBox()
        self.atenuacao_2 = QDoubleSpinBox()
        for nome, campo, minimo, maximo, unidade in (
            ("Velocidade c", self.velocidade_2, 1000.0, 30000.0, " m/s"),
            ("Densidade ρ", self.densidade_2, 500.0, 30000.0, " kg/m³"),
            ("Atenuação α", self.atenuacao_2, 0.0, 30.0, " dB/cm/MHz^1,1"),
        ):
            campo.setDecimals(4)
            campo.setRange(minimo, maximo)
            campo.setSuffix(unidade)
            formulario_2.addRow(nome, campo)

        self.a_raio = QDoubleSpinBox()
        self.a_raio.setRange(0.1, 7.0)
        self.a_raio.setSingleStep(0.1)
        self.a_raio.setValue(1)
        self.a_raio.setSuffix(" ×")
        formulario_1.addRow("A", self.a_raio)

        self.b_raio = QDoubleSpinBox()
        self.b_raio.setRange(0.1, 7.0)
        self.b_raio.setSingleStep(0.1)
        self.b_raio.setValue(1)
        self.b_raio.setSuffix(" ×")
        formulario_1.addRow("B", self.b_raio)

        self.c_raio = QDoubleSpinBox()
        self.c_raio.setRange(0.1, 7.0)
        self.c_raio.setSingleStep(0.1)
        self.c_raio.setValue(1)
        self.c_raio.setSuffix(" ×")
        formulario_1.addRow("C", self.c_raio)

        self.alongamento_x = QDoubleSpinBox()
        self.alongamento_x.setRange(0.5, 67)
        self.alongamento_x.setSingleStep(0.1)
        self.alongamento_x.setValue(1.4)
        self.alongamento_x.setSuffix(" ×")
        formulario_1.addRow("Alongamento X (toro)", self.alongamento_x)

        self.alongamento_y = QDoubleSpinBox()
        self.alongamento_y.setRange(0.5, 67)
        self.alongamento_y.setSingleStep(0.1)
        self.alongamento_y.setValue(1.4)
        self.alongamento_y.setSuffix(" ×")
        formulario_1.addRow("Alongamento Y (toro)", self.alongamento_y)

        self.frequencia = QDoubleSpinBox()
        self.frequencia.setRange(0.2, 12.0)
        self.frequencia.setSingleStep(0.05)
        self.frequencia.setValue(0.60)
        self.frequencia.setSuffix(" MHz")
        formulario.addRow("Frequência", self.frequencia)

        linha_disparos = QHBoxLayout()
        self.disparos = QSlider(Qt.Orientation.Horizontal)
        self.disparos.setRange(1, 21)
        self.disparos.setSingleStep(1)
        self.disparos.setPageStep(1)
        self.disparos.setValue(9)
        self.disparos.setToolTip(
            "Número máximo de emissões no modo B. A imagem usa os pares de ângulos "
            "concluídos dentro dos 180 s de cálculo."
        )
        valor_disparos = QLabel(str(self.disparos.value()))
        self.disparos.valueChanged.connect(lambda n: valor_disparos.setText(str(n)))
        self.disparos.valueChanged.connect(self.limpar_resultado)
        linha_disparos.addWidget(self.disparos, stretch=1)
        linha_disparos.addWidget(valor_disparos)
        formulario.addRow("Disparos de emissão", linha_disparos)

        self.amplitude = QDoubleSpinBox()
        self.amplitude.setRange(0.1, 10.0)
        self.amplitude.setSingleStep(0.1)
        self.amplitude.setValue(3.6)
        self.amplitude.setSuffix(" Pa")
        formulario.addRow("Amplitude", self.amplitude)

        self.alpha = QDoubleSpinBox()
        self.alpha.setDecimals(4)
        self.alpha.setRange(0.0, 30.0)
        self.alpha.setSingleStep(0.1)
        self.alpha.setSuffix(" dB/cm/MHz^1,1")
        self.alpha.setToolTip("Coeficiente de absorção do objeto.")
        formulario_1.addRow("Atenuação α", self.alpha)

        self.c = QDoubleSpinBox()
        self.c.setDecimals(4)
        self.c.setRange(1000.0, 30000.0)
        self.c.setSingleStep(0.1)
        self.c.setSuffix(" m/s")
        self.c.setToolTip("Velocidade de propagação no objeto.")
        formulario_1.addRow("Velocidade c", self.c)

        self.rho = QDoubleSpinBox()
        self.rho.setDecimals(4)
        self.rho.setRange(500.0, 30000.0)
        self.rho.setSingleStep(0.1)
        self.rho.setSuffix(" kg/m³")
        self.rho.setToolTip("Densidade do objeto.")
        formulario_1.addRow("Densidade ρ", self.rho)

        self.dispersao_objeto = QDoubleSpinBox()
        self.dispersao_objeto_2 = QDoubleSpinBox()

        for campo, formulario in (
            (self.dispersao_objeto, formulario_1),
            (self.dispersao_objeto_2, formulario_2),
        ):
            campo.setDecimals(2)
            campo.setRange(0, 5)
            campo.setSingleStep(0.1)
            campo.setSuffix(" %")
            formulario.addRow("Dispersão da densidade", campo)
            campo.valueChanged.connect(self.limpar_resultado)

        self.c_ambiente = QDoubleSpinBox()
        self.c_ambiente.setRange(1000.0, 30000.0)
        self.c_ambiente.setValue(C_PELE)
        self.c_ambiente.setSuffix(" m/s")
        formulario_ambiente.addRow("Velocidade c", self.c_ambiente)

        self.rho_ambiente = QDoubleSpinBox()
        self.rho_ambiente.setRange(500.0, 30000.0)
        self.rho_ambiente.setValue(RHO_PELE)
        self.rho_ambiente.setSuffix(" kg/m³")
        formulario_ambiente.addRow("Densidade ρ", self.rho_ambiente)

        self.alpha_ambiente = QDoubleSpinBox()
        self.alpha_ambiente.setDecimals(4)
        self.alpha_ambiente.setRange(0.0, 30.0)
        self.alpha_ambiente.setSingleStep(0.1)
        self.alpha_ambiente.setValue(ALPHA_PELE)
        self.alpha_ambiente.setSuffix(" dB/cm/MHz^1,1")
        formulario_ambiente.addRow("Atenuação α", self.alpha_ambiente)

        self.dispersao_pele = QDoubleSpinBox()
        self.dispersao_pele.setRange(0, 5)
        self.dispersao_pele.setSingleStep(0.25)
        self.dispersao_pele.setSuffix(" %")
        self.dispersao_pele.setToolTip("Variação aleatória da densidade da pele que gera ecos difusos.")
        formulario_ambiente.addRow("Dispersão da densidade", self.dispersao_pele)
        self.semente_pele = QSpinBox()
        self.semente_pele.setRange(0, 2147483647)
        self.semente_pele.setValue(7)
        formulario_ambiente.addRow("Semente da dispersão", self.semente_pele)

        self.info = QLabel()
        painel.addWidget(self.info)
        self.botao = QPushButton("Simular ondas")
        coluna_painel.addWidget(self.botao)
        self.progresso = QProgressBar()
        self.progresso.setRange(0, 0)
        self.progresso.hide()
        coluna_painel.addWidget(self.progresso)

        self.btn_play = QPushButton("▶ Reproduzir")
        self.btn_play.setEnabled(False)
        coluna_painel.addWidget(self.btn_play)

        self.estado = QLabel("Seleciona um objeto e clica em Simular ondas.")
        self.estado.setWordWrap(True)
        coluna_painel.addWidget(self.estado)
        painel.addStretch()

        self.timer_animacao = QTimer(self)
        self.timer_animacao.setInterval(33)
        self.timer_animacao.timeout.connect(self.avancar_frame)

        self.slider_tempo.valueChanged.connect(self.atualizar_frame)
        self.slider_tempo.sliderPressed.connect(self.pausar_animacao)
        self.btn_play.clicked.connect(self.toggle_animacao)
        self.botao.clicked.connect(self.executar_simulacao)
        self.material.currentTextChanged.connect(self.atualizar_material)

        self.timer_geometria = QTimer(self)
        self.timer_geometria.setSingleShot(True)
        self.timer_geometria.setInterval(90)
        self.timer_geometria.timeout.connect(self.atualizar_parametros)
        for campo in (
            self.raio, self.a_raio, self.b_raio, self.c_raio,
            self.raio_2, self.a_raio_2, self.b_raio_2, self.c_raio_2, self.posicao_x_2,
            self.grelha, self.passo_espacial, self.posicao_x_1, self.espessura_gel,
            self.x_pele, self.raio_sonda, self.abertura_sonda, self.altura_sonda,
        ):
            campo.valueChanged.connect(self.agendar_geometria)

        self.formato.currentTextChanged.connect(self.atualizar_parametros)
        self.rotacao_y.editingFinished.connect(self.atualizar_parametros)
        self.alongamento_x.editingFinished.connect(self.atualizar_parametros)
        self.alongamento_y.editingFinished.connect(self.atualizar_parametros)
        self.segundo_objeto.toggled.connect(self.atualizar_parametros)
        self.formato_2.currentTextChanged.connect(self.atualizar_parametros)
        self.material_2.currentTextChanged.connect(self.atualizar_material_2)
        for campo in (
            self.frequencia, self.amplitude, self.alpha,
            self.c_ambiente, self.rho_ambiente, self.alpha_ambiente,
            self.c_gel, self.rho_gel, self.alpha_gel,
            self.dispersao_pele, self.semente_pele,
            self.velocidade_2, self.densidade_2, self.atenuacao_2,
        ):
            campo.valueChanged.connect(self.limpar_resultado)

        self.c.valueChanged.connect(self.atualizar_info_material)
        self.rho.valueChanged.connect(self.atualizar_info_material)
        self.atualizar_material()
        self.atualizar_material_2()
        self.atualizar_parametros()

    @staticmethod
    def fixar_vista(plot):
        item = plot.getPlotItem()
        item.setMenuEnabled(False)
        item.hideButtons()
        vista = item.getViewBox()
        vista.disableAutoRange()
        vista.setMouseEnabled(x=False, y=False)

    @Slot()
    def agendar_geometria(self):
        self.timer_geometria.start()

    def criar_visualizacao_ecos(self, abas):
        pagina = QWidget()
        layout = QVBoxLayout(pagina)
        layout.setContentsMargins(4, 4, 4, 4)

        self.grafico_ecos = pg.PlotWidget(title="Ecografia simulada · modo B", background="k")
        self.grafico_ecos.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        self.fixar_vista(self.grafico_ecos)
        self.grafico_ecos.setAspectLocked(True)
        self.grafico_ecos.getViewBox().invertY(True)
        self.grafico_ecos.setLabel("bottom", "Posição lateral", units="mm")
        self.grafico_ecos.setLabel("left", "Profundidade desde a pele", units="mm")
        for eixo in ("bottom", "left"):
            self.grafico_ecos.getAxis(eixo).enableAutoSIPrefix(False)

        self.imagem_ecos = pg.ImageItem(axisOrder="col-major")
        self.grafico_ecos.addItem(self.imagem_ecos)
        layout.addWidget(self.grafico_ecos, stretch=1)

        self.detalhes_ecos = QPlainTextEdit()
        self.detalhes_ecos.setReadOnly(True)
        self.detalhes_ecos.setFixedHeight(72)
        layout.addWidget(self.detalhes_ecos)

        ferramentas = QHBoxLayout()
        ferramentas.addWidget(QLabel("Ganho"))
        self.ganho_ecos = QSpinBox()
        self.ganho_ecos.setRange(-40, 30)
        self.ganho_ecos.setValue(0)
        self.ganho_ecos.setSuffix(" dB")
        self.ganho_ecos.setToolTip("Ajusta o brilho do modo B em relação ao nível dos ecos.")
        self.ganho_ecos.valueChanged.connect(self.redesenhar_ecografia)
        ferramentas.addWidget(self.ganho_ecos)
        ferramentas.addWidget(QLabel("Ganho em profundidade"))
        self.tgc_ecos = QSpinBox()
        self.tgc_ecos.setRange(0, 40)
        self.tgc_ecos.setSuffix(" dB/cm")
        self.tgc_ecos.setToolTip("Aumenta o ganho dos ecos mais profundos na imagem.")
        self.tgc_ecos.valueChanged.connect(self.redesenhar_ecografia)
        ferramentas.addWidget(self.tgc_ecos)
        ferramentas.addStretch()

        self.botao_guardar_ecos = QPushButton("Guardar ecografia (PNG)")
        self.botao_guardar_ecos.setEnabled(False)
        self.botao_guardar_ecos.clicked.connect(self.guardar_ecografia)
        ferramentas.addWidget(self.botao_guardar_ecos)
        layout.addLayout(ferramentas)
        abas.addTab(pagina, "Ultrassonografia")
        self.abas_visualizacao = abas
        self.pagina_ecos = pagina

        self.limpar_ecos()

    def limpar_ecos(self):
        self.imagem_ecos.hide()
        self.grafico_ecos.setTitle("Ecografia simulada · modo B")
        self.botao_guardar_ecos.setEnabled(False)
        self.detalhes_ecos.setPlainText("Simula as ondas para adquirir ecos.")

    def preparar_ecos(self, ecos):
        lateral = 1000 * ecos.lateral
        profundidade = 1000 * ecos.eixo_profundidade
        dx = lateral[1] - lateral[0]
        dz = profundidade[1] - profundidade[0] if len(profundidade) > 1 else self.passo_espacial.value()
        esquerda, direita = lateral[0] - dx / 2, lateral[-1] + dx / 2
        topo, fundo = profundidade[0] - dz / 2, profundidade[-1] + dz / 2

        self.atualizar_ecos(ecos)
        self.imagem_ecos.setRect(QRectF(esquerda, topo, direita - esquerda, fundo - topo))

        # Ajusta o enquadramento à área coberta pela aquisição.
        iy, ix = np.nonzero(ecos.mascara_setor)
        if iy.size:
            esquerda, direita = lateral[iy.min()] - dx / 2, lateral[iy.max()] + dx / 2
            topo = min(0.0, profundidade[ix.min()] - dz / 2)
            fundo = profundidade[ix.max()] + dz / 2
        self.grafico_ecos.setRange(
            xRange=(esquerda, direita),
            yRange=(topo, fundo),
            padding=0.03,
            disableAutoRange=True,
        )
        geometria = "sonda linear" if self.raio_sonda.value() == 0 else "sonda convexa"
        self.grafico_ecos.setTitle(
            f"Ecografia simulada · modo B · {geometria} · {ecos.numero_disparos} disparos"
        )
        self.imagem_ecos.show()
        duracao = f"{ecos.duracao:.1f}".replace(".", ",")
        frequencia = f"{ecos.frequencia / 1e6:.2f}".replace(".", ",")
        passo = f"{ecos.passo_acustico * 1e3:.3f}".replace(".", ",")
        self.detalhes_ecos.setPlainText(
            f"{ecos.numero_disparos}/{ecos.solicitados} disparos · {duracao} s de cálculo\n"
            f"{frequencia} MHz · Passo da grelha acústica 2D: {passo} mm"
        )

    def atualizar_ecos(self, ecos):
        validos = ecos.mascara_setor & np.isfinite(ecos.imagem_db)
        niveis = np.full_like(ecos.imagem_db, ecos.piso_db)
        ganho = self.ganho_ecos.value() + self.tgc_ecos.value() * ecos.eixo_profundidade * 100
        niveis[validos] = np.clip((ecos.imagem_db + ganho[None, :])[validos], ecos.piso_db, 0)
        self.imagem_ecos.setImage(niveis, autoLevels=False, levels=(ecos.piso_db, 0))
        self.botao_guardar_ecos.setEnabled(bool(np.any(validos)))

    def redesenhar_ecografia(self):
        if self.simulacao is not None:
            self.atualizar_ecos(self.simulacao.ecos)

    def guardar_ecografia(self):
        if not self.botao_guardar_ecos.isEnabled():
            return

        imagem = self.grafico_ecos.grab()
        caminho, _ = QFileDialog.getSaveFileName(
            self, "Guardar ecografia simulada", "ecografia.png", "Imagem PNG (*.png)"
        )
        if not caminho:
            return
        if not caminho.lower().endswith(".png"):
            caminho += ".png"
        if not imagem.save(caminho, "PNG"):
            QMessageBox.warning(
                self, "Erro ao guardar", "Não foi possível gravar o ficheiro PNG."
            )

    def atualizar_material(self):
        propriedades = MATERIAIS[self.material.currentText()]
        self.c.setValue(propriedades["speed"])
        self.rho.setValue(propriedades["density"])
        self.alpha.setValue(propriedades["absorption"])
        self.dispersao_objeto.setValue(propriedades["dispersao"])
        self.atualizar_info_material()

    @Slot()
    def atualizar_info_material(self):
        c, rho = self.c.value(), self.rho.value()
        self.info.setText(
            f"Velocidade  {c} m/s\n"
            f"Densidade  {rho} kg/m³\n"
            f"Impedância  {c * rho / 1e6:.3f} MRayl"
        )
        self.limpar_resultado()

    @Slot()
    def atualizar_material_2(self):
        propriedades = MATERIAIS[self.material_2.currentText()]
        self.velocidade_2.setValue(propriedades["speed"])
        self.densidade_2.setValue(propriedades["density"])
        self.atenuacao_2.setValue(propriedades["absorption"])
        self.dispersao_objeto_2.setValue(propriedades["dispersao"])
        self.limpar_resultado()

    def dados_segundo_objeto(self):
        if not self.segundo_objeto.isChecked():
            return None
        return {
            "formato": self.formato_2.currentText(),
            "material": self.material_2.currentText(),
            "raio": self.raio_2.value(),
            "a": self.a_raio_2.value(),
            "b": self.b_raio_2.value(),
            "c": self.c_raio_2.value(),
            "x": self.posicao_x_2.value(),
            "velocidade": self.velocidade_2.value(),
            "densidade": self.densidade_2.value(),
            "atenuacao": self.atenuacao_2.value(),
            "dispersao": self.dispersao_objeto_2.value(),
        }


    # ALTERADO — a pré-visualização e o solver usam a mesma geometria de contacto.
    def desenhar_contacto(self, n, dx_mm):
        for campo, maximo in (
            (self.x_pele, n - 2),
            (self.abertura_sonda, n // 2 - 1),
            (self.altura_sonda, n // 2 - 1),
        ):
            bloqueado = campo.blockSignals(True)
            campo.setMaximum(maximo)
            campo.blockSignals(bloqueado)

        raio = self.raio_sonda.value()
        tipo_sonda = "Linear" if raio == 0 else f"Convexa · Raio {raio * dx_mm:.2f} mm"
        self.info_sonda.setText(
            f"{tipo_sonda} · "
            f"Meia abertura {self.abertura_sonda.value() * dx_mm:.2f} mm · "
            f"Meia altura {self.altura_sonda.value() * dx_mm:.2f} mm · "
            f"Gel {self.espessura_gel.value() * dx_mm:.2f} mm"
        )
        try:
            contacto = geometria_acoplamento(
                n, self.x_pele.value(), self.raio_sonda.value(),
                self.abertura_sonda.value(), self.altura_sonda.value(),
                self.espessura_gel.value(),
            )
        except ValueError as erro:
            self.interface_pele_3d.hide()
            self.marcador_fonte.hide()
            self.gel_3d.hide()
            for contorno in self.contornos_pele + self.contornos_gel:
                contorno.setData([], [])
            return str(erro)

        pontos = np.argwhere(contacto.fonte).astype(np.float32)
        self.marcador_fonte.setData(pos=pontos, size=4)
        self.marcador_fonte.show()
        fronteira = np.asarray(contacto.limite_pele, dtype=float) - 0.5
        y = np.arange(n, dtype=float)
        cy = n // 2

        # A superfície varia em Y e estende-se pelas fatias Z.
        linhas = []
        for z in (0, n // 2, n - 1):
            perfil = np.column_stack((fronteira, y, np.full(n, z)))
            linhas.append(np.stack((perfil[:-1], perfil[1:]), axis=1).reshape(-1, 3))
        for i in (0, cy, n - 1):
            linhas.append(np.array([[fronteira[i], i, 0], [fronteira[i], i, n - 1]]))
        self.interface_pele_3d.setData(pos=np.concatenate(linhas).astype(np.float32))
        self.interface_pele_3d.show()

        bordas = np.array([-0.5, n - 0.5]) * dx_mm
        self.contornos_pele[0].setData(fronteira * dx_mm, y * dx_mm)
        self.contornos_pele[1].setData(np.full(2, fronteira[cy] * dx_mm), bordas)
        # A vista 3D desenha apenas o volume de gel junto à face ativa.
        gel = contacto.gel
        vertices, faces, _, _ = marching_cubes(np.pad(gel.astype(np.float32), 1), level=0.5)
        self.gel_3d.setMeshData(vertexes=vertices - 1, faces=faces)
        self.gel_3d.show()
        fatias = (gel[:, :, cy], gel[:, cy, :])
        for item, fatia in zip(self.contornos_gel, fatias):
            linhas = find_contours(np.pad(fatia.astype(np.float32), 1), 0.5)
            if linhas:
                pontos = np.concatenate([np.vstack((linha - 1, [np.nan, np.nan])) for linha in linhas])
                item.setData(pontos[:, 0] * dx_mm, pontos[:, 1] * dx_mm, connect="finite")
            else:
                item.setData([], [])
        return ""

    @Slot()
    def aplicar_predefinicao_testiculo(self):
        self.timer_geometria.stop()
        if self.thread is not None:
            return

        # Suspende os sinais enquanto muda a grelha e os parâmetros interdependentes.
        n = 160
        material = "Testículo (modelo didático)"
        propriedades = MATERIAIS[material]
        valores = (
            (self.grelha.setValue, n),
            (self.passo_espacial.setValue, 0.36),
            (self.frequencia.setValue, 3.0),
            (self.x_pele.setValue, 12),
            (self.posicao_x_1.setMaximum, n - 2),
            (self.posicao_x_1.setValue, 51),
            (self.formato.setCurrentText, "Elipsoide"),
            (self.material.setCurrentText, material),
            (self.rotacao_y.setValue, 0),
            (self.raio.setValue, 20),
            (self.a_raio.setValue, 1.4),
            (self.b_raio.setValue, 3.5),
            (self.c_raio.setValue, 2.1),
            (self.c.setValue, propriedades["speed"]),
            (self.rho.setValue, propriedades["density"]),
            (self.alpha.setValue, propriedades["absorption"]),
            (self.dispersao_objeto.setValue, 0.70),
            (self.dispersao_pele.setValue, 0.10),
            (self.c_ambiente.setValue, 1540),
            (self.rho_ambiente.setValue, 1040),
            (self.alpha_ambiente.setValue, 0.50),
            (self.espessura_gel.setValue, 3),
            (self.abertura_sonda.setMaximum, n // 2 - 1),
            (self.abertura_sonda.setValue, 60),
            (self.altura_sonda.setValue, 8),
            (self.raio_sonda.setValue, 0),
            (self.segundo_objeto.setChecked, False),
            (self.ganho_ecos.setValue, 0),
            (self.tgc_ecos.setValue, 0),
        )
        bloqueados = {alterar.__self__ for alterar, _ in valores}
        sinais = {campo: campo.blockSignals(True) for campo in bloqueados}
        try:
            for alterar, valor in valores:
                alterar(valor)
        finally:
            for campo, bloqueado in sinais.items():
                campo.blockSignals(bloqueado)

        self.disparos.setValue(21)
        self.atualizar_info_material()
        self.atualizar_parametros(conservar_posicao=True)

    @Slot()
    def aplicar_predefinicao_tibia(self):
        if self.thread is not None:
            return

        self.timer_geometria.stop()

        # Grelha, sonda linear e aquisição.
        self.grelha.setValue(112)
        self.passo_espacial.setValue(0.360)
        self.raio_sonda.setValue(0)
        self.x_pele.setValue(12)
        self.espessura_gel.setValue(3)
        self.c_gel.setValue(C_GEL)
        self.rho_gel.setValue(RHO_GEL)
        self.alpha_gel.setValue(ALPHA_GEL)
        self.abertura_sonda.setMaximum(55)
        self.abertura_sonda.setValue(44)
        self.altura_sonda.setValue(8)
        self.frequencia.setValue(2.0)
        self.amplitude.setValue(3.6)
        self.disparos.setValue(9)

        # Fundo equivalente de tecido mole.
        self.c_ambiente.setValue(1510)
        self.rho_ambiente.setValue(1020)
        self.alpha_ambiente.setValue(0.55)
        self.dispersao_pele.setValue(0.25)
        self.semente_pele.setValue(7)

        # Objeto 1: região muscular.
        self.posicao_x_1.setMaximum(110)
        self.formato.setCurrentText("Cilindro")
        self.material.setCurrentText("Músculo")
        self.atualizar_material()
        self.posicao_x_1.setValue(55)
        self.raio.setValue(20)
        self.rotacao_y.setValue(0)
        self.a_raio.setValue(1.85)
        self.b_raio.setValue(1.70)
        self.c_raio.setValue(2.50)

        # Objeto 2: tíbia, com o eixo comprido ao longo de Z.
        self.posicao_x_2.setMaximum(110)
        self.formato_2.setCurrentText("Cilindro")
        self.material_2.setCurrentText("Osso cortical")
        self.atualizar_material_2()
        self.posicao_x_2.setValue(48)
        self.raio_2.setValue(18)
        self.a_raio_2.setValue(1.00)
        self.b_raio_2.setValue(1.70)
        self.c_raio_2.setValue(2.40)

        self.ganho_ecos.setValue(0)
        self.tgc_ecos.setValue(0)

        bloqueado = self.segundo_objeto.blockSignals(True)
        self.segundo_objeto.setChecked(True)
        self.segundo_objeto.blockSignals(bloqueado)
        self.atualizar_parametros(conservar_posicao=True)

    @Slot()
    def aplicar_predefinicao_vesicula(self):
        if self.thread is not None:
            return

        # Reutiliza a grelha de 112 vóxeis e a sonda linear.
        self.aplicar_predefinicao_tibia()
        self.frequencia.setValue(2.5)
        self.disparos.setValue(9)

        # Objeto 1: cavidade da vesícula, preenchida por líquido.
        self.formato.setCurrentText("Elipsoide")
        self.material.setCurrentText("Cisto (água)")
        self.atualizar_material()
        self.posicao_x_1.setValue(55)
        self.raio.setValue(20)
        self.rotacao_y.setValue(0)
        self.a_raio.setValue(1.50)
        self.b_raio.setValue(2.00)
        self.c_raio.setValue(1.50)

        # Objeto 2: cálculo junto à parede posterior.
        self.formato_2.setCurrentText("Elipsoide")
        self.material_2.setCurrentText("Osso cortical")
        self.atualizar_material_2()
        self.posicao_x_2.setValue(78)
        self.raio_2.setValue(6)
        self.a_raio_2.setValue(1.00)
        self.b_raio_2.setValue(1.00)
        self.c_raio_2.setValue(1.00)

        self.atualizar_parametros(conservar_posicao=True)

    def atualizar_opacidade_objeto_1(self, valor):
        self.objeto.setColor((0.02, 0.72, 0.77, valor / 100))
        self.objeto.setVisible(valor > 0)


    def atualizar_parametros(self, *, conservar_posicao=False):
        self.timer_geometria.stop()
        if self.thread is not None:
            return
        self.limpar_resultado()

        n = self.grelha.value()
        dx_mm = self.passo_espacial.value()
        cx = cy = cz = n // 2
        if n != self.n_desenhado:
            estava_centrado = self.posicao_x_1.value() == (self.n_desenhado or N) // 2
            self.n_desenhado = n
            self.posicao_x_1.blockSignals(True)
            self.posicao_x_1.setMaximum(n - 2)
            if estava_centrado and not conservar_posicao:
                self.posicao_x_1.setValue(cx)
            self.posicao_x_1.blockSignals(False)
            # Preserva a posição X do segundo objeto se couber na nova grelha.
            self.posicao_x_2.blockSignals(True)
            self.posicao_x_2.setMaximum(n - 2)
            self.posicao_x_2.blockSignals(False)

            lo, hi = -0.5, n - 0.5
            base = np.array([(lo, lo, lo), (hi, lo, lo), (hi, hi, lo), (lo, hi, lo)])
            topo = base + np.array([0, 0, n])
            arestas = np.concatenate((
                np.stack((base, np.roll(base, -1, axis=0)), axis=1),
                np.stack((topo, np.roll(topo, -1, axis=0)), axis=1),
                np.stack((base, topo), axis=1),
            )).reshape(-1, 3)

            self.cubo.setData(pos=arestas.astype(np.float32))
            self.gl_view.setCameraPosition(pos=QVector3D(cx, cy, cz), distance=2.75 * n)
            self.imagem_vazia = np.zeros((n, n), dtype=np.float32)

        if self.escala_desenhada != (n, dx_mm):
            self.escala_desenhada = (n, dx_mm)
            lo, hi = -0.5 * dx_mm, (n - 0.5) * dx_mm
            for imagem, plot in zip(self.imagens, self.plots_cortes):
                imagem.setImage(self.imagem_vazia, autoLevels=False)
                imagem.setRect(QRectF(lo, lo, n * dx_mm, n * dx_mm))
                plot.setRange(xRange=(lo, hi), yRange=(lo, hi), padding=0)

        erro_contacto = self.desenhar_contacto(n, dx_mm)
        self.contacto_valido = not erro_contacto
        raio = self.raio.value()
        mascara = criar_mascara(
            self.formato.currentText(),
            raio * self.a_raio.value(),
            raio * self.b_raio.value(),
            raio * self.c_raio.value(),
            self.rotacao_y.value(),
            self.alongamento_x.value(),
            self.alongamento_y.value(),
            n,
            centro=(self.posicao_x_1.value(), cy, cz),
        )
        dados_2 = self.dados_segundo_objeto()
        mascara_2 = mascara_segundo_objeto(n, dados_2) if dados_2 else None

        for malha, contornos, regiao in (
            (self.objeto, self.contornos, mascara),
            (self.objeto_2, self.contornos_2, mascara_2),
        ):
            if regiao is None or not regiao.any():
                malha.hide()
                for contorno in contornos:
                    contorno.setData([], [])
                continue

            volume = np.pad(regiao.astype(np.float32), 1)
            vertices, faces, _, _ = marching_cubes(volume, level=0.5)
            malha.setMeshData(vertexes=vertices - 1, faces=faces)
            if malha is self.objeto:
                self.atualizar_opacidade_objeto_1(self.opacidade_objeto_1.value())
            else:
                malha.show()

            cortes = (regiao[:, :, cz], regiao[:, cy, :])
            for contorno, corte in zip(contornos, cortes):
                linhas = find_contours(np.pad(corte.astype(np.float32), 1), 0.5)
                if linhas:
                    # Mantém separados os contornos interior e exterior do toro.
                    pontos = np.concatenate([
                        np.vstack((linha - 1, [np.nan, np.nan])) for linha in linhas
                    ])
                    contorno.setData(pontos[:, 0] * dx_mm, pontos[:, 1] * dx_mm, connect="finite")
                else:
                    contorno.setData([], [])

        self.ativar_controlos(True)
        if erro_contacto:
            self.estado.setText(erro_contacto)
        elif mascara_2 is not None and np.any(mascara & mascara_2):
            self.estado.setText("Na zona de sobreposição são usadas as propriedades do Objeto 2.")
        else:
            self.estado.setText("Clica em Simular ondas para aplicar os parâmetros.")

    def limpar_resultado(self):
        if self.simulacao is None:
            return

        self.pausar_animacao()
        self.simulacao = None
        self.slider_tempo.setRange(0, 0)
        self.slider_tempo.setEnabled(False)
        self.btn_play.setEnabled(False)
        self.tempo.setText("0,00 µs")
        self.limpar_ecos()
        for imagem in self.imagens:
            imagem.setImage(self.imagem_vazia, autoLevels=False)
        self.plot_pressao.clear()
        self.estado.setText("Clica em Simular ondas para aplicar os parâmetros.")

    def executar_simulacao(self):
        if self.thread is not None:
            return
        if self.timer_geometria.isActive():
            self.atualizar_parametros()
        if not self.contacto_valido:
            return

        self.limpar_resultado()
        self.ativar_controlos(False)
        self.estado.setText("A calcular com k-Wave…")
        self.progresso.show()

        self.thread = QThread(self)
        raio = self.raio.value()
        self.worker = TrabalhadorSimulacao({
            "N": self.grelha.value(),
            "dx": self.passo_espacial.value() * 1e-3,
            "x_pele": self.x_pele.value(),
            "espessura_gel": self.espessura_gel.value(),
            "c_gel": self.c_gel.value(),
            "rho_gel": self.rho_gel.value(),
            "alpha_gel": self.alpha_gel.value(),
            "x_objeto": self.posicao_x_1.value(),
            "raio_sonda": self.raio_sonda.value(),
            "abertura_sonda": self.abertura_sonda.value(),
            "altura_sonda": self.altura_sonda.value(),
            "dispersao_pele": self.dispersao_pele.value(),
            "semente_pele": self.semente_pele.value(),
            "formato": self.formato.currentText(),
            "raio": raio,
            "material": self.material.currentText(),
            "frequencia": self.frequencia.value() * 1e6,
            "numero_disparos": self.disparos.value(),
            "amplitude": self.amplitude.value(),
            "alpha": self.alpha.value(),
            "c": self.c.value(),
            "rho": self.rho.value(),
            "a_raio": raio * self.a_raio.value(),
            "b_raio": raio * self.b_raio.value(),
            "c_raio": raio * self.c_raio.value(),
            "c_ambiente": self.c_ambiente.value(),
            "rho_ambiente": self.rho_ambiente.value(),
            "alpha_ambiente": self.alpha_ambiente.value(),
            "angulo_y": self.rotacao_y.value(),
            "alongamento_x": self.alongamento_x.value(),
            "alongamento_y": self.alongamento_y.value(),
            "objeto_2": self.dados_segundo_objeto(),
            "dispersao_objeto": self.dispersao_objeto.value(),
        })
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.correr)
        self.worker.resultado.connect(self.sucesso_simulacao)
        self.worker.erro.connect(self.erro_simulacao)
        self.worker.progresso.connect(self.estado.setText)
        self.worker.resultado.connect(self.thread.quit)
        self.worker.erro.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.simulacao_terminou)
        self.thread.start()

    def ativar_controlos(self, ativar):
        for campo in (
            self.material,
            self.disparos, 
            self.botao_preset,
            self.raio,
            self.frequencia,
            self.amplitude,
            self.alpha,
            self.c,
            self.rho,
            self.a_raio,
            self.b_raio,
            self.c_raio,
            self.c_ambiente,
            self.rho_ambiente,
            self.alpha_ambiente,
            self.rotacao_y,
            self.alongamento_x,
            self.alongamento_y,
            self.grelha,
            self.passo_espacial,
            self.posicao_x_1,
            self.x_pele, self.raio_sonda, self.abertura_sonda, self.altura_sonda,
            self.espessura_gel, self.c_gel, self.rho_gel, self.alpha_gel,
            self.dispersao_pele, self.semente_pele,
            self.formato,
            self.segundo_objeto,
            self.dispersao_objeto,
        ):
            campo.setEnabled(ativar)
        self.botao.setEnabled(ativar and self.contacto_valido)

        for campo in (
            self.formato_2, self.material_2, self.raio_2, self.posicao_x_2,
            self.a_raio_2, self.b_raio_2, self.c_raio_2,
            self.velocidade_2, self.densidade_2, self.atenuacao_2, self.dispersao_objeto_2,
        ):
            campo.setEnabled(ativar and self.segundo_objeto.isChecked())

    @Slot(object)
    def sucesso_simulacao(self, resultado):
        if self.fechar_apos_calculo:
            return

        self.simulacao = resultado
        self.preparar_ecos(resultado.ecos)
        self.slider_tempo.setRange(0, len(resultado.tempos) - 1)
        self.slider_tempo.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.atualizar_grafico()
        self.atualizar_frame(0)
        self.estado.setText(f"Ecografia pronta · {resultado.tempos[-1] * 1e6:.2f} µs de propagação.")
        self.abas_visualizacao.setCurrentWidget(self.pagina_ecos)

    @Slot(str)
    def erro_simulacao(self, mensagem):
        self.estado.setText("Erro na simulação.")
        if not self.fechar_apos_calculo:
            QMessageBox.critical(self, "Erro no k-Wave", mensagem)

    @Slot()
    def simulacao_terminou(self):
        self.thread.deleteLater()
        self.thread = None
        self.worker = None
        self.progresso.hide()
        self.ativar_controlos(True)
        if self.fechar_apos_calculo:
            self.close()

    def closeEvent(self, evento):
        if self.thread is not None:
            self.fechar_apos_calculo = True
            self.hide()
            evento.ignore()
            return
        self.pausar_animacao()
        super().closeEvent(evento)

    def atualizar_grafico(self):
        sim = self.simulacao
        self.plot_pressao.clear()

        limite = max(float(np.max(np.abs(sim.sondas))), 1e-12)
        self.plot_pressao.setRange(
            xRange=(0, sim.tempos_sondas[-1] * 1e6),
            yRange=(-1.05 * limite, 1.05 * limite),
            padding=0,
            disableAutoRange=True,
        )
        nomes = ("Rasa", "Intermédia", "Profunda")
        cores = ("#f6d65b", "#57eee3", "#ff8a4f")
        for sinal, nome, cor in zip(sim.sondas, nomes, cores):
            self.plot_pressao.plot(
                sim.tempos_sondas * 1e6,
                sinal,
                pen=pg.mkPen(cor, width=2),
                name=nome,
            )

    def toggle_animacao(self):
        if self.timer_animacao.isActive():
            self.pausar_animacao()
        else:
            self.iniciar_animacao()

    def iniciar_animacao(self):
        if self.simulacao is None:
            return
        if self.slider_tempo.value() == self.slider_tempo.maximum():
            self.slider_tempo.setValue(0)
        self.btn_play.setText("⏸ Pausar")
        self.timer_animacao.start()

    def pausar_animacao(self):
        self.timer_animacao.stop()
        self.btn_play.setText("▶ Reproduzir")

    def avancar_frame(self):
        atual = self.slider_tempo.value()
        if atual == self.slider_tempo.maximum():
            self.pausar_animacao()
            return
        self.slider_tempo.setValue(atual + 1)


    @Slot(bool)
    def alternar_propagacao(self, mostrar):
        for imagem in self.imagens:
            imagem.setVisible(mostrar)

    
    def atualizar_cortes(self, sim, i):
        cortes, limite = sim.cortes, sim.limite
        for imagem, corte in zip(self.imagens, cortes):
            imagem.setImage(corte[i], autoLevels=False, levels=(-limite, limite))

    def atualizar_frame(self, i):
        sim = self.simulacao
        if sim is None:
            return

        tempo_us = f"{sim.tempos[i] * 1e6:.2f}".replace(".", ",")
        self.tempo.setText(f"{tempo_us} µs")
        self.atualizar_cortes(sim, i)



if __name__ == "__main__":
    app = QApplication.instance() or QApplication(sys.argv)
    janela = Demonstrador()
    janela.show()
    sys.exit(app.exec())