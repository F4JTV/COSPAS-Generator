#!/usr/bin/env python3
"""
Générateur de signal audio Cospas-Sarsat 406 MHz
Compatible avec le décodeur dec406 de F4EHY

Le signal attendu par dec406:
- Entrée audio WAV 16 bits mono (sortie discriminateur NFM de SDR++)
- Débit: 400 bps
- Modulation: Biphase-L (Manchester) - transition au milieu de chaque bit
- Détection par autocorrélation sur 2 bits

Polynômes BCH du décodeur:
- BCH-1: g(x) = 1001101101100111100011 (22 coefficients)
- BCH-2: g(x) = 1010100111001 (13 coefficients)

F4JTV / ADRASEC 06 - Usage test uniquement
"""

import argparse
import numpy as np
from scipy.io import wavfile
import sys
from dataclasses import dataclass
from typing import Tuple, List
import re


# === Constantes ===
SYMBOL_RATE = 400  # 400 bps

# Patterns de synchronisation (bits 0-23)
# Bits 0-14: Bit sync = 15 x "1"
# Bits 15-23: Frame sync = "000101111" (normal) ou "011010000" (test)
BIT_SYNC = [1] * 15
FRAME_SYNC_NORMAL = [0, 0, 0, 1, 0, 1, 1, 1, 1]
FRAME_SYNC_TEST = [0, 1, 1, 0, 1, 0, 0, 0, 0]

# Polynômes BCH du décodeur dec406 (coefficients MSB first)
# BCH-1: 22 coefficients pour 21 bits de parité
BCH1_POLY = [1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1]
# BCH-2: 13 coefficients pour 12 bits de parité  
BCH2_POLY = [1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1]


# =============================================================================
# BCH Encoder avec les polynômes exacts du décodeur
# =============================================================================

def compute_bch(data_bits: List[int], generator: List[int]) -> List[int]:
    """
    Calcule les bits de parité BCH par division polynomiale
    
    Méthode identique au décodeur dec406:
    - Division du message par le polynôme générateur
    - Le reste de la division = bits de parité
    """
    n_parity = len(generator) - 1  # Degré du polynôme = nb bits parité
    
    # Copier les données + espace pour parité
    dividend = list(data_bits) + [0] * n_parity
    
    # Division polynomiale (comme dans test_crc1/test_crc2)
    for i in range(len(data_bits)):
        if dividend[i] == 1:
            for j in range(len(generator)):
                dividend[i + j] ^= generator[j]
    
    # Les n_parity derniers bits sont le reste (parité)
    return dividend[-n_parity:]


def compute_bch1(pdf1_bits: List[int]) -> List[int]:
    """
    BCH-1: Protège PDF-1 (bits 24-84 = 61 bits)
    Génère 21 bits de parité (bits 85-105)
    """
    if len(pdf1_bits) != 61:
        raise ValueError(f"PDF-1 doit avoir 61 bits, reçu {len(pdf1_bits)}")
    return compute_bch(pdf1_bits, BCH1_POLY)


def compute_bch2(pdf2_bits: List[int]) -> List[int]:
    """
    BCH-2: Protège PDF-2 (bits 106-131 = 26 bits)
    Génère 12 bits de parité (bits 132-143)
    """
    if len(pdf2_bits) != 26:
        raise ValueError(f"PDF-2 doit avoir 26 bits, reçu {len(pdf2_bits)}")
    return compute_bch(pdf2_bits, BCH2_POLY)


# =============================================================================
# Parsing des coordonnées DMS
# =============================================================================

@dataclass
class DMSCoordinate:
    degrees: int
    minutes: int
    seconds: float
    direction: str
    
    def to_decimal(self) -> float:
        decimal = abs(self.degrees) + self.minutes / 60 + self.seconds / 3600
        if self.direction in ['S', 'W']:
            decimal = -decimal
        return decimal
    
    @classmethod
    def from_string(cls, dms_str: str) -> 'DMSCoordinate':
        dms_str = dms_str.strip().upper()
        patterns = [
            r"(\d+)[°D]\s*(\d+)[′'M]\s*([\d.]+)[″\"S]?\s*([NSEW])",
            r"(\d+)\s+(\d+)\s+([\d.]+)\s*([NSEW])",
            r"(\d+)[°D]?\s*(\d+)[′'M]?\s*([\d.]+)[″\"S]?\s*([NSEW])",
        ]
        for pattern in patterns:
            match = re.match(pattern, dms_str)
            if match:
                return cls(
                    degrees=int(match.group(1)),
                    minutes=int(match.group(2)),
                    seconds=float(match.group(3)),
                    direction=match.group(4)
                )
        raise ValueError(f"Format DMS non reconnu: {dms_str}")


def parse_coordinates(lat_str: str, lon_str: str) -> Tuple[float, float]:
    lat = DMSCoordinate.from_string(lat_str)
    lon = DMSCoordinate.from_string(lon_str)
    return lat.to_decimal(), lon.to_decimal()


# =============================================================================
# Construction du message Cospas-Sarsat
# =============================================================================

def build_beacon_message(
    latitude: float,
    longitude: float,
    country_code: int = 227,  # France
    protocol_code: int = 0b0010,  # Standard Location Protocol - EPIRB
    cert_num: int = 123,
    serial_num: int = 4567,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    Construit le message beacon (120 bits: PDF-1 + BCH-1 + PDF-2 + BCH-2)
    
    Structure (numérotation bits de la trame complète 0-143):
    - Bits 0-14: Bit sync (15 x 1)
    - Bits 15-23: Frame sync (9 bits)
    - Bits 24-84: PDF-1 (61 bits) - Format + Country + Protocol + Position coarse
    - Bits 85-105: BCH-1 (21 bits)
    - Bits 106-131: PDF-2 (26 bits) - Position fine
    - Bits 132-143: BCH-2 (12 bits)
    
    Encodage position selon dec406 de F4EHY:
    PDF-1:
      - Bit 64: N/S (0=N, 1=S)
      - Bits 65-71: Latitude degrés (7 bits, 0-90)
      - Bits 72-73: Latitude minutes / 15 (2 bits, 0-3 → 0,15,30,45 min)
      - Bit 74: E/W (0=E, 1=W)
      - Bits 75-82: Longitude degrés (8 bits, 0-180)
      - Bits 83-84: Longitude minutes / 15 (2 bits, 0-3 → 0,15,30,45 min)
    
    PDF-2:
      - Bits 106-109: Fixed "1101"
      - Bit 110: Position source (1=internal)
      - Bit 111: 121.5 MHz homing (1=yes)
      - Bit 112: Signe offset lat (1=positif, 0=négatif)
      - Bits 113-117: Offset minutes lat (5 bits, 0-14)
      - Bits 118-121: Offset secondes lat / 4 (4 bits, 0-15 → 0-60s)
      - Bit 122: Signe offset lon (1=positif, 0=négatif)
      - Bits 123-127: Offset minutes lon (5 bits, 0-14)
      - Bits 128-131: Offset secondes lon / 4 (4 bits, 0-15 → 0-60s)
    
    Returns:
        (pdf1, bch1, pdf2, bch2) - les 4 parties du message
    """
    
    # Conversion position en degrés/minutes/secondes
    lat_sign = 0 if latitude >= 0 else 1  # 0=N, 1=S
    lat_abs = abs(latitude)
    lat_deg = int(lat_abs)
    lat_min_total = (lat_abs - lat_deg) * 60
    lat_min = int(lat_min_total)
    lat_sec = (lat_min_total - lat_min) * 60
    
    lon_sign = 0 if longitude >= 0 else 1  # 0=E, 1=W
    lon_abs = abs(longitude)
    lon_deg = int(lon_abs)
    lon_min_total = (lon_abs - lon_deg) * 60
    lon_min = int(lon_min_total)
    lon_sec = (lon_min_total - lon_min) * 60
    
    # Calcul des valeurs grossières (PDF-1) et offsets (PDF-2)
    # Minutes grossières = multiple de 15
    lat_min_coarse = (lat_min // 15) * 15  # 0, 15, 30, ou 45
    lat_min_coarse_code = lat_min // 15    # 0, 1, 2, ou 3
    
    lon_min_coarse = (lon_min // 15) * 15
    lon_min_coarse_code = lon_min // 15
    
    # Offset = différence entre position réelle et position grossière
    lat_min_offset = lat_min - lat_min_coarse  # 0-14
    # Arrondi au multiple de 4 secondes (résolution du format)
    # On arrondit au plus proche, mais on limite à 15 max
    lat_sec_offset = min(round(lat_sec / 4), 15)  # 0-15
    
    lon_min_offset = lon_min - lon_min_coarse  # 0-14
    lon_sec_offset = min(round(lon_sec / 4), 15)  # 0-15
    
    # === PDF-1 (61 bits) ===
    # Indices relatifs dans PDF-1: 0-60
    # Indices dans trame: 24-84
    pdf1 = []
    
    # Bit 24 (idx 0): Format Flag = 1 (Long format with location)
    pdf1.append(1)
    
    # Bit 25 (idx 1): Protocol Flag = 0 (Standard Location Protocol)
    pdf1.append(0)
    
    # Bits 26-35 (idx 2-11): Country Code (10 bits)
    for i in range(9, -1, -1):
        pdf1.append((country_code >> i) & 1)
    
    # Bits 36-39 (idx 12-15): Protocol Code (4 bits)
    for i in range(3, -1, -1):
        pdf1.append((protocol_code >> i) & 1)
    
    # Bits 40-63 (idx 16-39): Identification data (24 bits)
    # Bits 40-49: C/S Type Approval Certificate (10 bits)
    cert_num = cert_num & 0x3FF
    for i in range(9, -1, -1):
        pdf1.append((cert_num >> i) & 1)
    
    # Bits 50-63: Serial Number (14 bits)
    serial_num = serial_num & 0x3FFF
    for i in range(13, -1, -1):
        pdf1.append((serial_num >> i) & 1)
    
    # Bit 64 (idx 40): Latitude N/S (0=N, 1=S)
    pdf1.append(lat_sign)
    
    # Bits 65-71 (idx 41-47): Latitude degrees (7 bits, 0-90)
    for i in range(6, -1, -1):
        pdf1.append((lat_deg >> i) & 1)
    
    # Bits 72-73 (idx 48-49): Latitude minutes coarse (2 bits, 0-3)
    for i in range(1, -1, -1):
        pdf1.append((lat_min_coarse_code >> i) & 1)
    
    # Bit 74 (idx 50): Longitude E/W (0=E, 1=W)
    pdf1.append(lon_sign)
    
    # Bits 75-82 (idx 51-58): Longitude degrees (8 bits, 0-180)
    for i in range(7, -1, -1):
        pdf1.append((lon_deg >> i) & 1)
    
    # Bits 83-84 (idx 59-60): Longitude minutes coarse (2 bits, 0-3)
    for i in range(1, -1, -1):
        pdf1.append((lon_min_coarse_code >> i) & 1)
    
    assert len(pdf1) == 61, f"PDF-1 should be 61 bits, got {len(pdf1)}"
    
    # === BCH-1 (21 bits) ===
    bch1 = compute_bch1(pdf1)
    
    # === PDF-2 (26 bits) ===
    # Indices relatifs dans PDF-2: 0-25
    # Indices dans trame: 106-131
    pdf2 = []
    
    # Bits 106-109 (idx 0-3): Fixed "1101"
    pdf2.extend([1, 1, 0, 1])
    
    # Bit 110 (idx 4): Position source (1=internal GNSS)
    pdf2.append(1)
    
    # Bit 111 (idx 5): 121.5 MHz homing (1=yes)
    pdf2.append(1)
    
    # Bit 112 (idx 6): Signe offset latitude (1=positif, 0=négatif)
    pdf2.append(1)
    
    # Bits 113-117 (idx 7-11): Offset minutes latitude (5 bits, 0-14)
    for i in range(4, -1, -1):
        pdf2.append((lat_min_offset >> i) & 1)
    
    # Bits 118-121 (idx 12-15): Offset secondes latitude / 4 (4 bits, 0-15)
    for i in range(3, -1, -1):
        pdf2.append((lat_sec_offset >> i) & 1)
    
    # Bit 122 (idx 16): Signe offset longitude (1=positif, 0=négatif)
    pdf2.append(1)
    
    # Bits 123-127 (idx 17-21): Offset minutes longitude (5 bits, 0-14)
    for i in range(4, -1, -1):
        pdf2.append((lon_min_offset >> i) & 1)
    
    # Bits 128-131 (idx 22-25): Offset secondes longitude / 4 (4 bits, 0-15)
    for i in range(3, -1, -1):
        pdf2.append((lon_sec_offset >> i) & 1)
    
    assert len(pdf2) == 26, f"PDF-2 should be 26 bits, got {len(pdf2)}"
    
    # === BCH-2 (12 bits) ===
    bch2 = compute_bch2(pdf2)
    
    # Debug: afficher les valeurs encodées
    print(f"\n📐 Encodage position:")
    print(f"   Entrée: {latitude:.6f}°, {longitude:.6f}°")
    print(f"   Lat: {lat_deg}° {lat_min}' {lat_sec:.1f}\" {'N' if lat_sign == 0 else 'S'}")
    print(f"   Lon: {lon_deg}° {lon_min}' {lon_sec:.1f}\" {'E' if lon_sign == 0 else 'W'}")
    print(f"   PDF-1 coarse: {lat_deg}° {lat_min_coarse}' / {lon_deg}° {lon_min_coarse}'")
    print(f"   PDF-2 offset: +{lat_min_offset}' {lat_sec_offset*4}\" / +{lon_min_offset}' {lon_sec_offset*4}\"")
    
    return pdf1, bch1, pdf2, bch2


def build_complete_frame(
    pdf1: List[int], 
    bch1: List[int], 
    pdf2: List[int], 
    bch2: List[int],
    test_mode: bool = False
) -> List[int]:
    """
    Assemble la trame complète (144 bits)
    """
    frame = []
    
    # Bits 0-14: Bit sync (15 x 1)
    frame.extend(BIT_SYNC)
    
    # Bits 15-23: Frame sync (9 bits)
    frame.extend(FRAME_SYNC_TEST if test_mode else FRAME_SYNC_NORMAL)
    
    # Bits 24-84: PDF-1 (61 bits)
    frame.extend(pdf1)
    
    # Bits 85-105: BCH-1 (21 bits)
    frame.extend(bch1)
    
    # Bits 106-131: PDF-2 (26 bits)
    frame.extend(pdf2)
    
    # Bits 132-143: BCH-2 (12 bits)
    frame.extend(bch2)
    
    assert len(frame) == 144, f"Frame should be 144 bits, got {len(frame)}"
    
    return frame


# =============================================================================
# Génération du signal audio
# =============================================================================

def generate_biphase_l_signal(
    bits: List[int],
    sample_rate: int,
    amplitude: float = 0.8,
    unmodulated_duration: float = 0.160
) -> np.ndarray:
    """
    Génère un signal audio Biphase-L (Manchester) pour le décodeur dec406
    
    Biphase-L encoding:
    - Bit "1": niveau haut -> niveau bas (transition descendante au milieu)
    - Bit "0": niveau bas -> niveau haut (transition montante au milieu)
    
    Le décodeur dec406 utilise l'autocorrélation pour détecter les transitions.
    Quand le signal est corrélé avec lui-même décalé d'1 bit:
    - Même valeur = corrélation positive (front montant dans Y1)
    - Valeur différente = corrélation négative (front descendant dans Y1)
    """
    samples_per_bit = sample_rate // SYMBOL_RATE
    half_bit = samples_per_bit // 2
    
    # 1. Porteuse non modulée (160 ms) - niveau constant
    n_carrier = int(unmodulated_duration * sample_rate)
    carrier = np.ones(n_carrier, dtype=np.float32) * amplitude
    
    # 2. Signal Biphase-L modulé
    modulated = []
    
    for bit in bits:
        if bit == 1:
            # Bit 1: haut puis bas
            modulated.extend([amplitude] * half_bit)
            modulated.extend([-amplitude] * half_bit)
        else:
            # Bit 0: bas puis haut
            modulated.extend([-amplitude] * half_bit)
            modulated.extend([amplitude] * half_bit)
    
    modulated = np.array(modulated, dtype=np.float32)
    
    # Combiner
    signal = np.concatenate([carrier, modulated])
    
    return signal


def save_wav_16bit(filename: str, sample_rate: int, signal: np.ndarray):
    """Sauvegarde en WAV 16 bits mono (format attendu par dec406)"""
    # Normaliser et convertir en int16
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val * 0.9
    
    signal_int16 = (signal * 32767).astype(np.int16)
    wavfile.write(filename, sample_rate, signal_int16)


def generate_iq_hackrf(
    bits: List[int],
    sample_rate: int = 2_000_000,
    unmodulated_duration: float = 0.160,
    freq_deviation: float = 800.0
) -> np.ndarray:
    """
    Génère un signal IQ FM pour HackRF
    
    Le décodeur dec406 attend un signal audio Biphase-L en sortie de démodulation NFM.
    Donc on doit générer un signal FM dont la fréquence instantanée représente
    le signal Biphase-L (Manchester).
    
    Biphase-L encoding (dans le domaine audio après démod FM):
    - Bit "1": niveau +1 puis niveau -1 (transition au milieu)
    - Bit "0": niveau -1 puis niveau +1 (transition au milieu)
    
    En FM, cela se traduit par:
    - Niveau +1 -> fréquence porteuse + deviation
    - Niveau -1 -> fréquence porteuse - deviation
    
    Args:
        bits: Liste des bits à moduler
        sample_rate: Taux d'échantillonnage (défaut 2 MHz pour HackRF)
        unmodulated_duration: Durée porteuse non modulée (160 ms)
        freq_deviation: Déviation de fréquence en Hz (±800 Hz typique pour NFM)
    
    Returns:
        Tuple (I, Q) en float32
    """
    samples_per_bit = sample_rate // SYMBOL_RATE
    half_bit = samples_per_bit // 2
    
    # 1. Construire le signal en bande de base (ce que le démodulateur FM doit sortir)
    # C'est le signal Biphase-L
    
    # Porteuse non modulée = fréquence constante = niveau 0 en bande de base
    n_carrier = int(unmodulated_duration * sample_rate)
    baseband_carrier = np.zeros(n_carrier, dtype=np.float32)
    
    # Signal Biphase-L modulé
    baseband_modulated = []
    
    for bit in bits:
        if bit == 1:
            # Bit 1: haut (+1) puis bas (-1)
            baseband_modulated.extend([1.0] * half_bit)
            baseband_modulated.extend([-1.0] * half_bit)
        else:
            # Bit 0: bas (-1) puis haut (+1)
            baseband_modulated.extend([-1.0] * half_bit)
            baseband_modulated.extend([1.0] * half_bit)
    
    baseband_modulated = np.array(baseband_modulated, dtype=np.float32)
    
    # Signal bande de base complet
    baseband = np.concatenate([baseband_carrier, baseband_modulated])
    
    # 2. Modulation FM: la fréquence instantanée est proportionnelle au signal bande de base
    # f(t) = f_carrier + freq_deviation * baseband(t)
    # La phase est l'intégrale de la fréquence:
    # phi(t) = 2*pi * integral(f(t)) = 2*pi * integral(freq_deviation * baseband(t))
    
    # Intégrale cumulative du signal bande de base
    # Multiplier par freq_deviation et par dt (1/sample_rate)
    phase = 2 * np.pi * freq_deviation * np.cumsum(baseband) / sample_rate
    
    # 3. Générer le signal IQ
    I = np.cos(phase).astype(np.float32)
    Q = np.sin(phase).astype(np.float32)
    
    return I, Q


def save_iq_hackrf(filename: str, I: np.ndarray, Q: np.ndarray, format_type: str = 'int8'):
    """
    Sauvegarde le fichier IQ pour HackRF
    
    Formats supportés:
    - int8: Format natif HackRF (hackrf_transfer -t)
    - int16: Format pour GNU Radio / SDR++
    - float32: Format pour analyse
    """
    # Normaliser
    max_val = max(np.max(np.abs(I)), np.max(np.abs(Q)))
    if max_val > 0:
        I = I / max_val
        Q = Q / max_val
    
    if format_type == 'int8':
        # Format HackRF natif: I/Q entrelacés en int8 [-127, 127]
        iq = np.empty(len(I) * 2, dtype=np.int8)
        iq[0::2] = (I * 127).astype(np.int8)
        iq[1::2] = (Q * 127).astype(np.int8)
        iq.tofile(filename)
        
    elif format_type == 'int16':
        # Format int16: I/Q entrelacés
        iq = np.empty(len(I) * 2, dtype=np.int16)
        iq[0::2] = (I * 32767).astype(np.int16)
        iq[1::2] = (Q * 32767).astype(np.int16)
        iq.tofile(filename)
        
    elif format_type == 'float32':
        # Format float32: I/Q entrelacés
        iq = np.empty(len(I) * 2, dtype=np.float32)
        iq[0::2] = I.astype(np.float32)
        iq[1::2] = Q.astype(np.float32)
        iq.tofile(filename)


def verify_bch(pdf1: List[int], bch1: List[int], pdf2: List[int], bch2: List[int]) -> bool:
    """Vérifie les BCH en simulant le décodeur"""
    
    # Test CRC1 comme dans dec406
    message1 = pdf1 + bch1  # 61 + 21 = 82 bits
    div = list(message1[:22])
    i = 0
    
    while i < 61:  # Jusqu'au bit 84 (relatif: 60)
        # XOR avec générateur
        for j in range(22):
            div[j] ^= BCH1_POLY[j]
        
        # Décaler tant que MSB = 0
        while div[0] == 0 and i < 61:
            div = div[1:] + [message1[22 + i] if (22 + i) < len(message1) else 0]
            i += 1
    
    crc1_ok = sum(div) == 0
    
    # Test CRC2 comme dans dec406
    message2 = pdf2 + bch2  # 26 + 12 = 38 bits
    div = list(message2[:13])
    i = 0
    
    while i < 26:
        for j in range(13):
            div[j] ^= BCH2_POLY[j]
        
        while div[0] == 0 and i < 26:
            div = div[1:] + [message2[13 + i] if (13 + i) < len(message2) else 0]
            i += 1
    
    crc2_ok = sum(div) == 0
    
    return crc1_ok and crc2_ok


def print_frame_info(frame: List[int], pdf1: List[int], bch1: List[int], 
                     pdf2: List[int], bch2: List[int],
                     latitude: float, longitude: float, test_mode: bool):
    """Affiche les informations détaillées"""
    print("\n" + "=" * 70)
    print("TRAME COSPAS-SARSAT 406 MHz - Compatible dec406")
    print("=" * 70)
    
    print(f"\nMode: {'TEST (self-test)' if test_mode else 'NORMAL'}")
    print(f"Position: {latitude:+.6f}°, {longitude:+.6f}°")
    
    print(f"\nStructure de la trame ({len(frame)} bits = {len(frame)/SYMBOL_RATE*1000:.0f} ms):")
    print(f"  Bits 0-14   : Bit sync     = {''.join(str(b) for b in frame[0:15])}")
    print(f"  Bits 15-23  : Frame sync   = {''.join(str(b) for b in frame[15:24])}")
    print(f"  Bits 24-84  : PDF-1        = {''.join(str(b) for b in frame[24:85])}")
    print(f"  Bits 85-105 : BCH-1        = {''.join(str(b) for b in frame[85:106])}")
    print(f"  Bits 106-131: PDF-2        = {''.join(str(b) for b in frame[106:132])}")
    print(f"  Bits 132-143: BCH-2        = {''.join(str(b) for b in frame[132:144])}")
    
    # Vérification BCH
    bch_ok = verify_bch(pdf1, bch1, pdf2, bch2)
    print(f"\nVérification BCH: {'✅ OK' if bch_ok else '❌ ERREUR'}")
    
    # Hex dump du message (bits 24-143)
    hex_str = ""
    for i in range(24, 144, 8):
        byte = 0
        for j in range(8):
            if i + j < 144:
                byte = (byte << 1) | frame[i + j]
        hex_str += f"{byte:02x}"
    print(f"\nMessage hex (bits 24-143): {hex_str}")
    
    # Trame complète
    print(f"\nTrame complète (144 bits):")
    print(''.join(str(b) for b in frame))


def main():
    parser = argparse.ArgumentParser(
        description="""
Générateur Cospas-Sarsat 406 MHz pour dec406 et HackRF

Formats de sortie:
  --format wav  : Fichier WAV 16 bits mono pour test avec dec406
  --format iq   : Fichier IQ int8 pour HackRF (hackrf_transfer -t)

Exemple WAV (test décodeur):
  python3 cospas_generator_dec406.py --lat "43°45'12\"N" --lon "7°25'30\"E" -o test.wav

Exemple IQ (HackRF):
  python3 cospas_generator_dec406.py --lat "43°45'12\"N" --lon "7°25'30\"E" -o test.iq --format iq
  hackrf_transfer -t test.iq -f 406025000 -s 2000000 -a 0 -x 0
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--lat', required=True, help='Latitude (ex: 43°45\'12.5"N)')
    parser.add_argument('--lon', required=True, help='Longitude (ex: 7°25\'30.2"E)')
    parser.add_argument('-o', '--output', default='beacon_406.wav')
    parser.add_argument('--format', choices=['wav', 'iq'], default='wav',
                        help='Format de sortie: wav ou iq (défaut: wav)')
    parser.add_argument('--sample-rate', type=int, default=None,
                        help='Sample rate (défaut: 48000 pour WAV, 2000000 pour IQ)')
    parser.add_argument('--iq-format', choices=['int8', 'int16', 'float32'], default='int8',
                        help='Format IQ: int8 (HackRF), int16, float32 (défaut: int8)')
    parser.add_argument('--country', type=int, default=227,
                        help='Code pays MID (défaut: 227 = France)')
    parser.add_argument('--cert', type=int, default=123,
                        help='Numéro certificat (défaut: 123)')
    parser.add_argument('--serial', type=int, default=4567,
                        help='Numéro série (défaut: 4567)')
    parser.add_argument('--repeat', type=int, default=1,
                        help='Nombre de répétitions')
    parser.add_argument('--test-mode', action='store_true',
                        help='Utiliser frame sync de test')
    parser.add_argument('--pause', type=float, default=0.5,
                        help='Pause entre répétitions (secondes)')
    parser.add_argument('--freq-deviation', type=float, default=800.0,
                        help='Déviation de fréquence FM en Hz (défaut: 800 Hz)')
    
    args = parser.parse_args()
    
    # Sample rate par défaut selon le format
    if args.sample_rate is None:
        args.sample_rate = 48000 if args.format == 'wav' else 2_000_000
    
    try:
        latitude, longitude = parse_coordinates(args.lat, args.lon)
    except ValueError as e:
        print(f"Erreur de coordonnées: {e}")
        sys.exit(1)
    
    print(f"\n📍 Position: {latitude:.6f}°, {longitude:.6f}°")
    
    # Construire le message
    pdf1, bch1, pdf2, bch2 = build_beacon_message(
        latitude=latitude,
        longitude=longitude,
        country_code=args.country,
        cert_num=args.cert,
        serial_num=args.serial
    )
    
    # Assembler la trame
    frame = build_complete_frame(pdf1, bch1, pdf2, bch2, test_mode=args.test_mode)
    
    # Afficher les infos
    print_frame_info(frame, pdf1, bch1, pdf2, bch2, latitude, longitude, args.test_mode)
    
    # Générer le signal selon le format
    print(f"\n🔧 Génération du signal ({args.format.upper()})...")
    
    if args.format == 'wav':
        # Signal audio Biphase-L pour dec406
        signal = generate_biphase_l_signal(
            frame,
            args.sample_rate,
            amplitude=0.8,
            unmodulated_duration=0.160
        )
        
        # Répétitions
        if args.repeat > 1:
            pause_samples = np.zeros(int(args.pause * args.sample_rate), dtype=np.float32)
            signals = [signal]
            for _ in range(args.repeat - 1):
                signals.append(pause_samples)
                signals.append(signal)
            signal = np.concatenate(signals)
        
        # Sauvegarder WAV
        save_wav_16bit(args.output, args.sample_rate, signal)
        
        duration = len(signal) / args.sample_rate
        file_size = len(signal) * 2  # 16 bits = 2 bytes
        
        print(f"\n✅ Fichier WAV généré: {args.output}")
        print(f"   Format: WAV 16 bits mono")
        print(f"   Sample rate: {args.sample_rate} Hz")
        print(f"   Durée: {duration:.2f} s")
        print(f"   Taille: {file_size / 1024:.1f} KB")
        
        print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  Test avec dec406:                                                   ║
║    ./dec406 {args.output:<54} ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    else:  # format == 'iq'
        # Signal IQ FM pour HackRF
        I, Q = generate_iq_hackrf(
            frame,
            args.sample_rate,
            unmodulated_duration=0.160,
            freq_deviation=args.freq_deviation
        )
        
        # Répétitions
        if args.repeat > 1:
            pause_samples = int(args.pause * args.sample_rate)
            I_pause = np.zeros(pause_samples, dtype=np.float32)
            Q_pause = np.zeros(pause_samples, dtype=np.float32)
            
            I_list, Q_list = [I], [Q]
            for _ in range(args.repeat - 1):
                I_list.extend([I_pause, I])
                Q_list.extend([Q_pause, Q])
            I = np.concatenate(I_list)
            Q = np.concatenate(Q_list)
        
        # Sauvegarder IQ
        save_iq_hackrf(args.output, I, Q, args.iq_format)
        
        duration = len(I) / args.sample_rate
        bytes_per_sample = {'int8': 1, 'int16': 2, 'float32': 4}[args.iq_format]
        file_size = len(I) * 2 * bytes_per_sample  # I + Q
        
        print(f"\n✅ Fichier IQ généré: {args.output}")
        print(f"   Format: {args.iq_format} (I/Q entrelacés)")
        print(f"   Sample rate: {args.sample_rate / 1e6:.1f} MHz")
        print(f"   Durée: {duration:.3f} s")
        print(f"   Taille: {file_size / 1024:.1f} KB")
        print(f"   Répétitions: {args.repeat}")
        
        print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  ⚠️  AVERTISSEMENT - INJECTION COAXIALE UNIQUEMENT                   ║
║                                                                      ║
║  Transmission avec HackRF (VIA CÂBLE COAXIAL + ATTÉNUATEURS):        ║
║    hackrf_transfer -t {args.output} -f 406025000 -s {args.sample_rate} -a 0 -x 0          ║
║                                                                      ║
║  NE PAS ÉMETTRE EN RF SANS COORDINATION FMCC/CNES                    ║
║  Une émission non autorisée sur 406 MHz déclenche une alerte SAR     ║
╚══════════════════════════════════════════════════════════════════════╝
""")


if __name__ == '__main__':
    main()
