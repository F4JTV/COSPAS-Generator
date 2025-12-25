# Générateur de signaux Cospas-Sarsat 406 MHz

Outil de génération de signaux de balises de détresse 406 MHz pour tests et exercices ADRASEC.

⚠️ **AVERTISSEMENT** : Cet outil est destiné uniquement aux tests en environnement contrôlé (injection coaxiale). Toute émission RF non autorisée sur 406 MHz déclenche une alerte SAR réelle et constitue une infraction pénale.

## Table des matières

1. [Présentation](#présentation)
2. [Installation](#installation)
3. [Utilisation](#utilisation)
4. [Format des trames Cospas-Sarsat](#format-des-trames-cospas-sarsat)
5. [Encodage de la position](#encodage-de-la-position)
6. [Codes BCH](#codes-bch)
7. [Modulation du signal](#modulation-du-signal)
8. [Compatibilité](#compatibilité)
9. [Références](#références)

---

## Présentation

Ce générateur Python produit des signaux de balises de détresse conformes au standard Cospas-Sarsat T.001. Il permet de :

- Générer des fichiers **WAV** pour tester des décodeurs logiciels (dec406 de F4EHY)
- Générer des fichiers **IQ** pour transmission via HackRF (tests par injection coaxiale)
- Encoder des positions GPS avec les coordonnées souhaitées
- Configurer les identifiants de balise (pays, certificat, numéro de série)

### Auteurs et crédits

- **Générateur** : Développé pour ADRASEC 06 (F4JTV)
- **Décodeur dec406** : F4EHY
- **Décodeur Arduino** : F1LVT

---

## Installation

### Prérequis

```bash
# Python 3.x avec les bibliothèques suivantes
pip install numpy scipy
```

---

## Utilisation

### Génération d'un fichier WAV (test décodeur)

```bash
python3 cospas_generator_dec406.py \
    --lat "43°45'12\"N" \
    --lon "7°25'30\"E" \
    -o test.wav \
    --format wav \
    --repeat 3
```

### Génération d'un fichier IQ (HackRF)

```bash
python3 cospas_generator_dec406.py \
    --lat "43°45'12\"N" \
    --lon "7°25'30\"E" \
    -o test.iq \
    --format iq \
    --freq-deviation 2000 \
    --repeat 3
```

Transmission via HackRF (⚠️ **INJECTION COAXIALE UNIQUEMENT**) :
```bash
hackrf_transfer -t test.iq -f 406025000 -s 2000000 -a 0 -x 0
```

### Options disponibles

| Option | Description | Défaut |
|--------|-------------|--------|
| `--lat` | Latitude (ex: `43°45'12"N`) | Requis |
| `--lon` | Longitude (ex: `7°25'30"E`) | Requis |
| `-o, --output` | Fichier de sortie | `beacon_406.wav` |
| `--format` | Format : `wav` ou `iq` | `wav` |
| `--sample-rate` | Taux d'échantillonnage | 48000 (WAV) / 2000000 (IQ) |
| `--freq-deviation` | Déviation FM en Hz (mode IQ) | 2000 |
| `--country` | Code pays MID | 227 (France) |
| `--cert` | Numéro de certificat | 123 |
| `--serial` | Numéro de série | 4567 |
| `--repeat` | Nombre de répétitions | 1 |
| `--test-mode` | Utiliser le frame sync de test | Non |
| `--pause` | Pause entre répétitions (s) | 0.5 |

---

## Format des trames Cospas-Sarsat

### Vue d'ensemble

Une trame Cospas-Sarsat 406 MHz "Long Format" comprend **144 bits** transmis à **400 bps**, soit une durée de **360 ms** pour la partie données, précédée de **160 ms** de porteuse non modulée.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TRAME COMPLÈTE (144 bits)                            │
├─────────┬───────────┬─────────┬─────────┬─────────┬─────────┬───────────┤
│ Bit     │ Bit       │ PDF-1   │ BCH-1   │ PDF-2   │ BCH-2   │           │
│ Sync    │ Frame     │         │         │         │         │           │
│ (15)    │ Sync (9)  │ (61)    │ (21)    │ (26)    │ (12)    │           │
├─────────┼───────────┼─────────┼─────────┼─────────┼─────────┼───────────┤
│ 0-14    │ 15-23     │ 24-84   │ 85-105  │ 106-131 │ 132-143 │           │
└─────────┴───────────┴─────────┴─────────┴─────────┴─────────┴───────────┘
```

### Détail de chaque champ

#### 1. Bit Sync (bits 0-14) - 15 bits

Séquence de synchronisation bit : **15 bits à "1"**

```
111111111111111
```

Cette séquence permet au récepteur de :
- Détecter le début d'une trame
- Synchroniser son horloge de récupération de bits
- Calibrer les seuils de décision

#### 2. Frame Sync (bits 15-23) - 9 bits

Mot de synchronisation trame identifiant le type de message :

| Mode | Séquence | Description |
|------|----------|-------------|
| Normal | `000101111` | Trame opérationnelle (traitée par satellites) |
| Test | `011010000` | Trame de test/exercice (ignorée par satellites) |

#### 3. PDF-1 - Protected Data Field 1 (bits 24-84) - 61 bits

Contient les informations d'identification et la position grossière.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PDF-1 (61 bits)                                 │
├────────┬──────────┬────────────┬──────────┬────────────────────────────┤
│ Format │ Protocol │ Country    │ Protocol │ Identification    │Position│
│ Flag   │ Flag     │ Code       │ Code     │ Data              │ Data   │
│ (1)    │ (1)      │ (10)       │ (4)      │ (24)              │ (21)   │
├────────┼──────────┼────────────┼──────────┼────────────────────────────┤
│ Bit 24 │ Bit 25   │ Bits 26-35 │Bits 36-39│ Bits 40-63        │64-84   │
└────────┴──────────┴────────────┴──────────┴────────────────────────────┘
```

##### Bit 24 : Format Flag
- `1` = Format long (144 bits) avec position encodée
- `0` = Format court (112 bits) sans position

##### Bit 25 : Protocol Flag
- `0` = Standard Location Protocol
- `1` = User Location Protocol

##### Bits 26-35 : Country Code (10 bits)
Code MID (Maritime Identification Digits) du pays d'immatriculation.

| Code | Pays |
|------|------|
| 226 | Roumanie |
| 227 | **France** |
| 228 | Monaco |
| 230 | Espagne |
| 244 | Pays-Bas |

##### Bits 36-39 : Protocol Code (4 bits)

| Code | Type de balise |
|------|----------------|
| 0010 | EPIRB - MMSI |
| 0110 | EPIRB - Radio Call Sign |
| 0111 | EPIRB - Serial |
| 1100 | ELT - Serial |
| 1000 | PLB - Serial |

##### Bits 40-63 : Identification Data (24 bits)
Dépend du Protocol Code. Pour le mode Serial :
- **Bits 40-49** (10 bits) : Numéro de certificat d'approbation
- **Bits 50-63** (14 bits) : Numéro de série

##### Bits 64-84 : Position Data (21 bits)
Position grossière avec résolution de 15 minutes d'arc (~28 km).

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Position grossière (21 bits)                      │
├────────┬───────────────┬──────────┬────────┬───────────────┬────────┤
│ N/S    │ Lat Degrés    │ Lat Min  │ E/W    │ Lon Degrés    │Lon Min │
│ (1)    │ (7 bits)      │ /15 (2)  │ (1)    │ (8 bits)      │/15 (2) │
├────────┼───────────────┼──────────┼────────┼───────────────┼────────┤
│ Bit 64 │ Bits 65-71    │ 72-73    │ Bit 74 │ Bits 75-82    │ 83-84  │
└────────┴───────────────┴──────────┴────────┴───────────────┴────────┘
```

| Champ | Bits | Valeurs | Description |
|-------|------|---------|-------------|
| N/S | 64 | 0=Nord, 1=Sud | Hémisphère latitude |
| Lat Degrés | 65-71 | 0-90 | Degrés de latitude |
| Lat Min/15 | 72-73 | 0-3 | Minutes ÷ 15 (0, 15, 30, 45) |
| E/W | 74 | 0=Est, 1=Ouest | Hémisphère longitude |
| Lon Degrés | 75-82 | 0-180 | Degrés de longitude |
| Lon Min/15 | 83-84 | 0-3 | Minutes ÷ 15 (0, 15, 30, 45) |

#### 4. BCH-1 (bits 85-105) - 21 bits

Code correcteur d'erreurs protégeant PDF-1.

#### 5. PDF-2 - Protected Data Field 2 (bits 106-131) - 26 bits

Contient la position fine (offset par rapport à la position grossière).

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PDF-2 (26 bits)                                 │
├────────┬────────┬────────┬────────┬──────────┬──────────┬──────────────┤
│ Fixed  │ Source │ Homing │ Lat    │ Lat      │ Lat      │ Longitude    │
│ 1101   │        │ 121.5  │ Sign   │ Min Off  │ Sec Off  │ Offset       │
│ (4)    │ (1)    │ (1)    │ (1)    │ (5)      │ (4)      │ (10)         │
├────────┼────────┼────────┼────────┼──────────┼──────────┼──────────────┤
│ 106-109│ 110    │ 111    │ 112    │ 113-117  │ 118-121  │ 122-131      │
└────────┴────────┴────────┴────────┴──────────┴──────────┴──────────────┘
```

##### Bits 106-109 : Fixed Pattern
Toujours `1101` pour indiquer la présence de données de position.

##### Bit 110 : Position Source
- `0` = Source externe (GPS externe)
- `1` = Source interne (GPS intégré)

##### Bit 111 : 121.5 MHz Homing
- `0` = Pas de signal 121.5 MHz
- `1` = Signal de ralliement 121.5 MHz actif

##### Bit 112 : Signe offset latitude
- `1` = Offset positif (ajouter à la position grossière)
- `0` = Offset négatif (soustraire, avec logique complémentaire)

##### Bits 113-117 : Offset minutes latitude (5 bits)
Valeur 0-14 à ajouter aux minutes grossières.

##### Bits 118-121 : Offset secondes latitude (4 bits)
Valeur 0-15, multipliée par 4 pour obtenir les secondes (résolution 4").

##### Bit 122 : Signe offset longitude
Identique au bit 112 pour la longitude.

##### Bits 123-127 : Offset minutes longitude (5 bits)
Valeur 0-14 à ajouter aux minutes grossières.

##### Bits 128-131 : Offset secondes longitude (4 bits)
Valeur 0-15, multipliée par 4 pour obtenir les secondes.

#### 6. BCH-2 (bits 132-143) - 12 bits

Code correcteur d'erreurs protégeant PDF-2.

---

## Encodage de la position

### Algorithme d'encodage

Pour encoder une position (exemple : **43°45'12"N, 7°25'30"E**) :

#### Étape 1 : Conversion en composantes

```
Latitude  : 43° 45' 12" N
Longitude :  7° 25' 30" E
```

#### Étape 2 : Calcul de la position grossière (PDF-1)

Les minutes sont arrondies au multiple de 15 inférieur :

```
Lat grossière : 43° 45' (45 ÷ 15 = 3 → code = 3)
Lon grossière :  7° 15' (25 ÷ 15 = 1 → code = 1)
```

#### Étape 3 : Calcul des offsets (PDF-2)

```
Offset lat minutes : 45 - 45 = 0'
Offset lat secondes : 12" ÷ 4 = 3 → 3 × 4 = 12"

Offset lon minutes : 25 - 15 = 10'
Offset lon secondes : 30" ÷ 4 = 7.5 → arrondi à 8 → 8 × 4 = 32"
```

#### Étape 4 : Position reconstruite

```
Latitude  : 43° + 45' + 0' + 12" = 43° 45' 12" ✓
Longitude :  7° + 15' + 10' + 32" = 7° 25' 32" (écart de 2")
```

### Résolution et précision

| Composante | Résolution | Précision |
|------------|------------|-----------|
| Degrés | 1° | ~111 km |
| Minutes grossières | 15' | ~28 km |
| Minutes fines | 1' | ~1.85 km |
| Secondes | 4" | ~120 m |

**Précision maximale** : ±2 secondes d'arc ≈ **±60 mètres**

### Cas particulier : valeurs non multiples de 4

Les secondes ne peuvent être encodées qu'en multiples de 4. Les valeurs intermédiaires sont arrondies :

| Secondes réelles | Encodage | Secondes décodées | Écart |
|------------------|----------|-------------------|-------|
| 0" | 0 | 0" | 0" |
| 2" | 1 | 4" | +2" |
| 4" | 1 | 4" | 0" |
| 6" | 2 | 8" | +2" |
| 30" | 8 | 32" | +2" |
| 58" | 15 | 60" | +2" |

---

## Codes BCH

### Principe

Les codes BCH (Bose-Chaudhuri-Hocquenghem) sont des codes correcteurs d'erreurs cycliques. Ils permettent de détecter et corriger des erreurs de transmission.

### BCH-1 : Protection de PDF-1

- **Type** : BCH(82, 61) raccourci de BCH(127, 106)
- **Capacité de correction** : t = 3 erreurs
- **Bits de parité** : 21 bits

**Polynôme générateur** (22 coefficients) :
```
g(x) = x²¹ + x²⁰ + x¹⁷ + x¹⁶ + x¹⁵ + x¹⁴ + x¹¹ + x⁹ + x⁸ + x⁶ + x⁵ + x³ + x² + 1

En binaire : 1001101101100111100011
```

### BCH-2 : Protection de PDF-2

- **Type** : BCH(38, 26) raccourci de BCH(63, 51)
- **Capacité de correction** : t = 2 erreurs
- **Bits de parité** : 12 bits

**Polynôme générateur** (13 coefficients) :
```
g(x) = x¹² + x¹⁰ + x⁸ + x⁵ + x⁴ + x³ + 1

En binaire : 1010100111001
```

### Algorithme de calcul

Le calcul des bits BCH s'effectue par division polynomiale :

```python
def compute_bch(data_bits, generator):
    """
    Division polynomiale pour calcul BCH
    Le reste de la division = bits de parité
    """
    n_parity = len(generator) - 1
    dividend = data_bits + [0] * n_parity
    
    for i in range(len(data_bits)):
        if dividend[i] == 1:
            for j in range(len(generator)):
                dividend[i + j] ^= generator[j]
    
    return dividend[-n_parity:]
```

### Vérification

À la réception, le décodeur effectue la même division. Si le reste est nul, la trame est valide. Sinon, le syndrome permet de localiser et corriger les erreurs.

---

## Modulation du signal

### Caractéristiques RF

| Paramètre | Valeur |
|-----------|--------|
| Fréquence porteuse | 406.025 MHz |
| Débit symbole | 400 bps |
| Modulation | PM/BPSK avec codage Biphase-L |
| Déviation de phase | ±1.1 rad |
| Durée porteuse | 160 ms |
| Durée données | 360 ms |
| Durée totale burst | 520 ms |

### Codage Biphase-L (Manchester)

Chaque bit est représenté par une transition au milieu du temps bit :

```
Bit "1" : Niveau haut → Niveau bas (transition descendante)
Bit "0" : Niveau bas → Niveau haut (transition montante)

    Bit 1         Bit 0         Bit 1         Bit 1
  ┌───────┐     ┌───────┐     ┌───────┐     ┌───────┐
  │   ┌───┘     │       │     │   ┌───┘     │   ┌───┘
  │   │         │   ┌───┘     │   │         │   │
──┘   └─────────┘   └─────────┘   └─────────┘   └─────
```

Ce codage garantit :
- Au moins une transition par bit (synchronisation)
- Pas de composante continue
- Détection facile par autocorrélation

### Génération du signal IQ

Pour la transmission via HackRF, le signal est modulé en FM :

```python
# Signal bande de base Biphase-L : +1 ou -1
# Modulation FM : fréquence instantanée = f0 + deviation × baseband
# Phase = intégrale de la fréquence

phase = 2π × freq_deviation × ∫baseband(t) dt

I = cos(phase)
Q = sin(phase)
```

Après démodulation NFM par le récepteur, on retrouve le signal Biphase-L original.

---

## Compatibilité

### Décodeurs testés

| Décodeur | Auteur | Format d'entrée | Status |
|----------|--------|-----------------|--------|
| dec406 | F4EHY | WAV 16-bit mono | ✅ Compatible |
| Arduino | F1LVT | RF via récepteur | ✅ Compatible |
| SDR++ | - | IQ via HackRF | ✅ Compatible |

### Paramètres recommandés

**Pour dec406 (WAV)** :
```bash
--format wav --sample-rate 48000
```

**Pour HackRF (IQ)** :
```bash
--format iq --sample-rate 2000000 --freq-deviation 2000
```

---

## Références

### Documents officiels

- **T.001** : Specification for Cospas-Sarsat 406 MHz Distress Beacons
- **T.018** : Specification for Cospas-Sarsat 406 MHz EPIRB
- **C/S A.001** : Cospas-Sarsat System Overview

### Liens utiles

- [Cospas-Sarsat Official](https://www.cospas-sarsat.int/)
- [ANFR - Balises de détresse](https://www.anfr.fr/)
- [FMCC France](https://www.cnes.fr/)

### Code source des décodeurs

- **dec406** : F4EHY - Décodeur C pour Linux
- **Décodeur Arduino** : F1LVT - Décodeur embarqué

---

## Licence

Cet outil est fourni à des fins éducatives et de test uniquement.

**⚠️ RAPPEL LÉGAL** : L'émission sur 406 MHz est strictement réglementée. Toute émission non autorisée est passible de poursuites pénales et déclenche une alerte SAR réelle mobilisant des moyens de secours.

---
