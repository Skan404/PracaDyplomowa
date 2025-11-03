# --- Importy --- #
import librosa
import numpy as np
import pandas as pd
import os
import json
import random
import scipy.stats
from collections import Counter
import warnings 

# --- NOWY IMPORT --- #
# To jest kluczowa biblioteka do mierzenia głośności wg standardu LUFS
# Trzeba ją doinstalować: pip install pyloudnorm
try:
    import pyloudnorm as pyln
except ImportError:
    print("BŁĄD KRYTYCZNY: Biblioteka 'pyloudnorm' nie jest zainstalowana.")
    print("Proszę uruchomić: pip install pyloudnorm")
    exit()

# Wyłączam to ostrzeżenie, bo pyloudnorm krzyczy przy krótkich plikach
warnings.filterwarnings("ignore", category=UserWarning, module='pyloudnorm')

# --- FUNKCJE POMOCNICZE --- #

def normalize_key_name(key_str):
    """Mała funkcja, żeby 'A maj' i 'A# min' zawsze wyglądały tak samo (z '♯')."""
    if not isinstance(key_str, str): return "BŁĄD FORMATU"
    # Ujednolicamy spacje i zamieniamy # na ładny znak ♯
    return key_str.replace(" maj", " maj").replace(" min", " min").replace("#", "♯")

def amplitude_to_dbfs(amp):
    """Konwertuje amplitudę (z zakresu 0.0 do 1.0) na dBFS (decybele)."""
    if amp <= 0: return -np.inf # Zabezpieczenie przed log(0)
    # Wzór na zamianę amplitudy liniowej na logarytmiczną (dB)
    return 20 * np.log10(amp)

# --- GŁÓWNA FUNKCJA ANALIZY --- #

def przeanalizuj_plik(sciezka_pliku, folder_metadata_test=None):
    """
    To jest serce programu.
    Bierze ścieżkę do pliku, mieli go przez librosa i pyloudnorm,
    i zwraca słownik (dict) ze wszystkimi wynikami.
    """
    
    # KROK 1: WCZYTANIE AUDIO
    try:
        print(f"Ładowanie pliku: {sciezka_pliku}...")
        # Wczytujemy plik:
        # sr=None -> zachowuje oryginalne próbkowanie (ważne dla pyloudnorm!)
        # mono=False -> wczytuje stereo, jeśli jest
        y, sr = librosa.load(sciezka_pliku, sr=None, mono=False) 
        y_do_analizy = y # Robocza kopia danych audio

        duration_sec = librosa.get_duration(y=y_do_analizy, sr=sr)
        
        print(f"Plik załadowany (SR={sr} Hz, Długość={duration_sec:.2f}s). Rozpoczynam analizę...")

    except Exception as e:
        print(f"Błąd podczas ładowania pliku: {e}")
        return None

    # KROK 2: INICJALIZACJA WYNIKÓW
    nazwa_pliku = os.path.basename(sciezka_pliku)
    # Przygotowujemy "kontener" na wyniki
    wyniki = {
        'Nazwa Pliku': nazwa_pliku,
        'Długość (s)': round(duration_sec, 2),
        'Ścieżka': sciezka_pliku
    }

    # Sprawdzamy, czy plik jest stereo (ma więcej niż 1 wymiar i co najmniej 2 kanały)
    is_stereo = y_do_analizy.ndim > 1 and y_do_analizy.shape[0] >= 2
    
    if not is_stereo:
        # Plik jest mono
        y_mono = y_do_analizy 
    else:
        # Plik jest stereo (lub 5.1 itp.)
        # pyloudnorm wymaga kanałów w kolumnach, a librosa wczytuje w wierszach (kanał, czas)
        # Transponujemy (.T) tylko pierwsze dwa kanały (L, R)
        y_do_analizy_transposed = y_do_analizy[:2,:].T 
        # Do analizy tempa i tonacji i tak potrzebujemy mono - uśredniamy L i R
        y_mono = np.mean(y_do_analizy[:2, :], axis=0)

    # KROK 3A: TEMPO I STABILNOŚĆ
    try:
        # librosa.beat.tempo zwraca wektor tempa dynamicznego (zmiennego w czasie)
        tempo_dynamiczne = librosa.beat.tempo(y=y_mono, sr=sr)
        
        # Jeśli plik jest bardzo cichy albo nie ma rytmu, wektor może być pusty
        if len(tempo_dynamiczne) == 0: 
            print("   Info o tempie: Zbyt mało danych rytmicznych (plik jest cichy). Przypisuję 'BŁĄD'.")
            wyniki['Tempo Obliczone (BPM)'] = "BŁĄD (Cisza)"
            wyniki['Stabilność Rytmu (0-100)'] = "BŁĄD (Cisza)"
            raise Exception("Pusty wektor tempa (obsłużono)")

        # --- Logika do znalezienia "najlepszego" tempa ---
        # 1. Robimy histogram, żeby znaleźć najczęstsze tempo
        counts, bins = np.histogram(tempo_dynamiczne, bins=np.arange(30, 301))
        peak_index = np.argmax(counts); peak_tempo = round(bins[peak_index] + 0.5)
        
        # 2. Problem: librosa często myli 70 BPM ze 140 BPM (x2) albo 120 z 80 (x1.5 / x0.67)
        # Sprawdzamy więc "kandydatów" (peak_tempo i jego mnożniki)
        ratios_to_check = [1.0, 2.0, 0.5, 1.5, 2/3]; candidates = {}
        for ratio in ratios_to_check:
            # logika sprawdzania 'siły' tempa w histogramie
            related_tempo_raw = peak_tempo * ratio; related_tempo = round(related_tempo_raw)
            related_bin_edge_index = np.argmin(np.abs(bins - related_tempo_raw))
            related_bin_index = min(related_bin_edge_index, len(counts) - 1)
            related_strength = counts[related_bin_index]
            if related_tempo not in candidates: candidates[related_tempo] = related_strength
            
        # 3. Wybieramy najlepszego kandydata, ale preferujemy "normalny" zakres
        PREF_MIN_BPM, PREF_MAX_BPM = 70, 150
        preferred_candidates = {t: s for t, s in candidates.items() if PREF_MIN_BPM < t < PREF_MAX_BPM}
        
        best_tempo = max(candidates, key=candidates.get) # Najlepszy ogólnie
        if preferred_candidates: 
            best_tempo = max(preferred_candidates, key=preferred_candidates.get) # Najlepszy z preferowanego zakresu
            
        final_dominant_tempo = best_tempo
        if final_dominant_tempo != peak_tempo: print(f"   Info o tempie: Korekta {peak_tempo} -> {final_dominant_tempo} BPM.")
        
        wyniki['Tempo Obliczone (BPM)'] = float(final_dominant_tempo)
        
        # Obliczanie stabilności:
        # Jak bardzo tempo "skacze" (odchylenie standardowe) w stosunku do średniej
        if final_dominant_tempo == 0: ocena_stabilnosci_rytmu = 0.0
        else: 
            stabilnosc_rytmu_proc = (np.std(tempo_dynamiczne) / final_dominant_tempo) * 100
            # Przeskalowanie na 0-100 (im mniejsze odchylenie, tym wyższa ocena)
            ocena_stabilnosci_rytmu = max(0, 100 - stabilnosc_rytmu_proc * 5)
        wyniki['Stabilność Rytmu (0-100)'] = round(ocena_stabilnosci_rytmu, 2)
        
    except Exception as e:
        if str(e) == "Pusty wektor tempa (obsłużono)": pass # To obsłużyliśmy wyżej
        else:
            print(f"Błąd analizy tempa: {e}")
            if 'Tempo Obliczone (BPM)' not in wyniki: wyniki['Tempo Obliczone (BPM)'] = "BŁĄD"
            if 'Stabilność Rytmu (0-100)' not in wyniki: wyniki['Stabilność Rytmu (0-100)'] = "BŁĄD"

    # KROK 3B: TONACJA (Metoda: HPSS + CQT + Krumhansl)
    try:
        # To są "wzorcowe" profile tonacji durowej i molowej (wg Krumhansla)
        # Mówią, jak mocno powinny brzmieć poszczególne nuty (C, C#...) w danej tonacji
        prof_dur=np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
        prof_moll=np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
        nuty=['C','C♯','D','D♯','E','F','F♯','G','G♯','A','A♯','B']
        
        # 1. Rozdzielamy audio na część harmoniczną (melodia) i perkusyjną (bębny)
        print("   Info o tonacji: HPSS..."); 
        y_harmonic, _ = librosa.effects.hpss(y_mono, margin=1.0)
        
        # 2. Tworzymy chromagram (pokazuje "energię" każdej z 12 nut w czasie)
        # Używamy CQT (lepsze do tonacji niż standardowe STFT)
        print("  Info o tonacji: chroma_cqt..."); 
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
        if chroma.shape[1]==0: raise Exception("Pusty chromagram (np. cisza)")
        
        # 3. Uśredniamy chromagram, żeby dostać jeden "profil" dla całego utworu
        tonacja_profil_srednia = np.mean(chroma, axis=1)
        suma_profilu = np.sum(tonacja_profil_srednia)
        if suma_profilu <= 0: raise Exception("Pusty profil harmoniczny (cisza)")
        
        # Normalizujemy profil (żeby suma była 1)
        tonacja_profil_norm = tonacja_profil_srednia / suma_profilu
        
        # 4. Porównujemy nasz profil ze wszystkimi 24 tonacjami (12 dur, 12 moll)
        wszystkie_korelacje = []
        for i in range(12): # Pętla po wszystkich 12 nutach (C, C#...)
            # "Obracamy" nasz profil, żeby sprawdzić dopasowanie np. do D, potem D# itd.
            profil_obrocony = np.roll(tonacja_profil_norm, -i)
            # Liczymy korelację Pearsona (jak bardzo profil pasuje do wzorca dur/moll)
            korelacja_dur = scipy.stats.pearsonr(profil_obrocony, prof_dur)[0]
            korelacja_moll = scipy.stats.pearsonr(profil_obrocony, prof_moll)[0]
            wszystkie_korelacje.extend([(f"{nuty[i]} maj", korelacja_dur), (f"{nuty[i]} min", korelacja_moll)])
            
        # 5. Sortujemy wyniki - najlepsze dopasowanie (najwyższa korelacja) wygrywa
        posortowane_tonacje = sorted(wszystkie_korelacje, key=lambda x: x[1], reverse=True)
        top_5_tonacje = [t[0] for t in posortowane_tonacje[:5]]
        print(f"  Info o tonacji: Top 5 to {top_5_tonacje}")
        
        wyniki['Tonacja (Obliczona)'] = top_5_tonacje[0] if top_5_tonacje else "BŁĄD"
        wyniki['Top 5 Tonacje'] = ", ".join(top_5_tonacje)
        wyniki['Top 5 Tonacje (lista)'] = top_5_tonacje # Pomocnicze do walidacji
        
    except Exception as e:
        print(f"Błąd analizy tonacji: {e}")
        wyniki['Tonacja (Obliczona)'] = "BŁĄD"; wyniki['Top 5 Tonacje'] = "BŁĄD"; wyniki['Top 5 Tonacje (lista)'] = []

    # KROK 3C: (Głośność, Szczyt, Dynamika, Stereo)
    try:
        # Inicjalizujemy miernik głośności z poprawnym SR
        # block_size 400ms jest zgodny ze standardem BS.1770
        meter = pyln.Meter(sr, block_size=0.400) 
        
        # Przygotowujemy dane dla miernika (musi być 2D, nawet dla mono)
        # (dla stereo musieliśmy transponować)
        loudness_data = y_do_analizy_transposed if is_stereo else y_mono[:, np.newaxis]
        
        # Standard wymaga minimum 400ms audio do pomiaru
        if loudness_data.shape[0] < sr * 0.4:
            print("  Ostrzeżenie: Plik zbyt krótki (<400ms) do pomiaru LUFS.")
            lufs = -np.inf # Ustawiamy na "bardzo cicho"
        else:
            # Obliczamy ZINTEGROWANĄ głośność (integrated loudness) dla całego pliku
            lufs = meter.integrated_loudness(loudness_data)
        
        # Zabezpieczenie przed -inf, jeśli plik był totalną ciszą
        wyniki['Głośność (LUFS)'] = round(max(-70.0, lufs), 2) if np.isfinite(lufs) else -70.0

        print("  Info o głośności: Obliczam Sample Peak (dBFS)...")
        # Szczyt (Peak) - najgłośniejsza pojedyncza próbka w całym sygnale
        sample_peak = np.max(np.abs(y_do_analizy))
        sample_peak_dbfs = amplitude_to_dbfs(sample_peak)
        wyniki['Szczyt (Peak dBFS)'] = round(max(-70.0, sample_peak_dbfs), 2) if np.isfinite(sample_peak_dbfs) else -70.0
            
        # Dynamika (PLR - Peak-to-Loudness Ratio)
        # Różnica między szczytem a średnią głośnością (LUFS)
        peak_value_for_plr = wyniki.get('Szczyt (Peak dBFS)')
        if isinstance(peak_value_for_plr, (int, float)) and isinstance(wyniki['Głośność (LUFS)'], (int, float)):
            plr = peak_value_for_plr - wyniki['Głośność (LUFS)']
            wyniki['Dynamika (PLR dB)'] = round(plr, 2)
        else:
            wyniki['Dynamika (PLR dB)'] = "BŁĄD"

        # Korelacja Stereo
        if is_stereo:
            y_left = y_do_analizy[0, :]; y_right = y_do_analizy[1, :]
            # Sprawdzamy, czy kanały nie są identyczne (albo oba ciche)
            if np.all(y_left == 0) and np.all(y_right == 0): 
                correlation_coefficient = 1.0 # Idealna cisza to technicznie mono
            else:
                # Liczymy współczynnik korelacji między L i R
                correlation_matrix = np.corrcoef(y_left, y_right)
                correlation_coefficient = correlation_matrix[0, 1]
                # Czasem wychodzi NaN (np. jeden kanał cichy), ustawiamy na 0 (szeroko)
                if np.isnan(correlation_coefficient): correlation_coefficient = 0.0
            # 1.0 -> idealne mono
            # 0.0 -> bardzo szeroko
            # -1.0 -> przeciwfaza
            wyniki['Korelacja Stereo (-1 do 1)'] = round(correlation_coefficient, 3)
        else:
            # Plik mono z definicji ma korelację 1.0
            wyniki['Korelacja Stereo (-1 do 1)'] = 1.0
            
    except Exception as e:
        print(f"Błąd analizy metryk głośności/stereo: {e}")
        # Jak coś pójdzie nie tak, wpiszemy błędy
        for key in ['Głośność (LUFS)', 'Szczyt (Peak dBFS)', 'Dynamika (PLR dB)', 'Korelacja Stereo (-1 do 1)']:
            if key not in wyniki: wyniki[key] = "BŁĄD"

    # KROK 4: WALIDACJA (działa tylko w trybie 'verify')
    if folder_metadata_test:
        ground_truth_bpm = "BŁĄD ODCZYTU"; ground_truth_key = "BŁĄD ODCZYTU"
        try:
            # Szukamy pliku .json o tej samej nazwie co plik audio
            nazwa_bazowa, _ = os.path.splitext(nazwa_pliku)
            json_filename = f"{nazwa_bazowa}.json"; json_path = os.path.join(folder_metadata_test, json_filename)
            
            if os.path.exists(json_path):
                # Wczytujemy "prawdziwe" dane z JSONa
                with open(json_path, 'r', encoding='utf-8') as f: metadata = json.load(f)
                ground_truth_bpm = metadata.get('bpm', 'BRAK KLUCZA'); ground_truth_key = metadata.get('key', 'BRAK KLUCZA')
            else: print(f"OSTRZEŻENIE: Nie znaleziono JSON: {json_path}")
        except Exception as e: print(f"BŁĄD ODCZYTU JSON ({json_path}): {e}")
        
        wyniki['Prawdziwe BPM (JSON)'] = ground_truth_bpm
        wyniki['Prawdziwa Tonacja (JSON)'] = ground_truth_key
        
        # Sprawdzanie zgodności BPM
        wyniki['Zgodność BPM'] = 'NIE'
        try:
            obliczone_bpm = wyniki['Tempo Obliczone (BPM)']
            prawdziwe_bpm = float(ground_truth_bpm)
            tolerancja = 2.0 # Dajemy 2 BPM tolerancji
            
            # Sprawdzamy też, czy nie trafiliśmy w połowę, podwójne itp.
            relacje = {1.0:" (x1)", 2.0:" (x2)", 0.5:" (x0.5)", 1.5:" (x1.5)", 2/3:" (x0.67)"}
            for r_val, r_str in relacje.items():
                if abs(obliczone_bpm * r_val - prawdziwe_bpm) <= tolerancja: 
                    wyniki['Zgodność BPM'] = f'TAK{r_str}'; break
        except Exception as e: print(f"Błąd walidacji BPM: {e}"); wyniki['Zgodność BPM'] = 'BŁĄD'
        
        # Sprawdzanie zgodności Tonacji
        wyniki['Zgodność Tonacji'] = 'NIE'
        try:
            # Normalizujemy nazwę z JSONa, żeby dało się porównać
            prawdziwa_tonacja_std = normalize_key_name(ground_truth_key)
            # Sukces, jeśli prawdziwa tonacja jest w naszym Top 5
            if prawdziwa_tonacja_std in wyniki['Top 5 Tonacje (lista)']: 
                wyniki['Zgodność Tonacji'] = 'TAK'
        except Exception as e: print(f"Błąd walidacji Tonacji: {e}"); wyniki['Zgodność Tonacji'] = 'BŁĄD'
            
    # Usuwamy pomocniczą listę przed zapisem
    wyniki.pop('Top 5 Tonacje (lista)', None)
    
    print("Analiza zakończona.")
    return wyniki


# --- FUNKCJE ZAPISU (ZAKTUALIZOWANE O NOWĄ KOLUMNĘ) ---

def zapisz_wyniki_analizy(dane, plik_wyjsciowy):
    """
    Zapisuje wyniki do Excela. 
    Ważne: Ta funkcja DOPISUJE do istniejącego pliku (jeśli istnieje).
    """
    if not dane: 
        print("Brak danych do zapisu.")
        return
    
    df = pd.DataFrame(dane)
    
    # Definiujemy kolejność kolumn, żeby w Excelu był porządek
    kolejnosc_kolumn = [
        'Nazwa Pliku', 
        'Długość (s)',
        'Tempo Obliczone (BPM)', 
        'Tonacja (Obliczona)',
        'Stabilność Rytmu (0-100)', 
        'Głośność (LUFS)', 
        'Szczyt (Peak dBFS)',
        'Dynamika (PLR dB)', 
        'Korelacja Stereo (-1 do 1)', 
        'Top 5 Tonacje',
        'Ścieżka'
    ]
    
    # Bierzemy tylko te kolumny, które faktycznie mamy
    finalne_kolumny = [k for k in kolejnosc_kolumn if k in df.columns]
    df = df[finalne_kolumny]

    try:
        # Sprawdzamy, czy plik już istnieje i nie jest pusty
        if os.path.exists(plik_wyjsciowy) and os.path.getsize(plik_wyjsciowy) > 0:
            print(f"Dopisywanie do: {plik_wyjsciowy}")
            # Wczytujemy stary plik, żeby wiedzieć, od którego wiersza zacząć
            existing_df = pd.read_excel(plik_wyjsciowy)
            startrow = len(existing_df)
            
            # Otwieramy plik w trybie 'append' ('a')
            with pd.ExcelWriter(plik_wyjsciowy, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                # Dopasowujemy kolumny (na wszelki wypadek)
                nowe_kolumny = [k for k in existing_df.columns if k in df.columns]
                df_do_zapisu = df[nowe_kolumny]
                # Zapisujemy nowe dane BEZ NAGŁÓWKA (header=False)
                df_do_zapisu.to_excel(writer, index=False, header=False, startrow=startrow)
            print("Dopisano dane.")
        else:
            # Jeśli plik nie istnieje, tworzymy go normalnie (z nagłówkiem)
            print(f"Tworzenie nowego pliku: {plik_wyjsciowy}")
            df.to_excel(plik_wyjsciowy, index=False, engine='openpyxl')
            print("Zapisano plik.")
            
    except Exception as e:
        print(f"Wystąpił błąd przy zapisie do Excela: {e}")
        # To ważna uwaga - jak zmienimy kolumny, stary plik Excela trzeba usunąć ręcznie
        print("WAŻNE: Jeśli błąd to 'not in index', usuń stary plik 'analiza_muzyczna.xlsx', aby zaktualizować nagłówki.")


def zapisz_wyniki_walidacji(dane, plik_wyjsciowy):
    """
    Zapisuje wyniki walidacji.
    Ta funkcja ZAWSZE NADPISUJE stary plik.
    """
    if not dane: 
        print("Brak danych do zapisu.")
        return
    
    df = pd.DataFrame(dane)
    
    # Tutaj kolejność kolumn jest inna, uwzględnia porównanie
    kolejnosc_kolumn = [
        'Nazwa Pliku',
        'Długość (s)',
        'Tempo Obliczone (BPM)', 
        'Prawdziwe BPM (JSON)', 
        'Zgodność BPM',
        'Tonacja (Obliczona)',
        'Prawdziwa Tonacja (JSON)', 
        'Zgodność Tonacji',
        'Stabilność Rytmu (0-100)', 
        'Głośność (LUFS)', 
        'Szczyt (Peak dBFS)',
        'Dynamika (PLR dB)', 
        'Korelacja Stereo (-1 do 1)', 
        'Top 5 Tonacje',
        'Ścieżka'
    ]
    
    finalne_kolumny = [k for k in kolejnosc_kolumn if k in df.columns]
    df = df[finalne_kolumny]

    try:
        # Zwykły zapis, który zawsze nadpisuje plik
        print(f"Tworzenie/Nadpisywanie pliku: {plik_wyjsciowy}")
        df.to_excel(plik_wyjsciowy, index=False, engine='openpyxl')
        print("Zapisano plik.")
    except Exception as e:
        print(f"Wystąpił błąd przy zapisie do Excela: {e}")


# --- GŁÓWNA CZĘŚĆ SKRYPTU --- #
if __name__ == "__main__":
    
    # --- GŁÓWNY PRZEŁĄCZNIK ---
    # 'analyze' -> analizuje pliki z folderu 'muzyka' i zapisuje/dopisuje do 'analiza_muzyczna.xlsx'
    # 'verify'  -> analizuje pliki z 'muzyka_test', porównuje z 'meta_test' i nadpisuje 'walidacja_wynikow.xlsx'
    TRYB = 'analyze' 

    def sprawdz_folder(folder):
        """Mała funkcja, która tworzy folder, jeśli go nie ma."""
        if not os.path.exists(folder):
            print(f"Folder '{folder}' nie istnieje. Tworzę...")
            os.makedirs(folder)
            # Dodajemy plik "instrukcję" dla użytkownika
            if 'muzyka' in folder:
                with open(os.path.join(folder, 'umiesc_tutaj_swoje_pliki.txt'), 'w') as f:
                    f.write('Proszę, umieść tutaj pliki .mp3 lub .wav.')
            return False
        return True

    # --- Logika dla trybu ANALIZY --- #
    if TRYB == 'analyze':
        print("--- TRYB: ANALIZA ---")
        FOLDER_MUZYKA = 'muzyka'; PLIK_WYJSCIOWY = 'analiza_muzyczna.xlsx'
        sprawdz_folder(FOLDER_MUZYKA)
        
        # Szukamy tylko plików audio
        pliki_nazwy = [f for f in os.listdir(FOLDER_MUZYKA) if f.endswith(('.mp3', '.wav', '.flac'))]
        if not pliki_nazwy: print(f"Brak plików w '{FOLDER_MUZYKA}'."); exit()
        
        print(f"Znaleziono {len(pliki_nazwy)} plików. Analizuję...")
        sciezki = [os.path.join(FOLDER_MUZYKA, f) for f in pliki_nazwy]
        
        # To jest tzw. 'list comprehension' - sprytny sposób na pętlę
        # Dla każdej 'sciezka' w 'sciezki':
        # 1. Uruchom przeanalizuj_plik(sciezka, None) i zapisz wynik do 'res'
        # 2. Jeśli 'res' nie jest 'None' (czyli nie było błędu ładowania), dodaj 'res' do nowej listy
        wszystkie_wyniki = [res for sciezka in sciezki if (res := przeanalizuj_plik(sciezka, None)) is not None]
        
        # Zapisujemy wszystko, co udało się zebrać
        if wszystkie_wyniki: zapisz_wyniki_analizy(wszystkie_wyniki, PLIK_WYJSCIOWY)

    # --- Logika dla trybu WERYFIKACJI --- #
    elif TRYB == 'verify':
        print("--- TRYB: WERYFIKACJA ---")
        FOLDER_MUZYKA = 'muzyka_test'; FOLDER_META = 'meta_test'; PLIK_WYJSCIOWY = 'walidacja_wynikow.xlsx'
        # MAX_PLIKOW = 0 -> analizuj wszystkie
        # MAX_PLIKOW = 10 -> przeanalizuj 10 losowych (dobre do szybkich testów)
        MAX_PLIKOW = 10 
        
        if not sprawdz_folder(FOLDER_MUZYKA) or not sprawdz_folder(FOLDER_META): 
            print("BŁĄD: Sprawdź foldery testowe."); exit()
            
        pliki_nazwy_all = [f for f in os.listdir(FOLDER_MUZYKA) if f.endswith(('.mp3', '.wav', '.flac'))]
        if not pliki_nazwy_all: print(f"Brak plików w '{FOLDER_MUZYKA}'."); exit()
        
        # Losowanie próbki do testów
        if 0 < MAX_PLIKOW < len(pliki_nazwy_all):
            print(f"Znaleziono {len(pliki_nazwy_all)}. Losuję {MAX_PLIKOW}...")
            pliki_nazwy = random.sample(pliki_nazwy_all, MAX_PLIKOW)
        else:
            print(f"Znaleziono {len(pliki_nazwy_all)}. Analizuję wszystkie..."); 
            pliki_nazwy = pliki_nazwy_all
            
        sciezki = [os.path.join(FOLDER_MUZYKA, f) for f in pliki_nazwy]
        
        # Ta sama pętla co w trybie 'analyze', ale przekazujemy FOLDER_META
        wszystkie_wyniki = [res for sciezka in sciezki if (res := przeanalizuj_plik(sciezka, FOLDER_META)) is not None]
        
        if wszystkie_wyniki: zapisz_wyniki_walidacji(wszystkie_wyniki, PLIK_WYJSCIOWY)

    else: 
        print(f"BŁĄD: Nieznany tryb '{TRYB}'.")