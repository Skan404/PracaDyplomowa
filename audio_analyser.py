# --- Sekcja Importów Bibliotek --- #
import librosa
import numpy as np
import pandas as pd
import os
import json
import scipy.stats
import warnings 
import datetime
import pyloudnorm as pyln


# --- STAŁE KONFIGURACYJNE SYSTEMU OCEN (Skala 0-2 punkty) ---
# Poniższe progi definiują kryteria przyznawania punktów za zgodność
# wygenerowanego pliku z określonymi celami (scenariuszem) oraz ogólnymi
# standardami technicznymi produkcji muzycznej.

# 1. Długość (s): Ocena błędu bezwzględnego względem celu (DURATION_TARGET).
# 2 punkty: Błąd absolutny <= DURATION_P1_THRESHOLD (np. 1s)
# 1 punkt:  Błąd absolutny <= DURATION_P2_THRESHOLD (np. 10s)
DURATION_TARGET = 100.0
DURATION_P1_THRESHOLD = 1.0
DURATION_P2_THRESHOLD = 10.0

# 2. Tempo (BPM): Ocena błędu bezwzględnego względem celu (target_bpm).
# 2 punkty: Błąd absolutny <= TEMPO_P1_THRESHOLD (np. 5 BPM)
# 1 punkt:  Błąd absolutny <= TEMPO_P2_THRESHOLD (np. 10 BPM)
TEMPO_P1_THRESHOLD = 5.0 
TEMPO_P2_THRESHOLD = 10.0 

# 3. Tonacja (%): Ocena procentowej zgodności z tonacją docelową.
# 2 punkty: Zgodność >= KEY_P1_THRESHOLD (np. 80%)
# 1 punkt:  Zgodność >= KEY_P2_THRESHOLD (np. 50%)
KEY_P1_THRESHOLD = 80.0
KEY_P2_THRESHOLD = 50.0

# 4. Głośność (LUFS): Ocena, czy zintegrowana głośność mieści się w docelowym zakresie.
# 2 punkty: Wartość mieści się w zakresie LUFS_P1_RANGE
# 1 punkt:  Wartość mieści się w szerszym zakresie LUFS_P2_RANGE
# Uwaga: Standardy streamingowe (np. Spotify) normalizują do ok. -14 LUFS,
# jednak wiele gatunków (np. EDM, Pop) jest masterowanych znacznie głośniej.
LUFS_P1_RANGE = (-14.0, -8.0) # Zakres optymalny, profesjonalny
LUFS_P2_RANGE = (-18.0, -6.0) # Zakres akceptowalny, szerszy

# 5. Szczyt (Peak dBFS): Ocena "headroomu", czyli przestrzeni między najwyższym szczytem a 0 dBFS.
# 2 punkty: Szczyt <= PEAK_P1_THRESHOLD (np. -1.0 dBFS) - idealny headroom dla masteringu
# 1 punkt:  Szczyt <= PEAK_P2_THRESHOLD (np. 0.0 dBFS) - na granicy, ale bez cyfrowego przesteru (clippingu)
PEAK_P1_THRESHOLD = -1.0 
PEAK_P2_THRESHOLD = 0.0 

# 6. Dynamika (PLR dB): Ocena zakresu dynamicznego jako Peak-to-Loudness Ratio (PLR).
# 2 punkty: Wartość mieści się w zakresie PLR_P1_RANGE
# 1 punkt:  Wartość mieści się w szerszym zakresie PLR_P2_RANGE
# Niskie wartości PLR (np. < 8) oznaczają wysoką kompresję ("ściana dźwięku", ang. "loudness war"),
# typową dla niektórych gatunków elektronicznych. 
# Wysokie wartości (np. > 15) oznaczają dużą dynamikę, typową np. dla muzyki klasycznej.
PLR_P1_RANGE = (8.0, 15.0) # Zakres "zdrowy" dla większości gatunków
PLR_P2_RANGE = (6.0, 18.0) # Zakres akceptowalny, szerszy

# 7. Korelacja Stereo: Ocena szerokości obrazu stereo i zgodności fazowej.
# Wartości: 1.0 = idealne mono; 0 = maksymalna szerokość (lub brak korelacji); -1.0 = odwrotna faza
# 2 punkty: Wartość mieści się w zakresie STEREO_P1_RANGE
# 1 punkt:  Wartość mieści się w szerszym zakresie STEREO_P2_RANGE
STEREO_P1_RANGE = (0.1, 0.8) # Dobry, szeroki obraz stereo, bezpieczny dla kompatybilności mono
STEREO_P2_RANGE = (-0.1, 0.95) # Akceptowalny, unika skrajnej przeciwfazy i czystego mono

# --- Funkcje Pomocnicze --- #

def normalize_key_name(key_str):
    """
    Normalizuje formatowanie nazwy tonacji.
    Zapewnia spójność danych wejściowych z config.json i danych wyjściowych algorytmu.
    Przykład: "C#" -> "C♯"
    """
    if not isinstance(key_str, str): 
        return "BŁĄD FORMATU"
        
    return key_str.replace("#", "♯")

def amplitude_to_dbfs(amplitude):
    """
    Konwertuje liniową wartość amplitudy (zakres 0.0-1.0) na skalę logarytmiczną dBFS.
    (Decybele Względem Pełnej Skali).
    """
    if amplitude <= 0: return -np.inf # Logarytm z zera lub liczby ujemnej jest niezdefiniowany
    
    # Wzór na konwersję do decybeli dla amplitudy
    return 20 * np.log10(amplitude)

# --- Główna Funkcja Analityczna --- #

def analyze_file(file_path, model_name, scenario_name, scenario_config):
    """
    Centralna funkcja przetwarzająca pojedynczy plik audio.
    
    Pobiera ścieżkę do pliku, metadane (model, scenariusz) oraz konfigurację
    celów (docelowe tempo i tonację) ze słownika scenario_config.
    
    Zwraca słownik (dict) zawierający wszystkie obliczone metryki.
    """
    
    # Krok 1: Wczytanie sygnału audio
    try:
        print(f"\nŁadowanie pliku: {os.path.basename(file_path)}...")
        # Wczytujemy plik:
        # sr=None: zachowuje oryginalną częstotliwość próbkowania (Sample Rate)
        # mono=False: wczytuje plik jako stereo (jeśli to możliwe), co jest kluczowe dla analizy korelacji
        audio_signal, sample_rate = librosa.load(file_path, sr=None, mono=False) 
        
        # W dalszej części skryptu y_do_analizy będzie referencją do pełnego (potencjalnie stereo) sygnału
        analysis_signal = audio_signal 
        duration_sec = librosa.get_duration(y=analysis_signal, sr=sample_rate)
        print(f"Plik załadowany (SR={sample_rate} Hz, Długość={duration_sec:.2f}s). Rozpoczynam analizę...")
    except Exception as e:
        print(f"Krytyczny błąd podczas ładowania pliku: {e}")
        return None # Zwracamy None, aby zasygnalizować błąd ładowania

    # Krok 2: Inicjalizacja słownika wyników
    file_name = os.path.basename(file_path)
    results = {
        'Model': model_name,
        'Prompt (Scenariusz)': scenario_config.get('full_prompt', scenario_name),
        'Nazwa Pliku': file_name,
        'Długość (s)': round(duration_sec, 2),
        'Ścieżka': file_path
    }

    # Pobieranie celów analitycznych z konfiguracji scenariusza
    target_bpm = scenario_config.get('target_bpm') 
    target_key = scenario_config.get('target_key') 

    # Sprawdzenie, czy sygnał jest stereo
    is_stereo = analysis_signal.ndim > 1 and analysis_signal.shape[0] >= 2
    
    if not is_stereo:
        # Plik jest mono. Tworzymy referencję 'mono_signal'
        mono_signal = analysis_signal 
    else:
        # Plik jest stereo.
        # 'analysis_signal' (w formie [kanał, próbka]) musi zostać transponowany
        # do formatu [próbka, kanał] dla biblioteki pyloudnorm.
        # Ograniczamy do pierwszych dwóch kanałów (L, R) na wypadek plików wielokanałowych (np. 5.1).
        analysis_signal_transposed = analysis_signal[:2,:].T 
        # Tworzymy wersję mono (uśrednienie kanałów) dla algorytmów, które tego wymagają (np. HPSS, tempo).
        mono_signal = np.mean(analysis_signal[:2, :], axis=0)
    
    # --- Sekcja HPSS (Harmonic-Percussive Source Separation) ---
    # HPSS to technika separacji sygnału na część harmoniczną i część perkusyjną.
    # Używamy y_harmonic do analizy tonacji (jest czystsza)
    # Używamy y_percussive do analizy tempa (jest dokładniejsza)
    print("   Info: Wykonuję separację HPSS (Harmonic-Percussive)...")
    try:
        harmonic_signal, percussive_signal = librosa.effects.hpss(mono_signal, margin=1.0) 
    except Exception as e:
        # W przypadku błędu (np. bardzo krótkiego pliku), wracamy do używania pełnego sygnału mono
        print(f"   Ostrzeżenie: Separacja HPSS nie powiodła się ({e}). Używam pełnego sygnału mono.")
        harmonic_signal = mono_signal
        percussive_signal = mono_signal

    # Krok 3A: Analiza Tempa (BPM)
    try:
        # Jeśli sygnał perkusyjny jest bardzo cichy, analiza tempa na nim może zawieść. Lepiej wtedy użyć pełnego sygnału mono.
        if np.max(np.abs(percussive_signal)) < 0.01:
            print("   Info o tempie: Sygnał perkusyjny jest zbyt cichy. Używam pełnego sygnału mono.")
            audio_source_for_tempo = mono_signal
        else:
            print("   Info o tempie: Używam sygnału perkusyjnego (HPSS) do analizy tempa.")
            audio_source_for_tempo = percussive_signal
        
        # Obliczenie dynamicznego tempa (wektor tempogramu)
        dynamic_tempo = librosa.beat.tempo(y=audio_source_for_tempo, sr=sample_rate)
        
        if len(dynamic_tempo) == 0: 
            print("   Info o tempie: Nie wykryto wystarczających danych rytmicznych (plik może być ciszą).")
            results['Tempo (BPM)'] = "BŁĄD (Cisza)"
            raise Exception("Pusty wektor tempa (obsłużono)")

        # Tworzenie histogramu wykrytych wartości tempa
        counts, bins = np.histogram(dynamic_tempo, bins=np.arange(30, 301))
        # Znalezienie najsilniejszego piku (najczęściej wykrywanego tempa)
        peak_index = np.argmax(counts); peak_tempo = round(bins[peak_index] + 0.5)
        
        # Logika walidacji tempa (Problem "Ośmiokąta")
        # Algorytmy często mylą tempo o połowę (half-time) lub podwójnie (double-time).
        # Algorytmy mają problem z odróżnieniem metrum parzystego (np. 4/4) od triolowego (np. 12/8)
        # Ta sekcja sprawdza kandydatów (0.5x, 1x, 1.5x, 2x, 0.66x ) i ich siłę w histogramie.
        ratios_to_check = [1.0, 2.0, 0.5, 1.5, 2/3]; candidates = {}
        for ratio in ratios_to_check:
            related_tempo_raw = peak_tempo * ratio; related_tempo = round(related_tempo_raw)
            if not (30 <= related_tempo_raw <= 300): continue # Odrzucamy tempa poza realistycznym zakresem
                
            # Sprawdzamy, jaką siłę (counts) ma ten kandydat w histogramie
            related_bin_edge_index = np.argmin(np.abs(bins - related_tempo_raw))
            related_bin_index = min(related_bin_edge_index, len(counts) - 1)
            related_strength = counts[related_bin_index]
            
            # Zapisujemy najsilniejszego kandydata dla danego zaokrąglonego tempa
            if related_tempo not in candidates: 
                candidates[related_tempo] = related_strength
            else:
                candidates[related_tempo] = max(candidates[related_tempo], related_strength)
                
        if target_bpm and candidates:
            # Jeśli mamy cel (target_bpm), wybieramy kandydata najbliższego temu celowi.
            # To drastycznie poprawia trafność w scenariuszach testowych.
            final_dominant_tempo = min(
                candidates.keys(), 
                key=lambda candidate_tempo: abs(candidate_tempo - target_bpm)
            )
            print(f"   Info o tempie: Użyto celu {target_bpm} BPM. Wybrano {final_dominant_tempo} BPM z kandydatów.")
            
        elif candidates:
            # Jeśli nie mamy celu, wybieramy po prostu najsilniejszego kandydata.
            final_dominant_tempo = max(candidates, key=candidates.get)
            print(f"   Info o tempie: Brak celu. Wybrano najsilniejszy pik: {final_dominant_tempo} BPM.")
        else:
            # Sytuacja awaryjna (bardzo rzadka)
            final_dominant_tempo = peak_tempo 

        results['Tempo (BPM)'] = round(float(final_dominant_tempo), 2)
        results['Tempo (Cel)'] = target_bpm # Zapisujemy cel, aby użyć go w funkcji oceniającej

    except Exception as e:
        if str(e) == "Pusty wektor tempa (obsłużono)": pass # Oczekiwany błąd, już obsłużony
        else: print(f"Błąd podczas analizy tempa: {e}") 
        if 'Tempo (BPM)' not in results: results['Tempo (BPM)'] = "BŁĄD"

    # Krok 3B: Analiza Tonacji (Key)
    try:
        # Definicja profili chromatycznych (tzw. Krumhansl-Schmuckler key profiles).
        # Są to "idealne" rozkłady energii w 12 klasach wysokości dźwięku (chroma)
        # dla tonacji durowej (major) i molowej (minor).
        # Profil Dur (major_profile): Wskazuje na silne szczyty dla prymy (C), tercji wielkiej (E) i kwinty (G).
        major_profile=np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        # Profil Moll (minor_profile): Wskazuje na silne piki dla prymy (C), tercji małej (D#) i kwinty (G).
        minor_profile=np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        notes =                 [ 'C', 'C♯',  'D', 'D♯',  'E',  'F', 'F♯',  'G', 'G♯',  'A', 'A♯',  'B']
        
        print("   Info o tonacji: Używam sygnału harmonicznego (HPSS)...")
        
        # Obliczenie chromagramu (rozkład energii na 12 klas dźwięku)
        chroma = librosa.feature.chroma_cqt(y=harmonic_signal, sr=sample_rate)
        if chroma.shape[1]==0: raise Exception("Pusty chromagram (np. cisza)")
        
        # Uśrednienie chromagramu w czasie, aby uzyskać ogólny profil utworu
        mean_chroma_profile = np.mean(chroma, axis=1)
        profile_sum = np.sum(mean_chroma_profile)
        if profile_sum <= 0: raise Exception("Pusty profil harmoniczny (cisza)")
        
        # Normalizacja profilu
        normalized_profile = mean_chroma_profile / profile_sum
        
        all_correlations_dict = {} 
        # Pętla przez wszystkie 12 możliwych tonacji (C, C#, D, ...)
        for i in range(12): 
            # "Obracamy" profil utworu, aby symulować transpozycję do C
            rolled_profile = np.roll(normalized_profile, -i)
            
            # Obliczamy korelację Pearsona (jak bardzo profil pasuje)
            # z idealnym profilem Durowym (w tonacji C)
            major_correlation = scipy.stats.pearsonr(rolled_profile, major_profile)[0]
            # ... i Molowym (w tonacji C)
            minor_correlation = scipy.stats.pearsonr(rolled_profile, minor_profile)[0]
            
            # Zapisujemy wyniki korelacji dla danej tonacji
            all_correlations_dict[f"{notes[i]} maj"] = major_correlation
            all_correlations_dict[f"{notes[i]} min"] = minor_correlation
            
        if target_key:
            # Jeśli mamy cel (np. "E min")
            target_key_std = normalize_key_name(target_key) 
            if target_key_std in all_correlations_dict:
                # Pobieramy obliczoną korelację dla tej konkretnej tonacji
                target_correlation = all_correlations_dict[target_key_std]
                # Przeliczamy współczynnik korelacji (zakres -1 do 1) na procenty (zakres 0-100)
                match_percentage = max(0, target_correlation) * 100
                results['Zgodność Tonacji (%)'] = round(match_percentage, 2)
                print(f"   Info o tonacji: Cel '{target_key_std}' pasuje w {match_percentage:.2f}% (wsp. korelacji r={target_correlation:.4f})")
            else:
                # Jeśli nazwa tonacji w config.json jest błędna
                print(f"   Info o tonacji: Błędna nazwa tonacji docelowej '{target_key}' w config.json.")
                results['Zgodność Tonacji (%)'] = "Błędny cel"
        else:
            # Jeśli scenariusz nie definiuje tonacji docelowej
            print("   Info o tonacji: Brak celu w config.json. Pomijam obliczanie % zgodności.")
            results['Zgodność Tonacji (%)'] = "Brak celu"
            
    except Exception as e:
        print(f"Błąd podczas analizy tonacji: {e}")
        results['Zgodność Tonacji (%)'] = "BŁĄD"

    # Krok 3C: Analiza Metryk Głośności (LUFS, Peak, PLR) i Stereo
    try:
        # Inicjalizacja miernika głośności (zgodnego z ITU-R BS.1770)
        # Block_size=0.400s (400ms) jest standardem dla pomiaru "Momentary Loudness"
        meter = pyln.Meter(sample_rate, block_size=0.400) 
        
        # Przygotowanie danych dla miernika:
        # Wymagany format to [próbka, kanał].
        # Dla mono: [próbka, 1] (używamy np.newaxis)
        # Dla stereo: [próbka, 2] (używamy wcześniej przygotowanego 'analysis_signal_transposed')
        loudness_data = analysis_signal_transposed if is_stereo else mono_signal[:, np.newaxis]
        
        # Pyloudnorm wymaga co najmniej jednego bloku 400ms do analizy
        if loudness_data.shape[0] < sample_rate * 0.4:
            print("   Ostrzeżenie: Plik zbyt krótki (<400ms) do pomiaru LUFS. Ustawiam -inf.")
            lufs = -np.inf
        else:
            # Pomiar zintegrowanej głośności (Integrated Loudness)
            lufs = meter.integrated_loudness(loudness_data)
        
        # Ograniczamy dolny zakres LUFS, aby uniknąć skrajnych wartości dla ciszy
        results['Głośność (LUFS)'] = round(max(-70.0, lufs), 2) if np.isfinite(lufs) else -70.0

        # Pomiar szczytu (Sample Peak)
        # Znajdujemy próbkę o maksymalnej wartości bezwzględnej w całym sygnale
        sample_peak_amplitude = np.max(np.abs(analysis_signal))
        # Konwertujemy na dBFS
        sample_peak_dbfs = amplitude_to_dbfs(sample_peak_amplitude)
        results['Szczyt (Peak dBFS)'] = round(max(-70.0, sample_peak_dbfs), 2) if np.isfinite(sample_peak_dbfs) else -70.0
        
        # Pomiar Dynamiki (PLR - Peak-to-Loudness Ratio)
        # Jest to prosta miara różnicy między szczytem a średnią głośnością
        peak_value_for_plr = results.get('Szczyt (Peak dBFS)')
        if isinstance(peak_value_for_plr, (int, float)) and isinstance(results['Głośność (LUFS)'], (int, float)):
            plr = peak_value_for_plr - results['Głośność (LUFS)']
            results['Dynamika (PLR dB)'] = round(plr, 2)
        else:
            results['Dynamika (PLR dB)'] = "BŁĄD"

        # Pomiar Korelacji Stereo
        if is_stereo:
            y_left = analysis_signal[0, :]; y_right = analysis_signal[1, :]
            # Obsługa przypadku idealnej ciszy (uniknięcie dzielenia przez zero w np.corrcoef)
            if np.all(y_left == 0) and np.all(y_right == 0): 
                correlation_coefficient = 1.0 # Przyjmujemy, że cisza jest idealnie "mono"
            else:
                # Obliczenie macierzy korelacji między kanałem lewym a prawym
                correlation_matrix = np.corrcoef(y_left, y_right)
                correlation_coefficient = correlation_matrix[0, 1]
                # Obsługa błędów numerycznych (NaN)
                if np.isnan(correlation_coefficient): correlation_coefficient = 0.0
            results['Korelacja Stereo (-1 do 1)'] = round(correlation_coefficient, 3)
        else:
            # Plik mono jest z definicji idealnie skorelowany
            results['Korelacja Stereo (-1 do 1)'] = 1.0
            
        print("   Info o głośności: Pomiary LUFS/Peak/Stereo... Zakończone pomyślnie.")
            
    except Exception as e:
        print(f"Krytyczny błąd podczas analizy metryk głośności/stereo: {e}")
        # W przypadku błędu w tej sekcji, ustawiamy wszystkie powiązane metryki jako błędne
        for key in ['Głośność (LUFS)', 'Szczyt (Peak dBFS)', 'Dynamika (PLR dB)', 'Korelacja Stereo (-1 do 1)']:
            if key not in results: results[key] = "BŁĄD"

    return results

# --- Funkcja Systemu Ocen (Punktacji) --- #

def calculate_score(raw_results_data):
    """
    Przetwarza słownik surowych danych (z analyze_file) na słownik ocen (punktów 0-2).
    Wykorzystuje stałe (np. TEMPO_P1_THRESHOLD) zdefiniowane na początku skryptu.
    """
    scores = {}
    total_score = 0
    
    # Kopiowanie danych identyfikacyjnych do arkusza ocen
    scores['Model'] = raw_results_data.get('Model')
    scores['Prompt (Scenariusz)'] = raw_results_data.get('Prompt (Scenariusz)')
    scores['Nazwa Pliku'] = raw_results_data.get('Nazwa Pliku')
    scores['Ścieżka'] = raw_results_data.get('Ścieżka')

    # 1. Długość (s)
    points = 0
    calculated_duration = raw_results_data.get('Długość (s)')
    if isinstance(calculated_duration, (int, float)):
        error = abs(calculated_duration - DURATION_TARGET)
        if error <= DURATION_P1_THRESHOLD:
            points = 2
        elif error <= DURATION_P2_THRESHOLD:
            points = 1
    scores['Ocena Długość (0-2)'] = points
    total_score += points

    # 2. Tempo (BPM)
    points = 0
    calculated_value = raw_results_data.get('Tempo (BPM)')
    target_value = raw_results_data.get('Tempo (Cel)') # Pobieramy cel zapisany przez analyze_file
    # Sprawdzamy, czy obie wartości są poprawnymi liczbami
    if isinstance(calculated_value, (int, float)) and isinstance(target_value, (int, float)):
        error = abs(calculated_value - target_value)
        if error <= TEMPO_P1_THRESHOLD:
            points = 2
        elif error <= TEMPO_P2_THRESHOLD:
            points = 1
    scores['Ocena Tempo (0-2)'] = points
    total_score += points

    # 3. Tonacja (Zgodność)
    points = 0
    match_percentage = raw_results_data.get('Zgodność Tonacji (%)')
    if isinstance(match_percentage, (int, float)):
        if match_percentage >= KEY_P1_THRESHOLD:
            points = 2
        elif match_percentage >= KEY_P2_THRESHOLD:
            points = 1
    scores['Ocena Tonacja (0-2)'] = points
    total_score += points

    # 4. Głośność (LUFS)
    points = 0
    loudness = raw_results_data.get('Głośność (LUFS)')
    if isinstance(loudness, (int, float)):
        if LUFS_P1_RANGE[0] <= loudness <= LUFS_P1_RANGE[1]:
            points = 2
        elif LUFS_P2_RANGE[0] <= loudness <= LUFS_P2_RANGE[1]:
            points = 1
    scores['Ocena Głośność (0-2)'] = points
    total_score += points

    # 5. Ocena Szczyt (Peak dBFS)
    points = 0
    peak = raw_results_data.get('Szczyt (Peak dBFS)')
    if isinstance(peak, (int, float)):
        if peak <= PEAK_P1_THRESHOLD:
            points = 2
        elif peak <= PEAK_P2_THRESHOLD:
            points = 1
    scores['Ocena Szczyt (0-2)'] = points
    total_score += points

    # 6. Ocena Dynamika (PLR dB)
    points = 0
    dynamics = raw_results_data.get('Dynamika (PLR dB)')
    if isinstance(dynamics, (int, float)):
        if PLR_P1_RANGE[0] <= dynamics <= PLR_P1_RANGE[1]:
            points = 2
        elif PLR_P2_RANGE[0] <= dynamics <= PLR_P2_RANGE[1]:
            points = 1
    scores['Ocena Dynamika (0-2)'] = points
    total_score += points

    # 7. Ocena Korelacja Stereo
    points = 0
    correlation = raw_results_data.get('Korelacja Stereo (-1 do 1)')
    if isinstance(correlation, (int, float)):
        if STEREO_P1_RANGE[0] <= correlation <= STEREO_P1_RANGE[1]:
            points = 2
        elif STEREO_P2_RANGE[0] <= correlation <= STEREO_P2_RANGE[1]:
            points = 1
    scores['Ocena Stereo (0-2)'] = points
    total_score += points
    
    # Zapisanie sumy punktów (Maksymalnie 7 metryk * 2 punkty = 14)
    scores['SUMA (0-14)'] = total_score
    
    return scores


# --- Funkcje Zapisu do Plików Wyjściowych --- #

def save_analysis_results(data, output_file):
    """
    Zapisuje listę słowników z surowymi wynikami analizy (Plik 1) do pliku Excel.
    """
    if not data: return # Nie rób nic, jeśli lista jest pusta
    
    # Konwersja listy słowników na obiekt DataFrame biblioteki Pandas
    df = pd.DataFrame(data)
    
    # Definicja stałej kolejności kolumn dla czytelności raportu
    column_order = [
        'Model',
        'Prompt (Scenariusz)',
        'Nazwa Pliku', 
        'Długość (s)',
        'Tempo (BPM)', 
        'Zgodność Tonacji (%)',
        'Głośność (LUFS)', 
        'Szczyt (Peak dBFS)',
        'Dynamika (PLR dB)', 
        'Korelacja Stereo (-1 do 1)', 
        'Ścieżka'
    ]
    
    # Filtrujemy listę kolumn, aby upewnić się, że wszystkie istnieją w DataFrame
    # (na wypadek, gdyby jakaś kolumna nie została wygenerowana)
    final_columns = [col for col in column_order if col in df.columns]
    df = df[final_columns]

    try:
        print(f"\nTworzenie/Nadpisywanie pliku surowych danych: {output_file}")
        # Zapis do pliku Excel (wymaga 'openpyxl')
        df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"Zapisano pomyślnie. Znaleziono {len(df)} wyników.")
    except Exception as e:
        print(f"Wystąpił błąd podczas zapisu do pliku Excel (surowe dane): {e}")

def save_score_results(score_data, output_file):
    """
    Zapisuje listę słowników z wynikami punktowymi (Plik 2) do pliku Excel.
    """
    if not score_data: return
    
    df = pd.DataFrame(score_data)
    
    # Definicja stałej kolejności kolumn dla arkusza ocen
    column_order = [
        'Model',
        'Prompt (Scenariusz)',
        'Nazwa Pliku', 
        'Ocena Długość (0-2)',
        'Ocena Tempo (0-2)', 
        'Ocena Tonacja (0-2)',
        'Ocena Głośność (0-2)', 
        'Ocena Szczyt (0-2)',
        'Ocena Dynamika (0-2)', 
        'Ocena Stereo (0-2)', 
        'SUMA (0-14)',
        'Ścieżka'
    ]
    
    final_columns = [col for col in column_order if col in df.columns]
    df = df[final_columns]

    try:
        print(f"\nTworzenie/Nadpisywanie pliku ocen (punktów): {output_file}")
        df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"Zapisano pomyślnie. Znaleziono {len(df)} ocen.")
    except Exception as e:
        print(f"Wystąpił błąd podczas zapisu do pliku Excel (oceny): {e}")

# --- Główny Punkt Wykonawczy Skryptu --- #
if __name__ == "__main__":
    
    # Definicja głównych ścieżek
    SAMPLES_FOLDER = 'samples'
    CONFIG_FILE = 'config.json'
    
    # Generowanie unikalnych nazw plików wyjściowych na podstawie aktualnego czasu
    current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    RAW_DATA_OUTPUT_FILE = f'analiza_danych_{current_timestamp}.xlsx'
    SCORES_OUTPUT_FILE = f'ocena_modeli_{current_timestamp}.xlsx'
    
    # Inicjalizacja list do zbierania wyników z pętli
    all_results = []
    all_scores = []

    # Krok 1: Walidacja folderu 'samples'
    if not os.path.exists(SAMPLES_FOLDER):
        print(f"BŁĄD: Wymagany folder '{SAMPLES_FOLDER}' nie istnieje.")
        print(f"Tworzę folder '{SAMPLES_FOLDER}'.")
        os.makedirs(SAMPLES_FOLDER)
        print(f"Proszę umieścić pliki w strukturze: {SAMPLES_FOLDER}/[NazwaModelu]/[NazwaScenariusza]/plik.wav")
        exit()

    # Krok 2: Wczytanie pliku konfiguracyjnego
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            scenario_configurations = json.load(f)
        print(f"Pomyślnie wczytano konfigurację scenariuszy z '{CONFIG_FILE}'.")
    except FileNotFoundError:
        print(f"BŁĄD: Nie znaleziono pliku konfiguracyjnego '{CONFIG_FILE}'.")
        exit()
    except json.JSONDecodeError:
        print(f"BŁĄD KRYTYCZNY: Plik '{CONFIG_FILE}' jest uszkodzony (nieprawidłowy format JSON). Popraw go i spróbuj ponownie.")
        exit()

    print("--- Rozpoczynam Analizę Modeli AI ---")

    # Krok 3: Główna pętla przetwarzania - iteracja przez strukturę folderów
    # os.walk przechodzi przez drzewo katalogów
    # Oczekiwana struktura:
    # samples/
    # ├── Model_A/
    # │   ├── Lofi/
    # │   │   ├── Sample1.wav
    # │   │   └── Sample2.wav
    # │   └── Techno/
    # │       └── Sample1.wav
    # └── Model_B/
    #     └── Lofi/
    #         └── Sample1.wav
    
    for root, dirs, files in os.walk(SAMPLES_FOLDER, topdown=True):
        
        # Ignorujemy główny folder (SAMPLES_FOLDER)
        if root == SAMPLES_FOLDER:
            print(f"Znaleziono następujące foldery modeli do analizy: {dirs}")
            continue

        # Obliczanie ścieżki względnej (np. "Model_A/Lofi")
        relative_path = os.path.relpath(root, SAMPLES_FOLDER)
        path_parts = relative_path.split(os.sep)
        
        # Oczekujemy dokładnie dwóch poziomów: Model i Scenariusz
        if len(path_parts) == 2:
            model_name, scenario_name = path_parts
            print(f"\n--- Przetwarzam: Model={model_name}, Scenariusz={scenario_name} ---")

            # Sprawdzenie, czy dla danego scenariusza (folderu) istnieje konfiguracja w config.json
            if scenario_name not in scenario_configurations:
                print(f"OSTRZEŻENIE: Brak konfiguracji dla scenariusza '{scenario_name}' w pliku {CONFIG_FILE}.")
                print("         Analiza dla tego folderu będzie kontynuowana bez celów (BPM/Tonacja).")
                scenario_config = {} # Używamy pustej konfiguracji
            else:
                scenario_config = scenario_configurations[scenario_name]

            # Filtrowanie plików audio
            audio_files = [f for f in files if f.endswith(('.mp3', '.wav', '.flac', '.ogg'))]
            if not audio_files:
                print("Nie znaleziono plików audio w tym folderze.")
                continue
            
            print(f"Znaleziono {len(audio_files)} plików audio do analizy.")
            
            # Przetwarzanie każdego pliku audio w folderze scenariusza
            for file_name in audio_files:
                file_path = os.path.join(root, file_name)
                
                # 1. Uruchomienie głównej analizy
                analysis_result = analyze_file(file_path, model_name, scenario_name, scenario_config)
                
                if analysis_result:
                    # 2. Dodanie surowych wyników do listy
                    all_results.append(analysis_result)
                    
                    # 3. Obliczenie punktów i dodanie do listy ocen
                    score_result = calculate_score(analysis_result)
                    all_scores.append(score_result)
        
        elif len(path_parts) == 1:
            # To jest folder pierwszego poziomu (np. "Model_A").
            # Przechodzimy dalej, aby przetworzyć jego podfoldery (scenariusze).
            pass 
        else:
            # Obsługa nietypowej struktury folderów (np. zbyt głębokiej)
            if root != SAMPLES_FOLDER:
                print(f"Ignoruję nieoczekiwaną ścieżkę (zbyt głęboka?): {root}")

    print("\n--- Analiza wszystkich folderów została zakończona ---")
    
    # Krok 4: Zapisanie zebranych wyników do plików Excel
    
    # Zapis surowych danych
    if all_results:
        save_analysis_results(all_results, RAW_DATA_OUTPUT_FILE)
    else:
        print("Nie zebrano żadnych wyników (surowe dane). Plik nie został utworzony.")
        
    # Zapis ocen (punktów)
    if all_scores:
        save_score_results(all_scores, SCORES_OUTPUT_FILE)
    else:
        print("Nie zebrano żadnych wyników (oceny). Plik nie został utworzony.")

    print("\n--- Zakończono ---")