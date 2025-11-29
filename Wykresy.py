import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import pi

def load_data(filepath):
    return pd.read_csv(filepath)

def plot_global_pie_chart(df):
    counts = df.iloc[:, 1:].sum(axis=1)
    labels = df.iloc[:, 0]
    colors = ["#B8DAED", "#8dc2e6", "#6cafee", "#5097DD", "#226cc0ff"]
    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
    plt.axis('equal')
    plt.show()

def prepare_long_data(df):
    """
    Funkcja przekształca tabelę z licznikami
    na tabelę surową (lista wszystkich pojedynczych ocen)
    """
    records = []
    for col in df.columns[1:]:
        parts = col.split(';')
        if len(parts) != 3:
            continue
        app, scenario, criterion = parts
        
        for index, row in df.iterrows():
            rating_value = row['Ocena']
            count = row[col]
            if count > 0:
                records.extend([{
                    'Aplikacja': app,
                    'Scenariusz': scenario,
                    'Kryterium': criterion,
                    'Ocena': rating_value
                }] * int(count))
                
    return pd.DataFrame(records)
def plot_radar_chart(df_long):
    """
    Tworzy wykres radarowy (pajączkowy) porównujący średnie oceny aplikacji
    względem różnych kryteriów (Melodyjność, Miks, etc.)
    """
    # 1. Obliczamy średnią ocenę dla każdej pary Aplikacja-Kryterium
    # Filtrujemy, żeby nie brać 'Ocena ogólna' do radaru cech szczegółowych, albo bierzemy wszystko
    # Zazwyczaj na radarze dajemy cechy składowe (bez Oceny ogólnej, albo z nią jako jedną z osi)
    # Tu weźmiemy wszystko oprócz 'Ocena ogólna' dla czystości, chyba że chcesz inaczej.
    
    avg_scores = df_long[df_long['Kryterium'] != 'Ocena ogólna'].groupby(['Aplikacja', 'Kryterium'])['Ocena'].mean().reset_index()
    
    # Lista kategorii (osi wykresu)
    categories = list(avg_scores['Kryterium'].unique())
    N = len(categories)
    
    # Kąty dla osi
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # domknięcie pętli
    
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    
    # Ustawienie osi
    plt.xticks(angles[:-1], categories, color='grey', size=10)
    
    # Oś Y (skala ocen 1-5)
    ax.set_rlabel_position(0)
    plt.yticks([1, 2, 3, 4, 5], ["1","2","3","4","5"], color="grey", size=7)
    plt.ylim(0, 5)
    
    # Lista aplikacji do narysowania
    apps = df_long['Aplikacja'].unique()
    
    # Kolory dla aplikacji (dla spójności z poprzednim wykresem)
    colors = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00'] 
    
    for i, app in enumerate(apps):
        # Wyciągamy dane dla konkretnej apki
        values = avg_scores[avg_scores['Aplikacja'] == app]['Ocena'].tolist()
        
        # Czasami kolejność sortowania może się pomieszać, więc wymuszamy porządek wg listy categories
        app_data = avg_scores[avg_scores['Aplikacja'] == app].set_index('Kryterium')
        values = [app_data.loc[cat]['Ocena'] for cat in categories]
        
        values += values[:1] # domknięcie pętli
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=app, color=colors[i % len(colors)])
        ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.1)
    
    plt.title('Profil jakościowy aplikacji (średnia ocen)', size=15, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.savefig('radar_chart_comparison.png')
    plt.show()

def main():
    file_path = 'Wyniki.csv'
    df = load_data(file_path)
    df_long = prepare_long_data(df)

    plot_radar_chart(df_long)
    # plot_global_pie_chart(df)

if __name__ == "__main__":
    main()