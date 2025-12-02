import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

COLORS_TECH = ['#1f78b4'] 

def load_data(filepath):
    return pd.read_csv(filepath)

def analyze_and_plot(df):
    criteria_cols = [c for c in df.columns if 'Ocena' in c and 'SUMA' not in c]
    
    df_criteria = df.groupby('Model')[criteria_cols].mean()
    df_total = df.groupby('Model')['SUMA (0-14)'].mean().sort_values(ascending=False)
    
    clean_cols = [c.replace('Ocena ', '').replace(' (0-2)', '') for c in df_criteria.columns]
    df_criteria.columns = clean_cols

    # --- Heatmap ---
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_criteria, annot=True, cmap="RdYlGn", vmin=0, vmax=2, fmt=".2f",
                linewidths=.5, cbar_kws={'label': 'Średnia ocena (0-2)'})
    plt.title('Mapa zgodności kryteriów ilościowych')
    plt.xlabel('Parametr')
    plt.ylabel('Model')
    plt.tight_layout()
    plt.savefig('Skrypt_heatmap.png')
    plt.show()

    # --- Ranking slupkowy ---
    plt.figure(figsize=(8, 6))
    ax = df_total.plot(kind='bar', color='#4682B4', width=0.6)
    
    plt.title('Ranking ogólny spełnienia kryteriów ilościowych')
    plt.ylabel('Średnia suma punktów')
    plt.xlabel('Model')
    plt.ylim(0, 15)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=0)
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9), 
                    textcoords='offset points', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('Skrypt_ranking.png')
    plt.show()

def main():
    file_path = 'Wyniki_skrypt.csv'
    df = load_data(file_path)
    analyze_and_plot(df)

if __name__ == "__main__":
    main()