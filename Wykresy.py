import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import pi
import seaborn as sns

COLORS_LIKERT = ["#B8DAED", "#8dc2e6", "#6cafee", "#5097DD", "#226cc0ff"]
COLORS_APPS = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00']
COLORS_SCENARIOS = ['#ff9999', '#66b3ff', '#99ff99']

def load_data(filepath):
    return pd.read_csv(filepath)

def get_detailed_metrics(df):
    metrics = []
    for col in df.columns[1:]:
        parts = col.split(';')
        if len(parts) != 3:
            continue
        app, scenario, criterion = parts
        
        weighted_sum = (df['Ocena'] * df[col]).sum()
        total_count = df[col].sum()
        mean_score = weighted_sum / total_count if total_count > 0 else 0
        
        metrics.append({
            'Aplikacja': app,
            'Scenariusz': scenario,
            'Kryterium': criterion,
            'Srednia': mean_score
        })
    return pd.DataFrame(metrics)

def plot_global_pie_chart(df):
    counts = df.iloc[:, 1:].sum(axis=1)
    labels = df.iloc[:, 0]
    
    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=COLORS_LIKERT, startangle=140)
    plt.axis('equal')
    plt.title('Globalny rozkład ocen')
    plt.tight_layout()
    plt.show()

def plot_stacked_bar_chart(df):
    df_indexed = df.set_index('Ocena')
    app_counts = pd.DataFrame(index=df_indexed.index)
    app_names = set(col.split(';')[0] for col in df_indexed.columns)
    
    for app in app_names:
        app_cols = [c for c in df_indexed.columns if c.startswith(app + ';')]
        app_counts[app] = df_indexed[app_cols].sum(axis=1)
    
    app_pct = app_counts.div(app_counts.sum(axis=0), axis=1) * 100
    app_pct_T = app_pct.T
    
    ax = app_pct_T.plot(kind='bar', stacked=True, color=COLORS_LIKERT, figsize=(10, 6), width=0.7)
    
    plt.title('Struktura ocen dla poszczególnych modeli')
    plt.ylabel('Udział procentowy [%]')
    plt.xlabel('Model')
    plt.xticks(rotation=0)
    
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(reversed(handles), reversed(labels), title='Ocena', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    for c in ax.containers:
        labels_text = [f'{v.get_height():.1f}%' if v.get_height() > 3 else '' for v in c]
        ax.bar_label(c, labels=labels_text, label_type='center', color='black', fontsize=9)
    
    plt.tight_layout()
    plt.show()

def plot_radar_chart(df_metrics):
    df_radar = df_metrics[df_metrics['Kryterium'] != 'Ocena ogólna']
    avg_scores = df_radar.groupby(['Aplikacja', 'Kryterium'])['Srednia'].mean().reset_index()
    
    categories = list(avg_scores['Kryterium'].unique())
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    
    plt.xticks(angles[:-1], categories, color='grey', size=10)
    ax.set_rlabel_position(0)
    plt.yticks([1, 2, 3, 4, 5], ["1","2","3","4","5"], color="grey", size=7)
    plt.ylim(0, 5)
    
    apps = avg_scores['Aplikacja'].unique()
    
    for i, app in enumerate(apps):
        app_data = avg_scores[avg_scores['Aplikacja'] == app].set_index('Kryterium')
        values = [app_data.loc[cat]['Srednia'] for cat in categories]
        values += values[:1]
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=app, color=COLORS_APPS[i % len(COLORS_APPS)])
        ax.fill(angles, values, color=COLORS_APPS[i % len(COLORS_APPS)], alpha=0.1)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Profil jakościowy aplikacji (średnie)')
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df_metrics):
    df_metrics['Case'] = df_metrics['Aplikacja'] + ' - ' + df_metrics['Scenariusz']
    df_pivot = df_metrics.pivot(index='Case', columns='Kryterium', values='Srednia')
    
    corr_matrix = df_pivot.corr(method='spearman')
    cols = [c for c in corr_matrix.columns if c != 'Ocena ogólna'] + ['Ocena ogólna']
    corr_matrix = corr_matrix[cols].loc[cols]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    
    cbar = fig.colorbar(cax, shrink=0.8)
    cbar.ax.set_ylabel('Współczynnik korelacji', rotation=270, labelpad=15)
    
    ticks = np.arange(len(corr_matrix.columns))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(corr_matrix.columns, fontsize=10)
    
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            val = corr_matrix.iloc[i, j]
            text_color = 'white' if abs(val) > 0.6 else 'black'
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=9)
    
    plt.title('Macierz korelacji kryteriów oceny')
    plt.tight_layout()
    plt.show()

def plot_scenario_heatmap(df):
    data = {} 
    
    for col in df.columns[1:]:
        parts = col.split(';')
        if len(parts) != 3:
            continue
            
        _, scenario, criterion = parts
        key = (scenario, criterion)
        
        if key not in data:
            data[key] = {'weighted_sum': 0, 'count': 0}
            
        weighted_sum = (df['Ocena'] * df[col]).sum()
        total_count = df[col].sum()
        
        data[key]['weighted_sum'] += weighted_sum
        data[key]['count'] += total_count
        
    rows = []
    for (scenario, criterion), values in data.items():
        mean = values['weighted_sum'] / values['count'] if values['count'] > 0 else 0
        rows.append({'Scenariusz': scenario, 'Kryterium': criterion, 'Srednia': mean})
        
    df_means = pd.DataFrame(rows)
    
    heatmap_data = df_means.pivot(index='Kryterium', columns='Scenariusz', values='Srednia')
    
    idx = [c for c in heatmap_data.index if c != 'Ocena ogólna'] + ['Ocena ogólna']
    heatmap_data = heatmap_data.reindex(idx)
    
    plt.figure(figsize=(8, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdYlGn", vmin=1, vmax=5, 
                linewidths=.5, cbar_kws={'label': 'Średnia ocena'})
    
    plt.title('Szczegółowa ocena scenariuszy')
    plt.xlabel('Scenariusz')
    plt.ylabel('Kryterium')
    plt.tight_layout()
    plt.show()

def main():
    file_path = 'Wyniki.csv'
    df = load_data(file_path)
    df_metrics = get_detailed_metrics(df)
    
    # plot_global_pie_chart(df)
    # plot_stacked_bar_chart(df)
    # plot_radar_chart(df_metrics)
    # plot_correlation_heatmap(df_metrics)
    plot_scenario_heatmap(df)

if __name__ == "__main__":
    main()