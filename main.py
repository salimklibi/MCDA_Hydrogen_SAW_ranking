import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import os

# =========================
# PARAMÈTRES GLOBAUX (SAW pour simplicité, sans pymcdm)
# =========================
METHOD = "SAW"  # "SAW" (utilité additive min-max)

alternatives = [
    "Scénario 1 : H2 100% Vendée (type A)",
    "Scénario 2 : H2 Vendée + import régional (type B)",
    "Scénario 3 : Import H2 national/UE (type C)",
]

decision_matrix_H1 = np.array([
    # coût   GES   emploi   acceptabilité
    [6.0, 1.0, 120.0, 4.5],  # A
    [5.0, 1.5, 80.0, 4.0],  # B
    [4.0, 3.0, 30.0, 3.0],  # C
], dtype=float)

types = np.array([-1, -1, 1, 1], dtype=int)  # coût(-1), bénéfice(1)

actors = {
    "Industriels_Finance": np.array([0.35, 0.30, 0.20, 0.15]),
    "Collectivites": np.array([0.20, 0.30, 0.25, 0.25]),
    "ONG_Locales": np.array([0.10, 0.50, 0.15, 0.25]),
    "Autorites": np.array([0.25, 0.30, 0.20, 0.25]),
    "Scientifiques": np.array([0.25, 0.25, 0.25, 0.25]),
}
for k, w in actors.items():
    actors[k] = w / w.sum()  # Normalisation


# =========================
# HORIZONS TEMPORELS
# =========================
def build_horizons(M_H1):
    M_H1 = M_H1.copy()

    # H20
    M_H20 = M_H1.copy()
    M_H20[0, 0] *= 0.85;
    M_H20[1, 0] *= 0.90;
    M_H20[2, 0] *= 0.95  # Coûts ↓
    M_H20[0, 1] *= 0.90;
    M_H20[1, 1] *= 0.90;
    M_H20[2, 1] *= 0.60  # GES ↓
    M_H20[0, 2] *= 1.05;
    M_H20[1, 2] *= 1.20;
    M_H20[2, 2] *= 1.10  # Emploi ↑
    M_H20[0, 3] *= 1.05;
    M_H20[1, 3] *= 1.15;
    M_H20[2, 3] *= 1.02  # Accept ↑

    # H50
    M_H50 = M_H20.copy()
    M_H50[0, 0] *= 0.90;
    M_H50[1, 0] *= 0.90;
    M_H50[2, 0] *= 0.80
    M_H50[0, 1] *= 0.80;
    M_H50[1, 1] *= 0.75;
    M_H50[2, 1] *= 0.70
    M_H50[0, 2] *= 1.05;
    M_H50[1, 2] *= 1.15;
    M_H50[2, 2] *= 1.20
    M_H50[0, 3] *= 1.05;
    M_H50[1, 3] *= 1.10;
    M_H50[2, 3] *= 1.10

    # H100
    M_H100 = M_H50.copy()
    M_H100[0, 0] *= 0.90;
    M_H100[1, 0] *= 0.90;
    M_H100[2, 0] *= 0.90
    M_H100[0, 1] *= 0.70;
    M_H100[1, 1] *= 0.70;
    M_H100[2, 1] *= 0.70
    M_H100[0, 2] *= 1.05;
    M_H100[1, 2] *= 1.05;
    M_H100[2, 2] *= 1.10
    M_H100[0, 3] *= 1.05;
    M_H100[1, 3] *= 1.05;
    M_H100[2, 3] *= 1.15

    return {"H1": M_H1, "H20": M_H20, "H50": M_H50, "H100": M_H100}


horizons = build_horizons(decision_matrix_H1)


# =========================
# MÉTHODE SAW (MAUT additive)
# =========================
def run_saw(M, w, t):
    M_norm = M.astype(float).copy()
    for j in range(M_norm.shape[1]):
        col = M_norm[:, j]
        if col.max() == col.min():
            M_norm[:, j] = 0.0
            continue
        if t[j] == 1:  # bénéfice
            M_norm[:, j] = (col - col.min()) / (col.max() - col.min())
        else:  # coût
            M_norm[:, j] = (col.max() - col) / (col.max() - col.min())
    prefs = M_norm @ w
    ranks = np.argsort(-prefs).astype(int) + 1  # 1=meilleur
    return prefs, ranks


# =========================
# 1. VALIDATION RÉSULTATS
# =========================
print("=== VALIDATION RÉSULTATS SAW Multi-Acteurs / Horizons ===")
results = []
for h_name, M in horizons.items():
    for actor_name, weights in actors.items():
        prefs, ranks = run_saw(M, weights, types)
        for i, (alt, pref, rank) in enumerate(zip(alternatives, prefs, ranks)):
            results.append({
                'Horizon': h_name, 'Acteur': actor_name, 'Scénario': alt,
                'Score': pref, 'Rang': rank, 'Scénario_idx': i + 1
            })

df_results = pd.DataFrame(results)
print(df_results.groupby(['Horizon', 'Acteur'])[['Rang', 'Score']].first().round(4))

winners = df_results[df_results['Rang'] == 1].groupby('Horizon')['Scénario'].value_counts().unstack(fill_value=0)
print("\nGagnants par horizon:\n", winners)

# =========================
# 2. SENSIBILITÉ MONTE CARLO (1000 simus, ±20%)
# =========================
np.random.seed(42)
n_sims = 1000
sens_results = []
for h_name, M in horizons.items():
    base_ranks = {actor: run_saw(M, actors[actor], types)[1] for actor in actors}
    for sim in range(n_sims):
        for actor_name in actors:
            w_pert = actors[actor_name] * np.random.normal(1, 0.2, 4)
            w_pert /= w_pert.sum()
            _, ranks_sim = run_saw(M, w_pert, types)
            for i, rank in enumerate(ranks_sim):
                sens_results.append({
                    'Horizon': h_name, 'Acteur': actor_name, 'Sim': sim,
                    'Scénario_idx': i + 1, 'Rang_sim': rank, 'Base_rang': base_ranks[actor_name][i]
                })

df_sens = pd.DataFrame(sens_results)
stability = df_sens.groupby(['Horizon', 'Acteur', 'Scénario_idx']).agg(
    mean_rang=('Rang_sim', 'mean'), std_rang=('Rang_sim', 'std')
).round(4)
print("\nStabilité Monte Carlo (A=1,B=2,C=3):\n", stability.head(12))

# =========================
# 3. EXPORTS + DATA VIZ
# =========================
df_results.to_csv('mcda_results.csv', index=False)
df_sens.to_csv('mcda_sensitivity.csv', index=False)
print("\n✅ Exports: mcda_results.csv | mcda_sensitivity.csv")

pivot_scores = df_results.pivot_table('Score', ['Horizon', 'Acteur'], 'Scénario', 'mean').round(4)
print("\n📊 Data HEATMAP:\n", pivot_scores.head())

pivot_ranks = df_results.pivot_table('Rang', 'Horizon', 'Acteur', 'mean').round(2)
print("\n📈 Data ÉVOLUTION Rangs:\n", pivot_ranks.T)

print("\n🎯 Radar ex. H1 Industriels (normalisé):")
M_ex = horizons['H1']
norm_ex = M_ex / M_ex.max(0)
for i, alt in enumerate(alternatives):
    print(f"{alt}: {dict(zip(['Coût', 'GES', 'Emploi', 'Accept'], norm_ex[i].round(2)))}")

# =========================
# VISUALISATIONS AVANCÉES (CORRIGÉ)
# =========================
plt.style.use('default')
sns.set_palette("husl")

# Dossier output
os.makedirs('output_mcda', exist_ok=True)

# 1. HEATMAP SCORES
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_scores, annot=True, cmap='YlGnBu', fmt='.3f',
            cbar_kws={'label': 'Score SAW'})
plt.title('Heatmap Scores SAW - Horizons x Acteurs vs Scénarios')
plt.tight_layout()
plt.savefig('output_mcda/heatmap_scores.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. TIMELINE RANGs
hor_order = ['H1', 'H20', 'H50', 'H100']
pivot_ranks_ordered = pivot_ranks.reindex(hor_order)
plt.figure(figsize=(10, 6))
for acteur in pivot_ranks.columns:
    plt.plot(hor_order, pivot_ranks_ordered[acteur], marker='o', label=acteur, linewidth=2.5)
plt.title('Évolution Rangs Moyens par Acteur (H1 → H100)')
plt.xlabel('Horizon'); plt.ylabel('Rang moyen'); plt.legend()
plt.gca().invert_yaxis()  # Rang 1 en haut
plt.grid(True, alpha=0.3)
plt.savefig('output_mcda/timeline_rangs.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. RADARS PLOTLY H1 (basé sur tes résultats parfaits !)
fig = make_subplots(rows=1, cols=1, specs=[[{"type": "polar"}]],
                    subplot_titles=['Radars H1 - Tous Scénarios'])
criteria = ['Coût', 'GES', 'Emploi', 'Acceptabilité']

# Données de tes radars (normalisées H1)
radar_data = {
    'A': [1.00, 0.33, 1.00, 1.00],
    'B': [0.83, 0.50, 0.67, 0.89],
    'C': [0.67, 1.00, 0.25, 0.67]
}

colors = ['red', 'blue', 'green']
for i, (scen, values) in enumerate(radar_data.items()):
    fig.add_trace(go.Scatterpolar(
        r=values, theta=criteria, fill='toself',
        name=f'Scénario {scen}', line_color=colors[i],
        opacity=0.7
    ))

fig.update_layout(
    title="🎯 Radars Normalisés Horizon H1 (A dominant coût/emploi)",
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    height=600, showlegend=True,
    font=dict(size=12)
)
fig.write_html('output_mcda/radars_h1_interactif.html')
print("✅ Radar HTML sauvé ! Ouvre output_mcda/radars_h1_interactif.html")

# 4. HEATMAP PLOTLY
fig_hm = px.imshow(pivot_scores.values,
                   x=pivot_scores.columns.str.replace('Scénario ', ''),
                   y=pivot_scores.index.get_level_values(0) + '_' + pivot_scores.index.get_level_values(1),
                   color_continuous_scale='YlGnBu',
                   title='Heatmap Interactive Scores SAW')
fig_hm.write_html('output_mcda/heatmap_plotly.html')
print("✅ Heatmap interactive sauvée !")

print("\n🎉 TOUTES VIZ SAUVÉES dans output_mcda/ :")
print("- heatmap_scores.png")
print("- timeline_rangs.png")
print("- radars_h1_interactif.html (hover !)")
print("- heatmap_plotly.html")