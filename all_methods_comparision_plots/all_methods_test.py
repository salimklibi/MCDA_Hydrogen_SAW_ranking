import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# Suppression des avertissements futurs pour Matplotlib/Seaborn
warnings.simplefilter(action='ignore', category=FutureWarning)

# Imports Plotly (conservés si vous voulez les utiliser plus tard)
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Détection PyMCDM
try:
    from pymcdm.methods import TOPSIS, PROMETHEE_II
    from pymcdm.helpers import rrankdata

    PYMCDM_AVAILABLE = True
    print("✅ PyMCDM installé → TOPSIS & PROMETHEE OK")
except ImportError:
    PYMCDM_AVAILABLE = False
    print("⚠️ PyMCDM absent → SAW uniquement (pip install pymcdm)")

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'figure.dpi': 300, 'savefig.dpi': 400,
    'axes.labelsize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10
})

# =========================
# 1. CHOIX INTERACTIF MÉTHODE MCDA
# =========================
print("\n" + "=" * 60)
print("🛠️  MCDA HYDROGÈNE VENDÉE - CHOIX MÉTHODE")
print("=" * 60)
print("1. SAW     - Simple Additive Weighting")
print("2. TOPSIS  - Technique for Order Preference")
print("3. PROMETHEE - Preference Ranking")
print("0. SAW     - Par défaut")

# Pour l'automatisation du test, on décommente la ligne suivante ou on laisse l'input
# METHOD = "TOPSIS" # Décommenter pour tester direct sans taper
try:
    choice = input("Tape 0/1/2/3 : ").strip().upper()
    METHOD = {"1": "SAW", "2": "TOPSIS", "3": "PROMETHEE", "0": "SAW"}.get(choice, "SAW")
except:
    METHOD = "SAW"

print(f"\n🎯 Méthode : {METHOD}")

# =========================
# 2. PARAMÈTRES GLOBAUX
# =========================
alternatives = [
    "Scénario 1 : H2 100% Vendée (type A)",
    "Scénario 2 : H2 Vendée + import régional (type B)",
    "Scénario 3 : Import H2 national/UE (type C)",
]

decision_matrix_H1 = np.array([
    [6.0, 1.0, 120.0, 4.5],  # coût, GES, emploi, accept.
    [5.0, 1.5, 80.0, 4.0],
    [4.0, 3.0, 30.0, 3.0],
], dtype=float)

types = np.array([-1, -1, 1, 1], dtype=int)  # -1=coût, 1=bénéfice

actors = {
    "Industriels_Finance": np.array([0.35, 0.30, 0.20, 0.15]),
    "Collectivites": np.array([0.20, 0.30, 0.25, 0.25]),
    "ONG_Locales": np.array([0.10, 0.50, 0.15, 0.25]),
    "Autorites": np.array([0.25, 0.30, 0.20, 0.25]),
    "Scientifiques": np.array([0.25, 0.25, 0.25, 0.25]),
}
for k, w in actors.items():
    actors[k] = w / w.sum()


# =========================
# 3. HORIZONS TEMPORELS
# =========================
def build_horizons(M_H1):
    M_H1 = M_H1.copy()
    M_H20 = M_H1.copy()
    M_H20[0, 0] *= 0.85;
    M_H20[1, 0] *= 0.90;
    M_H20[2, 0] *= 0.95
    M_H20[0, 1] *= 0.90;
    M_H20[1, 1] *= 0.90;
    M_H20[2, 1] *= 0.60
    M_H20[0, 2] *= 1.05;
    M_H20[1, 2] *= 1.20;
    M_H20[2, 2] *= 1.10
    M_H20[0, 3] *= 1.05;
    M_H20[1, 3] *= 1.15;
    M_H20[2, 3] *= 1.02

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


# =========================
# 4. MÉTHODES MCDA
# =========================
def run_saw(M, w, t):
    M_norm = M.copy().astype(float)
    for j in range(M_norm.shape[1]):
        col = M_norm[:, j]
        if abs(col.max() - col.min()) < 1e-10:
            M_norm[:, j] = 0.0;
            continue
        if t[j] == 1:
            M_norm[:, j] = (col - col.min()) / (col.max() - col.min())
        else:
            M_norm[:, j] = (col.max() - col) / (col.max() - col.min())
    prefs = M_norm @ w
    ranks = np.argsort(-prefs).astype(int) + 1
    return prefs, ranks


def run_topsis(M, w, t):
    if not PYMCDM_AVAILABLE: raise ImportError("pymcdm requis pour TOPSIS")
    model = TOPSIS()
    prefs = model(M, w, t)
    ranks = rrankdata(prefs)
    return prefs, ranks


def run_promethee(M, w, t):
    if not PYMCDM_AVAILABLE: raise ImportError("pymcdm requis pour PROMETHEE")
    model = PROMETHEE_II("usual")
    prefs = model(M, w, t)
    ranks = (-prefs).argsort().astype(int) + 1
    return prefs, ranks


def run_method(M, w, t, method):
    method = method.upper()
    if method == "SAW":
        return run_saw(M, w, t)
    elif method == "TOPSIS":
        return run_topsis(M, w, t)
    elif method == "PROMETHEE":
        return run_promethee(M, w, t)
    else:
        raise ValueError("SAW/TOPSIS/PROMETHEE uniquement")


# =========================
# 5. CALCULS + VALIDATION
# =========================
horizons = build_horizons(decision_matrix_H1)
os.makedirs('output_mcda', exist_ok=True)

results = []
for h_name, M in horizons.items():
    print(f"⏳ {METHOD} : Horizon {h_name}")
    for actor_name, weights in actors.items():
        prefs, ranks = run_method(M, weights, types, METHOD)
        for i, (alt, pref, rank) in enumerate(zip(alternatives, prefs, ranks)):
            results.append({
                'Horizon': h_name, 'Acteur': actor_name, 'Scénario': alt,
                'Score': float(pref), 'Rang': int(rank), 'Méthode': METHOD
            })

df_results = pd.DataFrame(results)
df_results.to_csv(f'output_mcda/results_{METHOD.lower()}.csv', index=False)

# Résumé gagnants
winners = df_results[df_results['Rang'] == 1].groupby('Horizon')['Scénario'].value_counts()
print("\n🏆 Gagnants par horizon :\n", winners)


# =========================
# 6. VISUALISATIONS (CORRIGÉ)
# =========================

def compare_methods_visu(df_results_local, METHOD):
    methods_list = ['SAW']
    if PYMCDM_AVAILABLE: methods_list.extend(['TOPSIS', 'PROMETHEE'])

    all_results = []
    # Re-générer les données pour toutes les méthodes si nécessaire pour la comparaison
    # ou utiliser df_results_local si elle contient tout (ici elle ne contient que la méthode choisie)
    # Pour simplifier et éviter de recalculer tout si non nécessaire, on travaille avec df_results_local 
    # pour les figures spécifiques à la méthode choisie, et on calcule pour comparaison si besoin.

    # Ici, je génère la figure principale basée sur df_results (Méthode choisie)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'MCDA H2 Vendée : Résultats {METHOD}', fontsize=16, fontweight='bold')

    # a) Heatmap Rang Moyen par Acteur/Horizon
    pivot_avg = df_results_local.pivot_table('Rang', 'Acteur', 'Horizon', 'mean')
    sns.heatmap(pivot_avg, annot=True, cmap='RdYlGn_r', fmt='.1f', center=2,
                square=True, linewidths=0.5, cbar_kws={'shrink': 0.8, 'label': 'Rang (1=meilleur)'},
                ax=axes[0, 0])
    axes[0, 0].set_title(f'Rangs Moyens ({METHOD}) : Acteurs × Horizons', pad=20)

    # b) Boxplot Stabilité par Scénario
    sns.boxplot(data=df_results_local, x='Horizon', y='Rang', hue='Scénario', ax=axes[0, 1])
    axes[0, 1].set_title('Distribution Rangs par Scénario')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # c) Evolution Temporelle (Line plot)
    for scenario in df_results_local['Scénario'].unique():
        subset = df_results_local[df_results_local['Scénario'] == scenario]
        # Moyenne par horizon pour lisser
        avg_scores = subset.groupby('Horizon')['Score'].mean()
        axes[1, 0].plot(avg_scores.index, avg_scores.values, marker='o', label=scenario.split(':')[1].strip())

    axes[1, 0].set_title('Évolution Score Moyen par Horizon')
    axes[1, 0].set_ylabel('Score Normalisé')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # d) Radar Plot (Statique final H100)
    categories = ['Coût', 'GES', 'Emploi', 'Acceptabilité']
    M_h100 = horizons['H100']
    # Normalisation simple pour affichage 0-1
    M_norm = M_h100 / M_h100.max(axis=0)

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    ax_radar = axes[1, 1]
    ax_radar = plt.subplot(2, 2, 4, polar=True)

    for i, alt in enumerate(alternatives):
        values = M_norm[i].tolist()
        values += values[:1]
        ax_radar.plot(angles, values, linewidth=2, label=alt.split(':')[1].strip())
        ax_radar.fill(angles, values, alpha=0.1)

    ax_radar.set_theta_offset(np.pi / 2)
    ax_radar.set_theta_direction(-1)
    ax_radar.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax_radar.set_title('Performance Rel. H100 (Polar)', pad=20)
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.savefig('output_mcda/publication_suite.png', dpi=400, bbox_inches='tight')
    plt.show()


compare_methods_visu(df_results, METHOD)

# =========================
# 7. STABILITÉ MONTE CARLO AVANCÉE
# =========================
np.random.seed(42)
n_sims = 1000  # Réduit à 1000 pour aller plus vite dans ce test, remettez 5000 si voulu
sens_results = []

print(f"🔀 Lancement simulation Monte Carlo ({n_sims} itérations)...")

for h_name, M in horizons.items():
    # Pré-calcul des colonnes pour accélérer
    for sim in range(n_sims):
        for actor_name, base_weights in actors.items():
            # Perturbation log-normale des poids
            w_pert = base_weights * np.random.lognormal(0, 0.2, 4)
            w_pert = w_pert / w_pert.sum()

            prefs_sim, ranks_sim = run_method(M, w_pert, types, METHOD)
            for i, r in enumerate(ranks_sim):
                sens_results.append({
                    'Horizon': h_name, 'Acteur': actor_name, 'Sim': sim,
                    'Scénario_idx': i + 1, 'Rang': r
                })

df_sens = pd.DataFrame(sens_results)

fig_sens, axes_sens = plt.subplots(1, 2, figsize=(14, 5))

# a) CORRECTION ICI : Violinplot avec arguments corrects
# On visualise la distribution des rangs pour chaque scénario à travers les horizons
sns.violinplot(data=df_sens, x='Horizon', y='Rang', hue='Scénario_idx',
               inner=None, alpha=0.7, split=False, ax=axes_sens[0])
axes_sens[0].set_title(f'Stabilité des Rangs ({METHOD} + Monte Carlo)', fontweight='bold')
axes_sens[0].set_ylim(0.5, 3.5)  # Rangs vont de 1 à 3
axes_sens[0].legend(title='Scénario', bbox_to_anchor=(1.05, 1), loc='upper left')

# b) CORRECTION ICI : Calcul de la matrice de corrélation
# On regarde la corrélation des rangs moyens par acteur (ex: sur H20)
df_corr_data = df_sens[df_sens['Horizon'] == 'H20'].groupby(['Acteur', 'Scénario_idx'])['Rang'].mean().unstack()
corr_matrix = df_corr_data.corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
            square=True, ax=axes_sens[1], cbar_kws={'shrink': 0.8},
            vmin=-1, vmax=1)
axes_sens[1].set_title('Corrélation Préférences Acteurs (H20)', fontweight='bold')

plt.tight_layout()
plt.savefig('output_mcda/stabilite_avancee.png', dpi=400)
plt.show()

print("\n✅ Figures générées dans 'output_mcda/'")