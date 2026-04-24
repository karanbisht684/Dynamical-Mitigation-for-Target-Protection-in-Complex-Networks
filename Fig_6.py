import networkx as nx
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt


# =====================================================
# PLOT STYLE
# =====================================================

plt.rcParams['text.usetex'] = False

plt.rcParams.update({
    "font.family": "Computer Modern",
    "font.size": 8,
    "axes.labelsize": 26,
    "axes.titlesize": 18,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 15,
    "legend.frameon": True,
    "lines.linewidth": 1,
    "lines.markersize": 9,
    "mathtext.fontset": "dejavusans",
    "axes.spines.top": True,
    "axes.spines.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 5,
    "ytick.major.size": 5,
})


# =====================================================
# Load LastFM Network
# =====================================================

file_path = 'paste data path here'

df = pd.read_csv(file_path)

G_original = nx.from_pandas_edgelist(df, 'node_1', 'node_2')
G_original = G_original.to_undirected()

G_original = G_original.subgraph(
    max(nx.connected_components(G_original), key=len)
).copy()

print("Nodes:", G_original.number_of_nodes())
print("Edges:", G_original.number_of_edges())


# =====================================================
# Initial TKR
# =====================================================

def manual_initial_TKR(G, target_nodes):

    T = set(target_nodes)

    K = set()
    for t in T:
        K.update(G.neighbors(t))

    K -= T

    R = set(G.nodes) - T - K

    return T, K, R


# =====================================================
# TIA
# =====================================================

def tia_large_graph(G, T, K, R, max_iter=100):

    for _ in range(max_iter):

        prev_T, prev_K, prev_R = set(T), set(K), set(R)

        move_to_T = {node for node in K
                     if not any(neigh in R for neigh in G.neighbors(node))}

        T.update(move_to_T)
        K -= move_to_T

        move_to_K = {node for node in R
                     if any(neigh in T for neigh in G.neighbors(node))}

        K.update(move_to_K)
        R -= move_to_K

        if (T, K, R) == (prev_T, prev_K, prev_R):
            break

    return T, K, R


# =====================================================
# R_connected
# =====================================================

def get_R_connected(G, R_final, K_final):

    return {r for r in R_final if any(neigh in K_final for neigh in G.neighbors(r))}


# =====================================================
# Critical R
# =====================================================

def get_critical_R(G, R_final, R_connected, threshold):

    init_deg = {n: G.degree(n) for n in R_final}

    fragile_R = {
        r for r in R_final
        if init_deg[r] > 0 and (G.degree(r)-1)/init_deg[r] < threshold
    }

    critical = set()

    for r in R_connected:

        if r not in fragile_R:
            continue

        fragile_neighbors = sum(
            1 for nbr in G.neighbors(r)
            if nbr in fragile_R and nbr in R_final
        )

        if fragile_neighbors >= 2:
            critical.add(r)

    return critical


# =====================================================
# Graph coloring reduction
# =====================================================

def avg_color_degree(G):

    colors = nx.coloring.greedy_color(G)

    max_color = max(colors.values())

    avg_color = {}

    for c in range(max_color+1):

        nodes = [n for n in colors if colors[n] == c]

        if nodes:
            avg_color[c] = np.mean([G.degree(n) for n in nodes])
        else:
            avg_color[c] = 0

    return colors, avg_color


def coloring_reduction(G, critical_R):

    colors, avg_color = avg_color_degree(G)

    min_avg = min(avg_color.values())

    reduced = {r for r in critical_R if G.degree(r) > min_avg}

    return reduced


# =====================================================
# R-only cascade
# =====================================================

def cascade_R_only(G_orig, R_isolated, R_connected, shielded_R, threshold):

    G = G_orig.copy()
    init_deg = dict(G_orig.degree())

    failed_R = set()

    candidates = list(R_isolated)

    if not candidates:
        return failed_R

    seeds = random.sample(candidates, min(2, len(candidates)))

    failed_R.update(seeds)

    for s in seeds:
        if s in G:
            G.remove_node(s)

    while True:

        new_fail = set()

        for node in (R_isolated | R_connected) - failed_R:

            neighs = list(G_orig.neighbors(node))

            if any(n in failed_R for n in neighs):

                cur = G.degree(node) if node in G else 0
                ini = init_deg[node]

                if ini > 0 and cur/ini < threshold:
                    new_fail.add(node)

        if not new_fail:
            break

        for n in new_fail:
            if n in G:
                G.remove_node(n)

        failed_R |= new_fail

    return failed_R

# =====================================================
# Simulation parameters
# =====================================================

threshold_values = np.arange(0.80, 0.98, 0.01)

target_fraction = 0.15  #change degree(0.10 or 0.15)
runs = 10

# Store results

surv_high, surv_low, surv_rand = [], [], []

K_high, K_low, K_rand = [], [], []

Rcrit_high, Rcrit_low, Rcrit_rand = [], [], []


# =====================================================
# Simulation Loop
# =====================================================

for threshold in threshold_values:

    surv_runs_high, surv_runs_low, surv_runs_rand = [], [], []

    K_runs_high, K_runs_low, K_runs_rand = [], [], []

    crit_runs_high, crit_runs_low, crit_runs_rand = [], [], []

    for seed in range(runs):

        random.seed(seed)

        G = G_original.copy()

        N = G.number_of_nodes()

        topN = int(N * target_fraction)

        degree_sorted_desc = sorted(G.degree, key=lambda x: x[1], reverse=True)
        degree_sorted_asc = sorted(G.degree, key=lambda x: x[1])

        targets_high = [n for n,_ in degree_sorted_desc[:topN]]
        targets_low = [n for n,_ in degree_sorted_asc[:topN]]
        targets_rand = random.sample(list(G.nodes), topN)

        for label, targets in zip(
            ["high","low","rand"],
            [targets_high, targets_low, targets_rand]
        ):

            T, K, R = manual_initial_TKR(G, targets)

            T, K, R = tia_large_graph(G, T, K, R)

            R_connected = get_R_connected(G, R, K)
            R_isolated = R - R_connected

            critical_R = get_critical_R(G, R, R_connected, threshold)

            critical_R_coloring = coloring_reduction(G, critical_R)

            cascade_R_only(
                G,
                R_isolated,
                R_connected,
                critical_R_coloring,
                threshold
            )

            survival = 1

            if label == "high":

                surv_runs_high.append(survival)
                K_runs_high.append(len(K)/N)
                crit_runs_high.append(len(critical_R_coloring)/N)

            elif label == "low":

                surv_runs_low.append(survival)
                K_runs_low.append(len(K)/N)
                crit_runs_low.append(len(critical_R_coloring)/N)

            else:

                surv_runs_rand.append(survival)
                K_runs_rand.append(len(K)/N)
                crit_runs_rand.append(len(critical_R_coloring)/N)


    surv_high.append(np.mean(surv_runs_high))
    surv_low.append(np.mean(surv_runs_low))
    surv_rand.append(np.mean(surv_runs_rand))

    K_high.append(np.mean(K_runs_high))
    K_low.append(np.mean(K_runs_low))
    K_rand.append(np.mean(K_runs_rand))

    Rcrit_high.append(np.mean(crit_runs_high))
    Rcrit_low.append(np.mean(crit_runs_low))
    Rcrit_rand.append(np.mean(crit_runs_rand))

    print(f"\nThreshold = {threshold:.2f}")

    print(f"K after TIA (High targets)   = {np.mean(K_runs_high):.5f}")
    print(f"K after TIA (Low targets)    = {np.mean(K_runs_low):.5f}")
    print(f"K after TIA (Random targets) = {np.mean(K_runs_rand):.5f}") 

    print(f"Average Critical R (High targets)   = {np.mean(crit_runs_high):.5f}")
    print(f"Average Critical R (Low targets)    = {np.mean(crit_runs_low):.5f}")
    print(f"Average Critical R (Random targets) = {np.mean(crit_runs_rand):.5f}")


# =====================================================
# Combined Plot
# =====================================================

fig, ax1 = plt.subplots(figsize=(8,5))

# LEFT AXIS (Survival)

l1, = ax1.plot(threshold_values, surv_high,
               linestyle='--', color='black',
               label='Surviving $T$ (High)')

l2, = ax1.plot(threshold_values, surv_low,
               linestyle='-.', color='black',
               label='Surviving $T$ (Low)')

l3, = ax1.plot(threshold_values, surv_rand,
               linestyle=':', color='black',
               label='Surviving $T$ (Random)')

ax1.set_xlabel(r'Fractional Threshold ($\phi$)')
ax1.set_ylabel(r'Surviving $T$ Fraction')
ax1.set_ylim(-0.05,1.05)


# RIGHT AXIS

ax2 = ax1.twinx()

l4, = ax2.plot(threshold_values, K_high,
               marker='s', linestyle='--',
               label='High: Separator $K$')

l5, = ax2.plot(threshold_values, Rcrit_high,
               marker='^', linestyle='-.',
               label='High: Critical $R$')


l6, = ax2.plot(threshold_values, K_low,
               marker='o', linestyle='--',
               label='Low: Separator $K$')

l7, = ax2.plot(threshold_values, Rcrit_low,
               marker='v', linestyle='-.',
               label='Low: Critical $R$')


l8, = ax2.plot(threshold_values, K_rand,
               marker='D', linestyle='--',
               label='Random: Separator $K$')

l9, = ax2.plot(threshold_values, Rcrit_rand,
               marker='P', linestyle='-.',
               label='Random: Critical $R$')


ax2.set_ylabel('Fraction of Nodes')
ax2.set_ylim(-0.05,1.05)



ax1.text(
    0.02, 0.90, '(a)', #change according to degree
    transform=ax1.transAxes,
    fontsize=28,
    va='top'
)


# =====================================================
# Legend (custom legend box)
# =====================================================

lines = [l1,l2,l3,l4,l5,l6,l7,l8,l9]
labels = [l.get_label() for l in lines]

# ================= AXIS SETTINGS =================

# X-axis ticks
xticks = [0.80, 0.85, 0.90, 0.95]
ax1.set_xticks(xticks)
ax1.set_xticklabels([f"{x:.2f}" for x in xticks])

# Y-axis ticks (left)
yticks = [0.0, 0.5, 1.0]
ax1.set_yticks(yticks)
ax1.set_yticklabels([f"{y:.1f}" for y in yticks])

# Y-axis ticks (right)
ax2.set_yticks(yticks)
ax2.set_yticklabels([f"{y:.1f}" for y in yticks])

# =====================================================
# LEGENDS (Split into 3 groups)
# ====================================================

# -------- Surviving T --------
legend_T = ax1.legend(
    [l1, l2, l3],
    [l1.get_label(), l2.get_label(), l3.get_label()],
    loc='upper right',
    bbox_to_anchor=(0.98, 0.67),
    fontsize=13,
    borderpad=0.3,
    labelspacing=0.3,
    handletextpad=0.4
)
ax1.add_artist(legend_T)

# -------- Separator K --------
legend_K = ax2.legend(
    [l4, l6, l8],
    [l4.get_label(), l6.get_label(), l8.get_label()],
    loc='upper right',
    bbox_to_anchor=(0.98, 0.95),
    fontsize=13,
    borderpad=0.3,
    labelspacing=0.3,
    handletextpad=0.4
)
ax2.add_artist(legend_K)

# -------- Critical R --------
ax2.legend(
    [l5, l7, l9],
    [l5.get_label(), l7.get_label(), l9.get_label()],
    loc='upper right',
    bbox_to_anchor=(0.55, 0.95),
    fontsize=13,
    borderpad=0.3,
    labelspacing=0.3,
    handletextpad=0.4
)
