import networkx as nx
import random
from typing import Set, Tuple, List
import numpy as np
import matplotlib.pyplot as plt

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
# Initial T/K/R
# =====================================================
def manual_initial_TKR(G: nx.Graph, target_nodes: List[int]):

    T = set(target_nodes)

    K = set()
    for t in T:
        K.update(G.neighbors(t))

    K -= T

    R = set(G.nodes) - T - K

    return T, K, R


# =====================================================
# Lowest-degree critical R
# =====================================================
def get_low_degree_R(R, deg_dict, n_nodes):

    ranked = sorted(R, key=lambda x: deg_dict[x])

    return set(ranked[:min(n_nodes, len(ranked))])


# =====================================================
# Cascade
# =====================================================
def run_RKT_cascade(G_orig, T, K, R, shielded_K, threshold, n_initial_failures):

    G = G_orig.copy()
    init_deg = dict(G_orig.degree())

    failed_R, failed_K, failed_T = set(), set(), set()

    seeds = random.sample(list(R), min(n_initial_failures, len(R)))

    failed_R |= set(seeds)

    for s in seeds:
        G.remove_node(s)

    while True:

        newR, newK, newT = set(), set(), set()

        # R update
        for r in R - failed_R:

            if any(n in failed_R or n in failed_K for n in G_orig[r]):

                if init_deg[r] > 0 and G.degree(r)/init_deg[r] < threshold:

                    newR.add(r)

        # K update
        for k in (K - shielded_K) - failed_K:

            if any(n in failed_R or n in failed_K for n in G_orig[k]):

                if init_deg[k] > 0 and G.degree(k)/init_deg[k] < threshold:

                    newK.add(k)

        # T update
        for t in T - failed_T:

            if any(n in failed_K for n in G_orig[t]):

                if init_deg[t] > 0 and G.degree(t)/init_deg[t] < threshold:

                    newT.add(t)

        if not newR and not newK and not newT:
            break

        for x in newR | newK | newT:
            if x in G:
                G.remove_node(x)

        failed_R |= newR
        failed_K |= newK
        failed_T |= newT

    return failed_T


# =====================================================
# Experiment Function
# =====================================================
def run_experiment(network_type="BA"):

    N = 10000
    target_frac = 0.10
    n_runs = 10
    threshold_values = np.arange(0.80, 0.98, 0.01)
    n_initial_failures = 2

    surv_high = np.zeros(len(threshold_values))
    surv_low = np.zeros(len(threshold_values))
    surv_rand = np.zeros(len(threshold_values))

    for seed in range(n_runs):

        random.seed(seed)

        # -------- Build Graph --------
        if network_type == "BA":
            G = nx.barabasi_albert_graph(N, 3, seed=seed)
        else:
            p = 6 / N
            G = nx.erdos_renyi_graph(N, p, seed=seed)

        deg_dict = dict(G.degree)

        topN = int(N * target_frac)

        degs_desc = sorted(G.degree, key=lambda x: x[1], reverse=True)
        degs_asc = sorted(G.degree, key=lambda x: x[1])

        targets_high = [n for n,_ in degs_desc[:topN]]
        targets_low = [n for n,_ in degs_asc[:topN]]
        targets_rand = random.sample(list(G.nodes), topN)

        # ==================================================
        # Shielding numbers
        # ==================================================
        if network_type == "BA":
            shield_high = 23
            shield_low = 1343
            shield_rand = 938
        else:
            shield_high = 935
            shield_low = 3338
            shield_rand = 2166
        setups = []

        # High target nodes
        T, K, R = manual_initial_TKR(G, targets_high)
        critical_R = get_low_degree_R(R, deg_dict, shield_high)
        setups.append((T, K, R, critical_R))

        # Low target nodes
        T, K, R = manual_initial_TKR(G, targets_low)
        critical_R = get_low_degree_R(R, deg_dict, shield_low)
        setups.append((T, K, R, critical_R))

        # Random target nodes
        T, K, R = manual_initial_TKR(G, targets_rand)
        critical_R = get_low_degree_R(R, deg_dict, shield_rand)
        setups.append((T, K, R, critical_R))

        # -------- Loop thresholds --------
        for i, threshold in enumerate(threshold_values):

            for idx, (T, K, R, shielded_K) in enumerate(setups):

                failed_T = run_RKT_cascade(
                    G, T, K, R, shielded_K, threshold, n_initial_failures
                )

                surv = 1 - len(failed_T)/len(T)

                if idx == 0:
                    surv_high[i] += surv

                elif idx == 1:
                    surv_low[i] += surv

                else:
                    surv_rand[i] += surv

        print(f"{network_type} | Completed run {seed+1}/{n_runs}")

    surv_high /= n_runs
    surv_low /= n_runs
    surv_rand /= n_runs

    return threshold_values, surv_high, surv_low, surv_rand


# =====================================================
# RUN
# =====================================================
theta_BA, BA_high, BA_low, BA_rand = run_experiment("BA")

theta_ER, ER_high, ER_low, ER_rand = run_experiment("ER")


# =====================================================
# PLOT
# =====================================================
fig, ax = plt.subplots(figsize=(8,5))

# BA (filled markers, solid lines)
ax.plot(theta_BA, BA_high, marker='o',
        linestyle='-', label="BA High")

ax.plot(theta_BA, BA_low, marker='s',
        linestyle='-', label="BA Low")

ax.plot(theta_BA, BA_rand, marker='^',
        linestyle='-', label="BA Random")


# ER (open markers, solid lines)
ax.plot(theta_ER, ER_high, marker='o',
        linestyle='-',
        markerfacecolor='none', markeredgewidth=1.5,
        label="ER High")

ax.plot(theta_ER, ER_low, marker='s',
        linestyle='-',
        markerfacecolor='none', markeredgewidth=1.5,
        label="ER Low")

ax.plot(theta_ER, ER_rand, marker='^',
        linestyle='-',
        markerfacecolor='none', markeredgewidth=1.5,
        label="ER Random")

#ax.set_xlabel(r'Fractional Threshold ($\phi$)')
ax.set_ylabel(r'Surviving $T$ Fraction')

# ================= X-axis 
xticks = [0.80, 0.85, 0.90, 0.95]
ax.set_xticks(xticks)
ax.set_xticklabels([f"{x:.2f}" for x in xticks])

# ================= Y-axis (0.0, 0.5, 1.0)
yticks = [0.0, 0.5, 1.0]
ax.set_yticks(yticks)
ax.set_yticklabels([f"{y:.1f}" for y in yticks])

ax.text(
    0.02,0.85,'(b)',
    transform=ax.transAxes,
    fontsize=25,
    va='top'
)

# ===== Separate handles =====
handles, labels = ax.get_legend_handles_labels()

# BA = first 3
ba_handles = handles[:3]
ba_labels  = labels[:3]

# ER = next 3
er_handles = handles[3:]
er_labels  = labels[3:]



plt.tight_layout()
plt.show()
