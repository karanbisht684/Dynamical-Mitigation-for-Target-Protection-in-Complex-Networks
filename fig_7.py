import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt



# ============================================================
# NETWORK GENERATION
# ============================================================

def build_network(N, seed):
    np.random.seed(seed)
    random.seed(seed)

    G = nx.Graph()
    G.add_nodes_from(range(N))

    node_age = {}
    group_nodes = {g: [] for g in age_groups}

    counts = (age_dist * N).astype(int)
    idx = 0

    for i, g in enumerate(age_groups):
        for _ in range(counts[i]):
            if idx < N:
                node_age[idx] = g
                group_nodes[g].append(idx)
                idx += 1

    while idx < N:
        g = random.choice(age_groups)
        node_age[idx] = g
        group_nodes[g].append(idx)
        idx += 1

    # Build edges
    for i, gi in enumerate(age_groups):
        for u in group_nodes[gi]:
            for j, gj in enumerate(age_groups):
                k = np.random.poisson(M_rural[i, j])
                for _ in range(k):
                    if group_nodes[gj]:
                        v = random.choice(group_nodes[gj])
                        if u != v:
                            G.add_edge(u, v)

    return G, node_age

# ============================================================
# TARGET SET
# ============================================================

def get_target_nodes(node_age):
    return [n for n, age in node_age.items() if age in ["60+"]]

# ============================================================
# TIA
# ============================================================

def manual_initial_TKR(G, target_nodes):
    T = set(target_nodes)
    K = set()
    for t in T:
        K.update(G.neighbors(t))
    K -= T
    R = set(G.nodes) - T - K
    return T, K, R

def tia(G, T, K, R):
    while True:
        prev = (set(T), set(K), set(R))

        move_T = {n for n in K if not any(neigh in R for neigh in G.neighbors(n))}
        T |= move_T
        K -= move_T

        move_K = {n for n in R if any(neigh in T for neigh in G.neighbors(n))}
        K |= move_K
        R -= move_K

        if (T, K, R) == prev:
            break

    return T, K, R

# ============================================================
# CRITICAL R (NEW FRAGILITY)
# ============================================================

def get_critical_R(G, R, R_con, phi,threshold_no):
    deg = dict(G.degree())
#     avg_degree = np.mean(list(deg.values()))
#     print("Average degree:", avg_degree)
#     print(deg)

    fragile = {r for r in R if deg[r] > 0 and (1 / deg[r]) > phi}

    critical = set()
    for r in R_con:
        if r not in fragile:
            continue

        count = sum(1 for nbr in G.neighbors(r) if nbr in fragile and nbr in R)
        if count >= threshold_no:
            critical.add(r)

    return critical

# ============================================================
# GRAPH COLORING FILTER
# ============================================================

def coloring_filter(G, critical_R):
    colors = nx.coloring.greedy_color(G)
    avg_deg = {}

    for c in set(colors.values()):
        nodes = [n for n in colors if colors[n] == c]
        avg_deg[c] = np.mean([G.degree(n) for n in nodes])

    min_avg = min(avg_deg.values())

    return {r for r in critical_R if G.degree(r) > min_avg}

# ============================================================
# CASCADE (INFECTION MODEL)
# ============================================================

def run_cascade(G, R_iso, R_con, shielded, phi):

#     infected = set(random.sample(list(R_iso), min(5, len(R_iso))))
    seed_pool = list(R) if len(R_iso) == 0 else list(R_iso)
    n_seed = max(1, int(0.01 * len(seed_pool)))
    infected = set(random.sample(seed_pool, min(n_seed, len(seed_pool))))
    R_con_nonshielded = R_con - shielded

    while True:
        new_inf = set()

        for u in (R_iso - infected):
            if u in shielded:
                continue
            neigh = set(G.neighbors(u))
            if len(neigh) > 0 and (len(neigh & infected) / len(neigh)) > phi:
                new_inf.add(u)

        for u in (R_con_nonshielded - infected):
            neigh = set(G.neighbors(u))
            if len(neigh) > 0 and (len(neigh & infected) / len(neigh)) > phi:
                new_inf.add(u)

        if not new_inf:
            break

        infected |= new_inf

    return infected


# ============================================================
# CONTACT MATRICES (Fig. 3)
# ============================================================

age_groups = ["0-9","10-19","20-29","30-39","40-49","50-59","60+"]
threshold_no = 1
M_rural = np.array([
    [3.0, 2.7, 1.5, 1.7, 1.4, 2.1, 1.5],
    [2.0, 7.5, 2.9, 2.3, 2.1, 3.2, 1.8],
    [0.7, 2.0, 1.8, 1.4, 1.2, 1.7, 1.0],
    [0.6, 1.1, 1.0, 2.1, 1.7, 2.1, 1.4],
    [0.3, 0.6, 0.5, 1.1, 1.6, 2.0, 1.1],
    [0.3, 0.6, 0.5, 0.9, 1.3, 1.4, 1.0],
    [0.2, 0.4, 0.3, 0.6, 0.8, 1.1, 1.0]
])

M_urban = np.array([
    [1.8, 1.5, 1.4, 1.6, 0.9, 1.1, 1.1],
    [1.1, 5.2, 1.9, 2.1, 1.8, 1.7, 1.1],
    [0.7, 1.3, 1.0, 1.2, 0.9, 1.5, 0.8],
    [0.5, 1.0, 0.8, 1.8, 1.1, 1.1, 0.8],
    [0.2, 0.5, 0.4, 0.7, 0.9, 1.2, 0.6],
    [0.2, 0.3, 0.4, 0.5, 0.8, 0.7, 0.4],
    [0.2, 0.2, 0.2, 0.4, 0.4, 0.4, 1.0]
])
scale = 0.45   # tune this
threshold_no = 1
M_rural = scale * M_rural
# Age distribution (Table 1)
age_counts = np.array([531, 249, 125, 125, 104, 105, 124])
age_dist = age_counts / age_counts.sum()
# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    N = 1363
    thresholds = np.arange(0.01,0.2 , 0.01)
    n_runs = 10

    surv_protected = []
    surv_unprotected = []
    frac_CR = []
    frac_K = []

    for phi in thresholds:

        s_prot_runs = []
        s_unprot_runs = []
        CR_runs = []
        K_runs = []

        for seed in range(n_runs):

            G, node_age = build_network(N, seed)
            target_nodes = get_target_nodes(node_age)

            T, K, R = manual_initial_TKR(G, target_nodes)
            T, K, R = tia(G, T, K, R)

            R_con = {r for r in R if any(n in K for n in G.neighbors(r))}
            R_iso = R - R_con

            # -------- Protected case --------
            critical_R = get_critical_R(G, R, R_con, phi,threshold_no)
            critical_R = coloring_filter(G, critical_R)

            infected_R = run_cascade(G, R_iso, R_con, critical_R, phi)
#             print(len(infected_R))

#             infected_K = {k for k in K if len(set(G.neighbors(k)) & infected_R)/G.degree(k) > phi}
#             infected_T = {t for t in T if len(set(G.neighbors(t)) & infected_K)/G.degree(t) > phi}
            infected_K = {
                k for k in K
                if G.degree(k) > 0 and len(set(G.neighbors(k)) & infected_R) / G.degree(k) > phi
            }
            infected_T = {
            t for t in T
            if G.degree(t) > 0 and len(set(G.neighbors(t)) & infected_K) / G.degree(t) > phi
            }

            s_prot_runs.append(1 - len(infected_T)/max(len(T),1))

            # -------- No protection --------
            infected_R_np = run_cascade(G, R_iso, R_con, set(), phi)

            infected_K_np = {k for k in K if G.degree(k) > 0 and len(set(G.neighbors(k)) & infected_R_np)/G.degree(k) > phi}
            infected_T_np = {t for t in T if G.degree(t) > 0 and len(set(G.neighbors(t)) & infected_K_np)/G.degree(t) > phi}

            s_unprot_runs.append(1 - len(infected_T_np)/max(len(T),1))

            CR_runs.append(len(critical_R)/N)
            K_runs.append(len(K)/N)

        surv_protected.append(np.mean(s_prot_runs))
        surv_unprotected.append(np.mean(s_unprot_runs))
        frac_CR.append(np.mean(CR_runs))
        frac_K.append(np.mean(K_runs))

        print(f"phi={phi:.2f} | T_prot={np.mean(s_prot_runs):.3f} | "
              f"T_noProt={np.mean(s_unprot_runs):.3f} | "
              f"CR={np.mean(CR_runs):.3f} | K={np.mean(K_runs):.3f}")

# ============================================================
# PLOTS
# ============================================================

    plt.figure()
    plt.plot(thresholds, surv_protected, 'o-', label="Protected")
    plt.plot(thresholds, surv_unprotected, 's--', label="No protection")
    plt.xlabel("1 - φ")
    plt.ylabel("Survival of T")
    plt.title("Target Survivability")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(thresholds, frac_CR, 'o-', label="Critical-R")
    plt.plot(thresholds, frac_K, 's--', label="K (TIA)")
    plt.xlabel("1 - φ")
    plt.ylabel("Fraction")
    plt.title("Cost Comparison")
    plt.legend()
    plt.grid()
    plt.show()
