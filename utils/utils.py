from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import random
import warnings
# from matplotlib.pylab import chisquare
import optuna
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import spearmanr
import torch
import numpy as np
from pytorch_wavelets import DTCWTForward, DTCWTInverse
from scipy.stats import chi2_contingency, power_divergence
from sklearn.metrics import auc, confusion_matrix, mutual_info_score, precision_score, recall_score, f1_score, accuracy_score, roc_curve
import os
import seaborn as sns
import networkx as nx
from scipy.stats import chi2, kstest, gaussian_kde, combine_pvalues, norm, chisquare, permutation_test
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from PIL import Image, ImageDraw
import json
import sys


def view_subgraph(subgraph, title="Subgraph Visualization", save_path='subgraph.png'):
    """
    Visualize the subgraph using networkx and matplotlib.
    
    Parameters:
    -----------
    subgraph : networkx.Graph
        The subgraph to be visualized.
    title : str
        Title of the plot.
    save_path : str or None
        Path to save the plot as an image. If None, display interactively.
    """
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(subgraph)  # Layout for positioning the nodes
    
    # Assign random colors to edges
    edges = subgraph.edges()
    edge_colors = ["#" + "".join([random.choice("0123456789ABCDEF") for _ in range(6)]) for _ in edges]
        
    nx.draw(subgraph, pos, with_labels=False, node_size=400, node_color="skyblue", font_size=10, font_weight="bold")
    nx.draw_networkx_edges(subgraph, pos, edgelist=edges, edge_color=edge_colors, width=2)

    plt.title(title)
    plt.savefig(save_path, format='PNG')
    print(f"Subgraph saved to {save_path}")


def view_independence_subgraph(
        subgraph: nx.Graph,
        independent_nodes=None,
        *,
        k: float = 0.05,
        node_size: int = 2000,
        save_path: str = None,
        transparent: bool = True,
        edge_width: float = 2.25,
        highlight_width: float = 4.0,
):
    """
    Draw a NetworkX subgraph using a clean, publication‐ready style.

    Features
    --------
    • All nodes are **circular** with a white fill.
    • Nodes in `independent_nodes` keep an **orange outline**; others get blue.
    • Edges within the independent set become **solid, thick orange**.
    • All other edges are light-grey dashed.
    • Background can be fully transparent (ideal for PNG overlays).
    • Layout compactness controlled by `k` (Fruchterman–Reingold).

    Parameters
    ----------
    subgraph : networkx.Graph
        The graph (or subgraph) to draw.
    independent_nodes : list | set | None, optional
        Nodes to highlight.  Empty / None means “no special nodes”.
    k : float, optional
        Spring‐layout compactness (smaller ⇒ nodes closer).
    node_size : int, optional
        Size of each node (circles) in points².
    save_path : str | None, optional
        If provided, saves the figure to this path (PNG, 300 dpi).
        If None, shows the figure interactively.
    transparent : bool, optional
        Makes the figure background transparent when saving.
    edge_width : float, optional
        Width of the regular (grey dashed) edges.
    highlight_width : float, optional
        Width of the solid orange edges inside the independent set.
    """
    independent_nodes = set(independent_nodes) if independent_nodes else set()

    # ----- Layout -----------------------------------------------------------
    pos = nx.spring_layout(subgraph, k=k, seed=42)

    # ----- Figure / axis ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    if transparent:
        fig.patch.set_alpha(0)
        ax.set_facecolor('none')

    # ----- Edges ------------------------------------------------------------
    ALL_EDGE_KW       = dict(edge_color="lightgray", style="dashed", width=edge_width)
    HIGHLIGHT_EDGE_KW = dict(edge_color="tab:orange",  style="solid",  width=highlight_width)

    nx.draw_networkx_edges(subgraph, pos, **ALL_EDGE_KW)

    if independent_nodes:
        intra_edges = [
            e for e in subgraph.edges()
            if e[0] in independent_nodes and e[1] in independent_nodes
        ]
        nx.draw_networkx_edges(subgraph, pos, edgelist=intra_edges,
                               **HIGHLIGHT_EDGE_KW)

    # ----- Nodes ------------------------------------------------------------
    node_edge_colors = [
        "tab:orange" if n in independent_nodes else "tab:blue"
        for n in subgraph.nodes()
    ]

    nx.draw_networkx_nodes(
        subgraph, pos,
        node_color="white",
        edgecolors=node_edge_colors,
        linewidths=2.5,
        node_size=node_size,
        node_shape='o'
    )

    # ----- Labels -----------------------------------------------------------
    short_labels = {n: f"S{i+1}" for i, n in enumerate(subgraph.nodes())}
    nx.draw_networkx_labels(subgraph, pos, labels=short_labels, font_size=16)

    # ----- Cosmetics / export ----------------------------------------------
    ax.set_axis_off()
    plt.tight_layout()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300, bbox_inches="tight",
                    transparent=transparent)
        plt.close()
        print(f"Subgraph saved to {save_path}")


def chi_square_independence_test(observed):
    """
    chi2_stat : float
        Chi-square statistic.
    df : int
        Degrees of freedom.
    p_value : float
        P-value of the test.
    expected : 2D array
        Expected counts under null hypothesis of independence.
    """
    # Compute row and column sums
    row_totals = observed.sum(axis=1)
    col_totals = observed.sum(axis=0)
    grand_total = observed.sum()

    # Compute expected counts
    expected = np.outer(row_totals, col_totals) / grand_total

    # Compute the chi-square statistic
    chi2_stat = ((observed - expected)**2 / (expected + np.finfo(np.float16).eps)).sum()

    # Degrees of freedom
    r, c = observed.shape
    df = (r - 1) * (c - 1)

    # Compute p-value
    p_value = 1 - chi2.cdf(chi2_stat, df)

    return chi2_stat, p_value, df, expected


def calculate_chi2_and_corr(i, j, dist_1, dist_2, bins):
    """Compute chi-square p-value and correlation for two distributions."""
    try:
        corr, p_value = spearmanr(dist_1, dist_2)
        # correlation = abs(np.corrcoef(dist_1, dist_2)[0, 1])
        correlation = abs(corr)
        contingency_table, _, _ = np.histogram2d(dist_1, dist_2, bins=(bins, bins), range=[[0, 1], [0, 1]])
        chi2_stat, chi2_p, df, expected = chi2_contingency(contingency_table)
        return i, j, chi2_p, correlation
    except ValueError:
        return i, j, -1, correlation


def calculate_chi2_cremer_v_perm_and_corr(i, j, dist_1, dist_2, bins, n_perm=1000, seed=0):
    """
    Empirical p-value for independence using Cramér's V statistic with permutations.
    - Statistic: Cramér's V = sqrt(chi2 / (n * (min(bins_x, bins_y) - 1)))
    - Permutation: shuffle dist_2 only (break pairing, keep marginals)

    Returns: (i, j, p_empirical, |Spearman|)
    """
    # |Spearman|
    corr, _ = spearmanr(dist_1, dist_2)
    correlation = float(abs(corr)) if np.isfinite(corr) else 0.0

    # Bin the data (assumes values in [0,1] as in your original code)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bx = np.digitize(dist_1, edges) - 1
    by = np.digitize(dist_2, edges) - 1
    mask = (bx >= 0) & (bx < bins) & (by >= 0) & (by < bins)
    bx = bx[mask].astype(np.int32)
    by = by[mask].astype(np.int32)
    n = bx.size
    if n == 0:
        return i, j, -1.0, correlation

    # Row/col marginals and expected counts (fixed under permutations)
    row = np.bincount(bx, minlength=bins).astype(np.float64)
    col = np.bincount(by, minlength=bins).astype(np.float64)
    E = (row[:, None] * col[None, :]) / float(n)
    base_idx = bx * bins

    def _cremer_v_from_indices(by_indices):
        """Compute Cramér's V from permuted indices."""
        idx = base_idx + by_indices
        O = np.bincount(idx, minlength=bins * bins).astype(np.float64).reshape(bins, bins)
        chi2 = ((O - E) ** 2 / E).sum()
        return np.sqrt(chi2 / (n * (min(bins, bins) - 1)))

    # Observed Cramér's V
    cremer_v_obs = _cremer_v_from_indices(by)

    # Permutation null distribution
    rng = np.random.default_rng(seed)
    perm = np.arange(n, dtype=np.int32)
    ge = 0
    for _ in range(n_perm):
        rng.shuffle(perm)
        cremer_v_t = _cremer_v_from_indices(by[perm])
        if cremer_v_t >= cremer_v_obs:
            ge += 1

    # Empirical right-tailed p-value (add-one smoothing)
    p_emp = (ge + 1) / (n_perm + 1)

    return i, j, float(p_emp), correlation


def calculate_chi2_cremer_v_and_corr(i, j, dist_1, dist_2, bins):
    """Compute chi-square p-value and correlation for two distributions."""
    try:
        corr, p_value = spearmanr(dist_1, dist_2)
        # correlation = abs(np.corrcoef(dist_1, dist_2)[0, 1])
        correlation = abs(corr)
        contingency_table, _, _ = np.histogram2d(dist_1, dist_2, bins=(bins, bins), range=[[0, 1], [0, 1]])
        chi2_stat, chi2_p, df, expected = chi2_contingency(contingency_table)
        n = contingency_table.sum()
        k = min(contingency_table.shape)
        cramers_v = np.sqrt(chi2_stat / (n * (k - 1)))
        return i, j, cramers_v, correlation
    except ValueError:
        return i, j, -1, correlation
    

def calculate_mi_and_corr(i, j, dist_1, dist_2, bins):
    """Compute mutual information and correlation for two distributions."""
    try:
        corr, _ = spearmanr(dist_1, dist_2)
        correlation = abs(corr)
        contingency_table, _, _ = np.histogram2d(dist_1, dist_2, bins=(bins, bins))
        mi = mutual_info_score(None, None, contingency=contingency_table)
        return i, j, mi, correlation
    except ValueError:
        return i, j, -1, -1
    
def plot_contingency_table(contingency_table, save_path=None):
    """
    Plots a contingency table as a heatmap with cell values displayed.

    Args:
        contingency_table (numpy.ndarray): The contingency table to plot.
        save_path (str, optional): Path to save the plot. If None, the plot is not saved.

    Returns:
        None
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(contingency_table, cmap='viridis', interpolation='nearest')
    
    # Add cell values
    for i in range(contingency_table.shape[0]):
        for j in range(contingency_table.shape[1]):
            plt.text(j, i, str(np.round(contingency_table[i, j], 2)), ha='center', va='center', color='white', fontsize=10)
    
    # Add labels, title, and colorbar
    plt.colorbar(label='Value')
    plt.title("Contingency Table")
    plt.xlabel("Column")
    plt.ylabel("Row")
    
    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    

def calculate_chi2(i, j, dist_1, dist_2, bins):
    """Compute chi-square p-value and correlation for two distributions."""
    try:
        contingency_table, _, _ = np.histogram2d(dist_1, dist_2, bins=(bins, bins))
        chi2_stat, chi2_p, df, expected = chi2_contingency(contingency_table)
        return i, j, chi2_p
    except ValueError:
        return i, j, -1
    

def compute_chi2_matrix(keys, distributions, max_workers=128, plot_independence_heatmap=False, output_dir='logs', bins=10):
    """Compute Chi-Square p-value matrix and correlation matrix."""
    num_dists = len(distributions)
    chi2_p_matrix = np.zeros((num_dists, num_dists))

    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, key1 in enumerate(keys):
            dist_1 = distributions[i]
            for j, key2 in enumerate(keys):
                if i <= j:  # Skip duplicates and diagonal
                    continue
                dist_2 = distributions[j]
                tasks.append(executor.submit(calculate_chi2, i, j, dist_1, dist_2, bins))

        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Processing Chi2 and Correlation tests..."):
            i, j, chi2_p = future.result()
            if chi2_p is not None:
                chi2_p_matrix[i, j] = chi2_p
                chi2_p_matrix[j, i] = chi2_p  # Symmetry

    if plot_independence_heatmap:
        create_heatmap(chi2_p_matrix, keys, 'Chi-Square Test (P-values)', output_dir, 'chi2_heatmap.png', annot=len(keys) < 64)

    return chi2_p_matrix, None


def compute_chi2_and_corr_matrix(keys, distributions, max_workers=128, plot_independence_heatmap=False, output_dir='logs', bins=10):
    """Compute Chi-Square p-value matrix and correlation matrix."""
    num_dists = len(distributions)
    chi2_p_matrix = np.zeros((num_dists, num_dists))
    corr_matrix = np.zeros((num_dists, num_dists))

    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, key1 in enumerate(keys):
            dist_1 = distributions[i]
            for j, key2 in enumerate(keys):
                if i <= j:  # Skip duplicates and diagonal
                    continue
                dist_2 = distributions[j]
                tasks.append(executor.submit(calculate_chi2_cremer_v_and_corr, i, j, dist_1, dist_2, bins))

        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Processing Chi2 and Correlation tests..."):
            i, j, chi2_p, corr = future.result()
            if chi2_p is not None:
                chi2_p_matrix[i, j] = chi2_p
                chi2_p_matrix[j, i] = chi2_p  # Symmetry

            if chi2_p is not None:
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr  # Symmetry

    if plot_independence_heatmap:
        create_heatmap(chi2_p_matrix, keys, 'Chi-Square Test (P-values)', output_dir, 'chi2_heatmap.png', annot=len(keys) < 64)
        create_heatmap(corr_matrix, keys, 'Correlation Matrix', output_dir, 'corr_heatmap.png', annot=len(keys) < 64)

    return chi2_p_matrix, corr_matrix


def compute_mi_and_corr_matrix(keys, distributions, max_workers=128, plot_independence_heatmap=False, output_dir='logs', bins=10):
    num_dists = len(distributions)
    mi_matrix = np.zeros((num_dists, num_dists))
    corr_matrix = np.zeros((num_dists, num_dists))

    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, key1 in enumerate(keys):
            dist_1 = distributions[i]
            for j, key2 in enumerate(keys):
                if i <= j:
                    continue
                dist_2 = distributions[j]
                tasks.append(executor.submit(calculate_mi_and_corr, i, j, dist_1, dist_2, bins))

        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Processing MI and Correlation tests..."):
            i, j, mi, corr = future.result()
            if mi is not None:
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi
            if mi is not None:
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr

    if plot_independence_heatmap:
        create_heatmap(mi_matrix, keys, 'Mutual Information', output_dir, 'mi_heatmap.png', annot=len(keys) < 64)
        create_heatmap(corr_matrix, keys, 'Correlation Matrix', output_dir, 'corr_heatmap.png', annot=len(keys) < 64)

    return mi_matrix, corr_matrix


def find_largest_independent_group(keys, chi2_p_matrix, p_threshold=0.05, test_type="chi2"):
    """Find the largest independent group using the Chi-Square p-value matrix."""
    G = nx.Graph()
    G.add_nodes_from(keys)
    
    # Add edges where p-values are above the threshold
    if test_type == "chi2":
        # Chi-square: dependency = p < threshold (normal logic)
        indices = np.triu(chi2_p_matrix < p_threshold, k=1)
    else:
        # Permutation test: dependency = p < threshold, 
        # but mask lower triangle with 1 so it's ignored correctly
        masked_p = chi2_p_matrix.copy()
        masked_p[np.tril_indices_from(masked_p)] = 1
        indices = masked_p < p_threshold
    
    rows, cols = np.where(indices)
    edges = np.column_stack((np.array(keys)[rows], np.array(keys)[cols]))
    G.add_edges_from(edges)

    # Subgraph of nodes with edges (dependencies)
    subgraph = G.subgraph([node for node, degree in G.degree() if degree > 0])
    
    # Find the largest independent set (nodes not connected to others)
    independent_set = nx.algorithms.approximation.clique.max_clique(subgraph)
    return list(independent_set) if independent_set else [keys[0]]


def find_largest_independent_group_with_plot(keys, chi2_p_matrix, p_threshold=0.05, output_dir='logs'):
    """Find and plot the largest independent group using the Chi-Square p-value matrix."""
    G = nx.Graph()
    G.add_nodes_from(keys)

    indices = np.triu(chi2_p_matrix, k=1) < p_threshold
    rows, cols = np.where(indices)
    edges = np.column_stack((np.array(keys)[rows], np.array(keys)[cols]))
    G.add_edges_from(edges)

    subgraph = G.subgraph([node for node, degree in G.degree() if degree > 0])
    independent_set = nx.algorithms.approximation.clique.max_clique(subgraph)

    view_independence_subgraph(subgraph, save_path=os.path.join(output_dir, 'independence_graph.png'))
    view_independence_subgraph(subgraph, independent_set, save_path=os.path.join(output_dir, 'independent_clique.png'))

    return list(independent_set) if independent_set else [keys[0]]


def find_largest_uncorrelated_group(keys, corr_matrix, p_threshold=0.05):
    """Find the largest independent group using the Chi-Square p-value matrix."""
    G = nx.Graph()
    G.add_nodes_from(keys)
    
    # Add edges where p-values are below the threshold
    indices = np.triu(corr_matrix, k=1) < p_threshold
    rows, cols = np.where(indices)
    edges = np.column_stack((np.array(keys)[rows], np.array(keys)[cols]))
    G.add_edges_from(edges)

    # Subgraph of nodes with edges (dependencies)
    subgraph = G.subgraph([node for node, degree in G.degree() if degree > 0])
    
    # Find the largest independent set (nodes not connected to others)
    independent_set = nx.algorithms.approximation.clique.max_clique(subgraph)
    return list(independent_set) if independent_set else [keys[0]]


# def find_largest_independent_group_iterative(keys, chi2_p_matrix, p_threshold=0.05):
#     """Find the largest independent group using the Chi-Square p-value matrix."""
#     G = nx.Graph()
#     G.add_nodes_from(keys)
    
#     # indices = np.triu(chi2_p_matrix, k=1) > p_threshold
#     indices = np.triu(chi2_p_matrix, k=1) < p_threshold

#     rows, cols = np.where(indices)
#     edges = np.column_stack((np.array(keys)[rows], np.array(keys)[cols]))
#     G.add_edges_from(edges)

#     # Subgraph of nodes with edges (dependencies)
#     subgraph = G.subgraph([node for node, degree in G.degree() if degree > 0])
    
#     # All maximal cliques for node
#     cliques = list(nx.find_cliques(subgraph))
#     return cliques


def find_largest_independent_group_iterative(keys, p_matrix, p_threshold=0.05, test_type="chi2"):
    """
    Find the largest independent groups using a p-value matrix.

    Args:
        keys (list): Variable names.
        p_matrix (np.ndarray): Matrix of p-values.
        p_threshold (float): Threshold for dependency.
        test_type (str): 
            - "chi2" → keep original chi-square logic.
            - "perm" or anything else → opposite direction (permutation test).

    Returns:
        list: A list of maximal cliques (independent groups).
    """
    G = nx.Graph()
    G.add_nodes_from(keys)

    # Build dependency mask based on test type
    if test_type == "chi2":
        # Chi-square: dependency = p < threshold (normal logic)
        indices = np.triu(p_matrix < p_threshold, k=1)
    else:
        # Permutation test: dependency = p < threshold, 
        # but mask lower triangle with 1 so it's ignored correctly
        masked_p = p_matrix.copy()
        masked_p[np.tril_indices_from(masked_p)] = 1
        indices = masked_p < p_threshold

    # Add edges based on dependency mask
    rows, cols = np.where(indices)
    edges = np.column_stack((np.array(keys)[rows], np.array(keys)[cols]))
    G.add_edges_from(edges)

    # Get subgraph of nodes that actually have dependencies
    subgraph = G.subgraph([node for node, degree in G.degree() if degree > 0])

    # Find all maximal cliques
    cliques = list(nx.find_cliques(subgraph))

    return cliques


def create_heatmap(data, keys, title, output_dir, filename, figsize=(50, 25), annot=True):
    sorted_indices = np.argsort(keys)
    sorted_keys = [keys[i] for i in sorted_indices]
    sorted_data = data[sorted_indices][:, sorted_indices]


    mask = np.zeros_like(sorted_data, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    # Apply mask to zero-out the unwanted part
    sorted_data = np.ma.masked_where(mask, sorted_data)
    
    plt.figure(figsize=figsize)
    sns.heatmap(sorted_data, annot=annot, fmt=".2f", cmap=sns.color_palette("YlOrBr", as_cmap=True), 
                xticklabels=sorted_keys, yticklabels=sorted_keys, mask=mask, 
                cbar_kws={'label': 'Percentage (%)'}, annot_kws={"size": 10})
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.title(title)
    plt.tight_layout()
    
    # Save the heatmap
    plt.savefig(f"{output_dir}/{filename}")
    plt.close()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_population_histogram(real_population_histogram, portion):
    """
    Split the population histogram into tuning and training portions.
    - If portion <= 1.0: Use it as a fraction of the total population.
    - If portion > 1.0: Split into two histograms of sizes `portion` and `N - portion`.
    - If the population size is smaller than the portion, return a single split and None with a warning.
    """

    # Get the population length
    population_length = len(list(real_population_histogram.values())[0])

    # Ensure portion is valid
    if portion <= 0:
        raise ValueError(f"Invalid portion: {portion}. Must be greater than 0.")

    # Handle case where portion is larger than population size
    if (portion <= 1.0 and population_length * portion > population_length) or (portion > 1.0 and portion > population_length):
        warnings.warn(f"Portion {portion} is larger than the population size {population_length}. Returning a single split.")
        return real_population_histogram, None

    # Generate shuffled indices
    indices = list(range(population_length))
    random.shuffle(indices)

    if portion <= 1.0:
        # Portion as a fraction
        split_point = int(population_length * portion)
    else:
        # Portion as an absolute size
        split_point = int(portion)

    # Create the tuning and training histograms based on shuffled indices
    tuning_histogram = {
        k: [v[i] for i in indices[:split_point]] for k, v in real_population_histogram.items()
    }
    training_histogram = {
        k: [v[i] for i in indices[split_point:]] for k, v in real_population_histogram.items()
    }

    return tuning_histogram, training_histogram


def preprocess(image_batch, wavelet_decompose: DTCWTForward, wavelet_compose: DTCWTInverse):
    """Apply wavelet transform and reconstruction to a batch of images"""
    Yl, Yh = wavelet_decompose(image_batch)  # Perform wavelet decomposition
    Yh_zero = [torch.zeros_like(h) for h in Yh]  # Initialize all Yh components to zero
    Yl = torch.zeros_like(Yl)  # Set Yl to zeros
    Yh_zero[0] = Yh[0]  # Retain only the first Yh component
    reconstructed_batch = wavelet_compose((Yl, Yh_zero))  # Reconstruct the image batch
    return reconstructed_batch


def plot_pvalues_vs_bh_threshold(p_values_per_test, alpha=0.05, figname='k_m_plot.png'):
    """
    Plots the p-values vs. their order along with the k/m line (Benjamini-Hochberg threshold).
    """
    # Sort the p-values
    sorted_pvals = np.sort(p_values_per_test)
    m = len(p_values_per_test)

    # Calculate k/m line (BH threshold)
    k = np.arange(1, m+1)
    bh_line = (k * alpha) / m

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(k, sorted_pvals, label="p-values", marker="o", linestyle="--")
    plt.plot(k, bh_line, label="k/m line (BH threshold)", color="red", linestyle="-")

    # Labels and title
    plt.xlabel("Order of p-values")
    plt.ylabel("P-values")
    plt.title("P-values vs. k/m (BH Threshold)")
    plt.legend()

    # Display the plot
    plt.savefig(figname)


def plot_cdf(cdf_data, title="Empirical CDF Plot", xlabel="Value", ylabel="CDF", output_file='plot.png'):
    """
    Plot the CDF in a discrete, binned style using the original bins and cumulative values.
    """
    _, bin_edges, cdf = cdf_data

    # Plot the discrete CDF
    plt.figure(figsize=(8, 6))
    plt.bar(bin_edges[:-1], cdf, width=np.diff(bin_edges), align='edge', alpha=0.7, label="Empirical CDF")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.legend()
    plt.savefig(output_file)


def remove_nans_from_tests(tests_dict):
    """
    Filters out tests from a dictionary where any test contains NaN values.

    Args:
        tests_dict (dict): A dictionary where keys are test names and values are NumPy arrays.

    Returns:
        dict: A new dictionary with tests that do not contain NaN values.
    """
    cleaned_tests = {}

    for test_name, values in tests_dict.items():
        if np.isnan(values).any():
            warnings.warn(f"Test '{test_name}' contains NaN values and will be excluded.")
        else:
            cleaned_tests[test_name] = values

    return cleaned_tests


def compute_dist_cdf(distribution="normal", size=10000, bins=1000):
    """
    Compute and return the histogram, bin edges, and CDF for a given distribution.
    """
    # Generate data based on the specified distribution
    if distribution == "normal":
        data = np.random.normal(loc=0, scale=1, size=size)  # Standard normal
    elif distribution == "uniform":
        data = np.random.uniform(low=0, high=1, size=size)  # Standard uniform

    # Compute the histogram and CDF
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    cdf = np.cumsum(hist) * np.diff(bin_edges)
    
    return hist, bin_edges, cdf


def compute_cdf(histogram_values, bins=1000, test_id=None):
    """Compute CDFs from histogram values for each statistic descriptor."""
    hist, bin_edges = np.histogram(histogram_values, bins=bins, density=True)
    cdf = np.cumsum(hist) * np.diff(bin_edges)
    cdf_dict = (hist, bin_edges, cdf)  # Store both the histogram and the CDF
    return cdf_dict


def save_population_histograms(histograms, file_path):
    """Save the population histograms to a file."""
    with open(file_path, 'wb') as f:
        pickle.dump(histograms, f)


def load_population_histograms(file_path):
    """Load the population histograms from a file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    

def calculate_metrics(test_labels, predictions):
    """
    Calculate evaluation metrics including precision, recall, specificity, F1-score, and accuracy.
    """
    # Confusion matrix components
    cm = confusion_matrix(test_labels, predictions)
    tn, fp, fn, tp = cm.ravel()

    # Metrics
    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)  # Sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(test_labels, predictions)
    accuracy = accuracy_score(test_labels, predictions)

    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1,
        "accuracy": accuracy,
        "confusion_matrix": cm.tolist()  # Include CM for reference
    }


def plot_sensitivity_specificity_by_patch_size(results, statistic, threshold, output_dir):
    """Plot sensitivity (recall) and specificity across statistics and patches."""
    patches = sorted(results.keys())
    recalls = [results[patch]['recall'] for patch in patches]
    specificities = [results[patch]['specificity'] for patch in patches]

    # Plot recall and specificity
    plt.figure()
    plt.plot(patches, recalls, marker='o', label='Recall (Sensitivity)')
    plt.plot(patches, specificities, marker='x', label='Specificity')
    plt.title(f'Sensitivity and Specificity for Statistic: {statistic} (Alpha={threshold})')
    plt.xlabel('Patch Size')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'sensitivity_specificity_{statistic}_alpha_{threshold}.png'))
    plt.close()


def plot_pvalue_histograms(
    real_pvalues,
    fake_pvalues,
    figname,
    title,
    xlabel="p-values",
    ylabel="Density",
    bins=100,
    figsize=(8, 6),
    title_fontsize=14,
    label_fontsize=12,
    legend_fontsize=10
):
    """Plot histograms for real and fake p-values with transparency and save to a file."""
    plt.figure(figsize=figsize)
    plt.hist(real_pvalues, bins=bins, alpha=0.6, label="Real", color='tab:blue', density=True)
    plt.hist(fake_pvalues, bins=bins, alpha=0.6, label="Fake", color='tab:orange', density=True)
    plt.xlabel(xlabel, fontsize=label_fontsize)
    plt.ylabel(ylabel, fontsize=label_fontsize)
    plt.title(title, fontsize=title_fontsize)
    plt.legend(fontsize=legend_fontsize)
    plt.tight_layout()
    plt.savefig(figname)
    plt.close()



def plot_histograms(hist, figname='plot.png', title='histogram', density=True, bins=50):
    """Plot histograms for real and fake p-values with transparency and save to a file."""
    plt.figure(figsize=(12, 6))
    plt.hist(hist, bins=bins, alpha=0.5, color='blue', density=density, edgecolor='k')
    plt.xlabel("p-values")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figname)


def plot_binned_histogram(hist, bin_edges, figname='plot.png'):
    """Plot a histogram from precomputed bin counts and bin edges."""
    widths = np.diff(bin_edges)
    plt.bar(bin_edges[:-1], hist, width=widths, align='edge',
            alpha=0.7, edgecolor='k')
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.savefig(figname)


def plot_uniform_and_nonuniform(pvalue_distributions, uniform_indices, output_dir, bins=50):
    """
    Plots 5 uniform and 5 non-uniform p-value distributions in a 2x5 grid.
    Displays the p-value of the KS test for uniformity on each plot.
    """
    # Convert uniform_indices to a set for faster lookup
    uniform_set = set(uniform_indices)
    uniform_distributions = [pvalue_distributions[i] for i in uniform_indices[:5]]

    # Select up to 5 non-uniform distributions
    nonuniform_distributions = [
        pvalue_distributions[i] for i in range(pvalue_distributions.shape[0]) if i not in uniform_set
    ][:5]

    # Create a 2x5 grid for plots
    fig, axes = plt.subplots(2, 5, figsize=(40, 8))

    # Add column titles
    fig.suptitle("Uniform vs Non-Uniform P-Value Distributions", fontsize=16)

    # Plot uniform distributions in the first row
    for idx, ax in enumerate(axes[0, :]):
        if idx < len(uniform_distributions):
            dist = uniform_distributions[idx]
            ax.hist(dist, bins=bins, alpha=0.5, color='blue', density=False, edgecolor='k')
            pval = kstest(dist, 'uniform').pvalue
            ax.text(0.05, 0.95, f"p-value: {pval:.3f}", transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
        else:
            ax.axis('off')  # Leave the remaining cells blank

        ax.set_xlabel("p-value")
        ax.set_ylabel("Density")

    # Plot non-uniform distributions in the second row
    for idx, ax in enumerate(axes[1, :]):
        if idx < len(nonuniform_distributions):
            dist = nonuniform_distributions[idx]
            ax.hist(dist, bins=bins, alpha=0.5, color='red', density=False, edgecolor='k')
            pval = kstest(dist, 'uniform').pvalue
            ax.text(0.05, 0.95, f"p-value: {pval:.3f}", transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
        else:
            ax.axis('off')  # Leave the remaining cells blank

        ax.set_xlabel("p-value")
        ax.set_ylabel("Density")

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to leave space for the title

    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "uniform_vs_nonuniform_distributions_2x5.png")
    plt.savefig(output_path)
    plt.close()


def plot_roc_curve(results, test_id, output_dir):
    """Plot ROC curve for a specific test."""
    labels = results['labels']
    scores = results['scores']
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curves
    plt.figure()
    plt.plot(fpr, tpr, label=f"{results['n_tests']} Tests (AUC = {roc_auc:.2f})")

    plt.title(f'ROC Curve for Test: {test_id}')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Recall)')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(os.path.join(output_dir, f'roc_curve.png'))
    plt.close()
    return roc_auc


def plot_stouffer_analysis(
    pvalues, inverse_z_scores, stouffer_statistic, stouffer_pvalues, 
    plot_bins=50, output_folder='logs', num_plots_pvalues=10, num_plots_zscores=10
):
    """
    Generate and save plots for the p-values, inverse z-scores, Stouffer's statistics, and Stouffer's p-values.

    Parameters:
    - pvalues: Array of p-values (N x T).
    - inverse_z_scores: Array of inverse z-scores (N x T).
    - stouffer_statistic: Array of combined z-scores (N x 1).
    - stouffer_pvalues: Array of Stouffer's combined p-values (N x 1).
    - plot_bins: Number of bins for the plotted histograms.
    - output_folder: Folder to save the output figures.
    - num_plots_pvalues: Number of columns to display for p-values subplots.
    - num_plots_zscores: Number of columns to display for inverse z-scores subplots.
    """
    T = pvalues.shape[1]  # Number of tests

    # Step 1: Plot sampled p-values for the first `num_plots_pvalues` tests
    fig_pvalues, axes_pvalues = plt.subplots(1, min(num_plots_pvalues, T), figsize=(20, 4))
    if T == 1:  # If there's only one test, axes_pvalues is not iterable
        axes_pvalues = [axes_pvalues]

    for i in range(min(num_plots_pvalues, T)):
        axes_pvalues[i].hist(pvalues[:, i], bins=plot_bins, edgecolor='k', alpha=0.7, density=True, color='blue')
        axes_pvalues[i].set_title(f"P-Values Test {i+1}")
        axes_pvalues[i].set_xlabel("P-Value")
        axes_pvalues[i].set_ylabel("Density")

    fig_pvalues.tight_layout()
    fig_pvalues.savefig(f"{output_folder}/pvalues_histogram_subset.png")
    plt.close(fig_pvalues)

    # Step 2: Plot inverse z-scores for the first `num_plots_zscores` tests
    fig_zscores, axes_zscores = plt.subplots(1, min(num_plots_zscores, T), figsize=(20, 4))
    if T == 1:  # If there's only one test, axes_zscores is not iterable
        axes_zscores = [axes_zscores]

    for i in range(min(num_plots_zscores, T)):
        axes_zscores[i].hist(inverse_z_scores[:, i], bins=plot_bins, edgecolor='k', alpha=0.7, density=True, color='orange')
        axes_zscores[i].set_title(f"Z-Scores Test {i+1}")
        axes_zscores[i].set_xlabel("Z-Score Value")
        axes_zscores[i].set_ylabel("Density")

    fig_zscores.tight_layout()
    fig_zscores.savefig(f"{output_folder}/inverse_zscores_histogram_subset.png")
    plt.close(fig_zscores)

    # Step 3: Plot Stouffer's statistics
    fig_stouffer_stat, ax_stouffer_stat = plt.subplots(figsize=(10, 6))
    ax_stouffer_stat.hist(stouffer_statistic, bins=plot_bins, edgecolor='k', alpha=0.7, density=True, color='green')
    ax_stouffer_stat.set_title("Histogram of Stouffer's Statistics")
    ax_stouffer_stat.set_xlabel("Stouffer Statistic")
    ax_stouffer_stat.set_ylabel("Density")
    fig_stouffer_stat.tight_layout()
    fig_stouffer_stat.savefig(f"{output_folder}/stouffer_statistics_histogram.png")
    plt.close(fig_stouffer_stat)

    # Step 4: Plot Stouffer's p-values
    fig_stouffer_pvalues, ax_stouffer_pvalues = plt.subplots(figsize=(10, 6))
    ax_stouffer_pvalues.hist(stouffer_pvalues, bins=plot_bins, edgecolor='k', alpha=0.7, density=True, color='purple')
    ax_stouffer_pvalues.set_title("Histogram of Stouffer's P-Values")
    ax_stouffer_pvalues.set_xlabel("P-Value")
    ax_stouffer_pvalues.set_ylabel("Density")
    fig_stouffer_pvalues.tight_layout()
    fig_stouffer_pvalues.savefig(f"{output_folder}/stouffer_pvalues_histogram.png")
    plt.close(fig_stouffer_pvalues)


def plot_ks_vs_pthreshold(thresholds, ks_pvalues, output_dir):
    """
    Plots KS p-value vs p-threshold.
    """
    sorted_indices = np.argsort(thresholds)
    sorted_thresholds = np.array(thresholds)[sorted_indices]
    sorted_ks_pvalues = np.array(ks_pvalues)[sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.plot(sorted_thresholds, sorted_ks_pvalues, marker='o', linestyle='-', label="KS p-value")
    plt.xlabel("p_threshold")
    plt.ylabel("KS p-value")
    plt.title("KS p-value vs p-threshold")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "ks_pvalue_wrt_p_threshold.png"))


def AUC_tests_filter(tuning_pvalue_distributions, fake_calibration_pvalue_distributions, auc_threshold=0.6):
    """
    Calculates AUC scores for each test and selects indices with AUC > threshold.
    Adjusts for the relationship where smaller p-values indicate outliers.
    
    Parameters:
    - tuning_pvalue_distributions (np.ndarray): P-value distributions for real data (shape: tests x samples).
    - fake_calibration_pvalue_distributions (np.ndarray): P-value distributions for fake data (shape: tests x samples).
    - auc_threshold (float): Threshold for selecting the best keys based on AUC (default: 0.6).
    
    Returns:
    - auc_scores (np.ndarray): AUC scores for each test.
    - best_keys (np.ndarray): Indices of tests with AUC > auc_threshold.
    """
    # Reverse the p-values
    real_scores = 1 - tuning_pvalue_distributions
    fake_scores = 1 - fake_calibration_pvalue_distributions
    
    # Combine distributions and labels
    combined_pvalues = np.concatenate([real_scores, fake_scores], axis=1)
    combined_labels = np.concatenate([
        np.zeros_like(tuning_pvalue_distributions),
        np.ones_like(fake_calibration_pvalue_distributions)
    ], axis=1)
    
    # Remove entries with NaN values
    # TODO: why are there NaN values? - check the input data
    valid_indices = ~np.isnan(combined_pvalues).any(axis=1)
    combined_pvalues = combined_pvalues[valid_indices]
    combined_labels = combined_labels[valid_indices]

    # Calculate AUC scores using list comprehension
    auc_scores = np.array([roc_auc_score(combined_labels[i], combined_pvalues[i]) for i in range(combined_pvalues.shape[0])])
    
    # Select best keys with AUC > threshold
    best_keys = np.where(auc_scores > auc_threshold)[0]
    
    return auc_scores, best_keys


def save_to_csv(keys, auc_scores, filename="auc_scores.csv"):
    # Create a DataFrame from the arrays
    df = pd.DataFrame({"Keys": keys, "AUC Scores": auc_scores})
    
    # Save DataFrame to a CSV file
    df.to_csv(filename, index=False)
    
    print(f"File saved as {filename}")


def compute_mean_std_dict(input_dict):
    """
    Given a dictionary where each key maps to an array of shape (N, L),
    compute the mean and std along axis=1 and store them in a new dictionary.
    
    Args:
        input_dict (dict): Dictionary with arrays of shape (N, L) where L varies.
    
    Returns:
        dict: Dictionary with mean and std stored as new keys with '_mean' and '_std' suffixes.
    """
    output_dict = {}

    for key, array in input_dict.items():
        # Ensure the array is a NumPy array
        array = np.asarray(array)

        if array.shape[-1] == 1:
            output_dict[key] = array.squeeze()
            continue

        # Compute mean and std along axis 1 (for each row)
        mean_values = np.mean(array, axis=1)  # Shape: (N,)
        std_values = np.std(array, axis=1)    # Shape: (N,)

        # Store in dictionary with updated keys
        output_dict[f"{key}_mean"] = mean_values
        output_dict[f"{key}_std"] = std_values

    return output_dict


def plot_pvalue_histograms_from_arrays(
    real_pvals_array, 
    inference_pvals_array, 
    artifact_path,
    keys
):
    """
    Plots p-value histograms for each test using plot_pvalue_histograms."
    """

    _, T = real_pvals_array.shape
    assert inference_pvals_array.shape[-1] ==  T, "Input arrays must have the same shape"

    for t in range(T):
        real_pvals = real_pvals_array[:, t]
        inf_pvals = inference_pvals_array[:, t]
        corrected_name = keys[t].replace(" ", "_").replace(".", "_").replace("=", "-")
        output_file = f"{artifact_path}_test_{corrected_name}"

        plot_pvalue_histograms(
            real_pvals,
            inf_pvals,
            output_file,
            title=f"Histogram of P-values - Test {corrected_name}"
        )

def build_backbones_statistics_list(models, noise_levels, prefix="RIGID"):
    """
    Generates a list of statistic names in the format RIGID.{MODEL}.{NOISE}.

    Args:
        models (list of str): Model names, e.g. ['DINO', 'CLIP']
        noise_levels (list of str): Noise levels as strings, e.g. ['001', '01', '05']
        prefix (str): Optional prefix, default is 'RIGID'

    Returns:
        list of str: All statistic combinations like RIGID.DINO.01, RIGID.CLIP.05, ...
    """
    return [f"{prefix}.{model}.{noise}" for model in models for noise in noise_levels]


import os
import numpy as np
import matplotlib.pyplot as plt

def plot_fakeness_score_distribution(results, test_id, output_dir, threshold=0.5):
    """
    Plot fakeness scores per sample with block-wise shuffling,
    styled to match the current matplotlib environment configuration.

    Parameters:
    - results (dict): Must contain 'scores' (fakeness score: high=fake) and 'labels' (0=real, 1=fake).
    - test_id (str): Unique ID for the test (used for filename).
    - output_dir (str): Directory to save the plot.
    - threshold (float): Score threshold for decision boundary (default 0.5).

    Returns:
    - plot_path (str): File path of the saved plot.
    """
    # Extract and shuffle scores per class
    scores = np.array(results['scores'])
    labels = np.array(results['labels'])

    real_scores = np.random.permutation(scores[labels == 0])
    fake_scores = np.random.permutation(scores[labels == 1])

    real_x = np.arange(len(real_scores))
    fake_x = np.arange(len(real_scores), len(real_scores) + len(fake_scores))

    # Plot
    plt.figure(figsize=(12, 4))
    plt.scatter(real_x, real_scores, alpha=0.5, label='Real', s=10, color='tab:blue')
    plt.scatter(fake_x, fake_scores, alpha=0.5, label='Fake', s=10, color='tab:orange')
    plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')

    plt.xlabel("Sample Index")
    plt.ylabel("Fakeness Prob")
    plt.title("Ensemble Probability per Sample (Block-Shuffled)")
    plt.legend(loc='upper left')
    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{test_id}_fakeness_score_plot.svg")
    plt.savefig(plot_path)
    plt.close()

    print(f"[INFO] Fakeness score plot saved to: {plot_path}")
    return plot_path


def plot_fakeness_score_histogram(results, test_id, output_dir, threshold=0.5):
    """
    Plot a histogram of fakeness scores for real and fake samples
    with consistent matplotlib styling and standardized text sizes.
    """
    scores = np.array(results['scores'])
    labels = np.array(results['labels'])

    real_scores = scores[labels == 0]
    fake_scores = scores[labels == 1]

    # Plot
    plt.figure(figsize=(8, 4))
    plt.hist(real_scores, bins=50, alpha=0.6, label='Real', density=True, color='tab:blue')
    plt.hist(fake_scores, bins=50, alpha=0.6, label='Fake', density=True, color='tab:orange')
    plt.axvline(x=threshold, color='red', linestyle='--', label=f'significance level α = {1 - threshold:.2f}')

    # Font size settings
    plt.title("Ensemble Probability Distribution", fontsize=18)
    plt.xlabel("Fakeness Prob", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=14)

    plt.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{test_id}_classifier_score_histogram.svg")
    plt.savefig(plot_path)
    plt.close()

    print(f"[INFO] Classifier score histogram saved to: {plot_path}")
    return plot_path


def plot_kde_with_image_markers(pvals_real, image_pvals, image_labels, figsize=(6, 6), bw=0.2):
    """
    Plots a KDE of the real p-value distribution and overlays image p-value markers.
    
    Args:
        pvals_real (array-like): Reference real p-value distribution.
        image_pvals (array-like): List of p-values for individual images.
        image_labels (list of str): List of labels for each image ('real' or 'fake').
        figsize (tuple): Size of the output figure (default: (6, 6)).
        bw (float): Bandwidth for KDE smoothing (default: 0.2).
    """
    # Compute KDE
    kde = gaussian_kde(pvals_real, bw_method=bw)
    x_vals = np.linspace(0, 1, 1000)
    y_vals = kde(x_vals)

    # Create figure
    plt.figure(figsize=figsize)
    plt.plot(x_vals, y_vals, color='blue', linewidth=2)
    plt.fill_between(x_vals, y_vals, color='blue', alpha=0.4)

    # Plot vertical lines for each image
    for pval, label in zip(image_pvals, image_labels):
        color = 'green' if label == 'real' else 'red'
        plt.axvline(pval, ymin=0, ymax=0.3, color=color, linestyle='--', alpha=0.8, linewidth=2)

    # Clean styling
    plt.xlabel("")
    plt.ylabel("")
    plt.yticks([])
    plt.xticks(np.linspace(0, 1, 5))
    plt.title("")
    plt.box(False)
    plt.grid(False)
    plt.tight_layout()
    plt.show()


def save_per_image_kde_and_images(
    image_paths,
    test_labels,
    tuning_real_population_pvals,
    input_samples_pvalues,
    independent_statistics_keys_group,
    output_dir,
    max_per_class=3
):
    """
    Saves KDE plots with vertical lines marking each selected image's p-value,
    along with a copy of the image.

    Args:
        image_paths (list): List of image file paths.
        test_labels (list): Corresponding list of labels (0 for real, 1 for fake).
        tuning_real_population_pvals (np.ndarray): [num_samples x num_statistics] real p-values.
        input_samples_pvalues (list of list): Per-image p-values for the statistics.
        independent_statistics_keys_group (list): List of statistic keys in order.
        output_dir (str): Directory to save plots.
        max_per_class (int): Max number of images per class to plot.
    """
    save_dir = os.path.join(output_dir, "image_kde_markers")
    os.makedirs(save_dir, exist_ok=True)

    num_stats = len(independent_statistics_keys_group)
    num_real_samples, stats_dim = tuning_real_population_pvals.shape
    num_test_images = len(image_paths)

    # Ensure statistics match
    assert stats_dim == num_stats, f"Expected {num_stats} statistics, but tuning_real_population_pvals has {stats_dim}"
    assert len(test_labels) == len(image_paths) == len(input_samples_pvalues), "Mismatch in number of test images, labels, or p-values"

    for i, pvals in enumerate(input_samples_pvalues):
        assert len(pvals) == num_stats, f"Sample {i} p-value length mismatch: expected {num_stats}, got {len(pvals)}"

    label_indices = {0: [], 1: []}
    for idx, label in enumerate(test_labels):
        if len(label_indices[label]) < max_per_class:
            label_indices[label].append(idx)

    for label, indices in label_indices.items():
        label_name = 'real' if label == 0 else 'fake'
        division_color = 'black' if label == 0 else 'red'

        label_dir = os.path.join(save_dir, label_name)
        os.makedirs(label_dir, exist_ok=True)

        for i in indices:
            pvals = input_samples_pvalues[i]
            img_path = image_paths[i]
            img_basename = os.path.splitext(os.path.basename(img_path))[0]

            # Save original image
            img_save_path = os.path.join(label_dir, f"{img_basename}.png")
            Image.open(img_path).convert("RGB").save(img_save_path)

            for j, stat_key in enumerate(independent_statistics_keys_group):
                fig, ax = plt.subplots(figsize=(4, 4))

                # Plot histogram of reference (real) p-values
                ax.hist(
                    tuning_real_population_pvals[:, j],
                    bins=20,
                    density=True,
                    color="tab:blue",
                    alpha=0.6,
                )

                # Vertical marker for current image's p-value
                ax.axvline(pvals[j], color=division_color, linestyle="--", linewidth=2)

                # Minimal layout
                ax.set_yticks([])
                ax.set_xticks([])
                ax.grid(False)
                ax.set_xlim(0, 1)

                stat_filename = stat_key.replace("=", "-").replace(",", "_").replace(" ", "_")
                plot_path = os.path.join(label_dir, f"{img_basename}_{stat_filename}.png")
                plt.tight_layout()
                plt.savefig(plot_path, bbox_inches="tight", pad_inches=0.1)
                plt.close()


def create_pvalue_grid_figure(
    image_paths,
    pvalues,
    test_labels,
    threshold=0.05,
    max_per_group=4,
    figsize=(15, 5),
    output_path="pvalue_grid_figure.png"
):
    """
    Creates a 2-row figure showing images grouped by p-value threshold.
    Top row: fake (label==1) with p < threshold
    Bottom row: real (label==0) with p >= threshold
    """
    assert len(image_paths) == len(pvalues) == len(test_labels), "Length mismatch"

    def preprocess_image(path):
        img = Image.open(path).convert("RGB")
        w, h = img.size
        scale = 512 / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        left = (new_w - 512) // 2
        upper = (new_h - 512) // 2
        img = img.crop((left, upper, left + 512, upper + 512))
        return img

    top_group = [
        (img, pv) for img, pv, label in zip(image_paths, pvalues, test_labels)
        if pv < threshold and label == 1
    ]
    bottom_group = [
        (img, pv) for img, pv, label in zip(image_paths, pvalues, test_labels)
        if pv >= threshold and label == 0
    ]

    if len(top_group) < max_per_group or len(bottom_group) < max_per_group:
        return False  # Skip this figure

    top_group = top_group[:max_per_group]
    bottom_group = bottom_group[:max_per_group]
    n_cols = max(len(top_group), len(bottom_group))
    n_rows = 2

    fig, axes = plt.subplots(
        n_rows, n_cols + 1, figsize=figsize, 
        gridspec_kw={'width_ratios': [0.5] + [1.0] * n_cols, 'wspace': 0.05}  # ⬅️ Shrink label column
    )

    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")

    # Increase font sizes by ~25%
    label_fontsize = 36
    title_fontsize = 28

    # Left column group labels
    axes[0][0].text(
        0.5, 0.5, f"$p < {threshold}$",
        fontsize=label_fontsize, ha="center", va="center",
        transform=axes[0][0].transAxes
    )
    axes[1][0].text(
        0.5, 0.5, f"$p \\geq {threshold}$",
        fontsize=label_fontsize, ha="center", va="center",
        transform=axes[1][0].transAxes
    )

    # Fill in images + p-values
    for i, (img_path, pv) in enumerate(top_group):
        img = preprocess_image(img_path)
        axes[0][i + 1].imshow(img)
        axes[0][i + 1].set_title(f"$p = {pv:.3f}$", fontsize=title_fontsize)

    for i, (img_path, pv) in enumerate(bottom_group):
        img = preprocess_image(img_path)
        axes[1][i + 1].imshow(img)
        axes[1][i + 1].set_title(f"$p = {pv:.3f}$", fontsize=title_fontsize)

    # plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return True


def create_multiband_pvalue_grid_figure(
    image_paths,
    pvalues,
    test_labels,
    thresholds=[0.1, 0.25, 0.5],
    max_per_group=4,
    figsize=(12, 10),
    output_path="pvalue_multiband_grid.png"
):
    """
    Creates a multi-row figure showing images grouped by p-value intervals.
    Each row corresponds to a p-value bin between two thresholds.
    Only fakes (label=1) are used for lower intervals, and only reals (label=0) for higher ones.
    """
    assert len(image_paths) == len(pvalues) == len(test_labels), "Length mismatch"

    def preprocess_image(path):
        img = Image.open(path).convert("RGB")
        w, h = img.size
        scale = 512 / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        left = (new_w - 512) // 2
        upper = (new_h - 512) // 2
        img = img.crop((left, upper, left + 512, upper + 512))
        return img

    # Define threshold bands
    bands = [0.0] + thresholds + [1.0]
    num_rows = len(bands) - 1

    grouped_rows = []
    for i in range(num_rows):
        lower = bands[i]
        upper = bands[i + 1]
        # Label logic
        if i == 0:
            # Lowest p-value bin: only allow fake (label == 1)
            group = [
                (img, pv, label) for img, pv, label in zip(image_paths, pvalues, test_labels)
                if lower < pv < upper and label == 1
            ]
        elif i == num_rows - 1:
            # Highest p-value bin: only allow real (label == 0)
            group = [
                (img, pv, label) for img, pv, label in zip(image_paths, pvalues, test_labels)
                if lower <= pv < upper and label == 0
            ]
        else:
            # Middle bins: allow all labels
            group = [
                (img, pv, label) for img, pv, label in zip(image_paths, pvalues, test_labels)
                if lower < pv < upper
            ]

        if len(group) < max_per_group:
            return False  # Skip figure if any row is underpopulated

        grouped_rows.append(group[:max_per_group])

    n_cols = max(len(row) for row in grouped_rows)

    # Dynamically scale widths: 25% for labels, 75% divided among image columns
    label_col_ratio = 0.2
    img_col_ratio = (1.0 - label_col_ratio) / n_cols
    fig, axes = plt.subplots(
        num_rows, n_cols + 1,
        figsize=figsize,
        gridspec_kw={
            'width_ratios': [label_col_ratio] + [img_col_ratio] * n_cols,
            'wspace': 0.01,
            'hspace': 0.01
        }
    )

    label_fontsize = 16
    # title_fontsize = 20

    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")

    for row_idx, row_group in enumerate(grouped_rows):
        lower = bands[row_idx]
        upper = bands[row_idx + 1]

        # Label for left column
        if row_idx == 0:
            label_text = f"$0 < p < {upper}$"
        elif upper == 1.0:
            label_text = f"${lower} \\leq p \\leq 1$"
        else:
            label_text = f"${lower} \\leq p < {upper}$"

        axes[row_idx][0].text(
            0.5, 0.5, label_text,
            fontsize=label_fontsize, ha="center", va="center",
            transform=axes[row_idx][0].transAxes
        )

        # Fill in the images
        for col_idx, (img_path, pv, label) in enumerate(row_group):
            img = preprocess_image(img_path)
            ax = axes[row_idx][col_idx + 1]
            ax.imshow(img)
            # axes[row_idx][col_idx + 1].set_title(f"$p = {pv:.3f}$", fontsize=title_fontsize)

            # Add label overlay to each image (Fake or Real)
            lbl_text = "Fake" if label == 1 else "Real"
            lbl_color = "red" if label == 1 else "green"
            ax.text(
                5, 20, lbl_text,  # Position in pixels from top-left
                fontsize=12,
                color=lbl_color,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2')
            )

    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    # plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return True


def save_ensembled_pvalue_kde_and_images(
    image_paths,
    test_labels,
    ensembled_pvalues,
    tuning_ensembled_pvalues,
    output_dir,
    max_per_class=3
):
    """
    Saves KDE plots of ensembled p-values for selected real/fake images,
    with vertical lines showing where each image falls.

    Args:
        image_paths (list): List of image file paths.
        test_labels (list): Corresponding labels (0=real, 1=fake).
        ensembled_pvalues (list or np.ndarray): Inference-time ensembled p-values.
        tuning_ensembled_pvalues (np.ndarray): Reference distribution from training set.
        output_dir (str): Directory to save output.
        max_per_class (int): Number of images per class to save.
    """
    save_dir = os.path.join(output_dir, "ensemble_kde_markers")
    os.makedirs(save_dir, exist_ok=True)

    label_indices = {0: [], 1: []}
    for idx, label in enumerate(test_labels):
        if len(label_indices[label]) < max_per_class:
            label_indices[label].append(idx)

    for label, indices in label_indices.items():
        label_name = 'real' if label == 0 else 'fake'
        division_color = 'black' if label == 0 else 'red'
        label_dir = os.path.join(save_dir, label_name)
        os.makedirs(label_dir, exist_ok=True)

        for i in indices:
            img_path = image_paths[i]
            img_basename = os.path.splitext(os.path.basename(img_path))[0]

            # Save original image
            img_save_path = os.path.join(label_dir, f"{img_basename}.png")
            Image.open(img_path).convert("RGB").save(img_save_path)

            # Plot histogram + marker
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.hist(
                tuning_ensembled_pvalues,
                bins=20,
                density=True,
                color="tab:blue",
                alpha=0.6,
            )

            ax.axvline(ensembled_pvalues[i], color=division_color, linestyle="--", linewidth=2)

            ax.set_yticks([])
            ax.set_xticks([])
            ax.grid(False)
            ax.set_xlim(0, 1)

            plot_path = os.path.join(label_dir, f"{img_basename}_ensemble.png")
            plt.tight_layout()
            plt.savefig(plot_path, bbox_inches="tight", pad_inches=0.1)
            plt.close()


def save_real_statistics_kde(statistics_dict, statistics_keys, output_dir):
    """Save KDE plots of raw statistic distributions for each key."""
    save_dir = os.path.join(output_dir, "real_statistics_kde")
    os.makedirs(save_dir, exist_ok=True)

    for stat_key in statistics_keys:
        if stat_key not in statistics_dict:
            continue

        values = np.asarray(statistics_dict[stat_key])
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.kdeplot(values, fill=True, color="blue", bw_adjust=0.5, ax=ax)

        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_title("")
        ax.grid(False)

        stat_filename = (
            stat_key.replace("=", "-").replace(",", "_").replace(" ", "_")
        )
        plot_path = os.path.join(save_dir, f"{stat_filename}.png")
        plt.tight_layout()
        plt.savefig(plot_path, bbox_inches="tight", pad_inches=0.1)
        plt.close()


def ks_uniform_sanity_check(output_dir, uniform_p_threshold, logger,
                            tuning_real_population_pvals, pvalue_distributions, keys):
    """Filter tests with p-value distributions close to uniform."""
    uniform_tests = []
    ks_pvalues = []

    for i, test_pvalues in tqdm(
            enumerate(pvalue_distributions),
            desc="Filtering uniform pvalues distributions",
            total=len(keys)):
        p_value = kstest(test_pvalues, 'uniform')[1]
        ks_pvalues.append(p_value)
        if p_value > uniform_p_threshold:
            uniform_tests.append(i)

    uniform_keys = [keys[i] for i in uniform_tests]
    uniform_dists = tuning_real_population_pvals[:, uniform_tests]

    plot_uniform_and_nonuniform(pvalue_distributions, uniform_tests, output_dir)
    plot_histograms(ks_pvalues, os.path.join(output_dir, 'ks_pvalues.png'),
                    title='Kolmogorov-Smirnov', bins=20)

    if logger:
        logger.log_param("num_uniform_tests", len(uniform_keys))
        logger.log_param("non_uniform_proportion", (len(keys) - len(uniform_keys)) / len(keys))

    return uniform_keys, uniform_dists


def perform_ensemble_testing(pvalues, ensemble_test, output_dir='logs', plot=False):
    """Perform ensemble testing (Stouffer)."""
    if ensemble_test == 'stouffer':
        return [combine_pvalues(p, method='stouffer')[1] for p in pvalues]
    elif ensemble_test == 'manual-stouffer':
        pvalues = np.clip(pvalues, np.finfo(np.float32).eps, 1.0 - np.finfo(np.float32).eps)
        inverse_z_scores = norm.ppf(pvalues)
        stouffer_z = np.sum(inverse_z_scores, axis=1) / np.sqrt(pvalues.shape[1])
        stouffer_pvalues = norm.cdf(stouffer_z)
        if plot:
            plot_stouffer_analysis(pvalues, inverse_z_scores, stouffer_z, stouffer_pvalues, num_plots_pvalues=5, num_plots_zscores=5, output_folder=output_dir)
        return stouffer_z, stouffer_pvalues
    elif ensemble_test == 'minp':
        # Ensure p-values are within (0,1) for numerical stability
        pvalues = np.clip(pvalues, np.finfo(np.float32).eps, 1.0 - np.finfo(np.float32).eps)
        min_pvals = np.min(pvalues, axis=1)
        n = pvalues.shape[1]
        # Aggregate p-values using the CDF of the min of n uniform(0,1) variables
        aggregated_pvals = 1 - (1 - min_pvals) ** n
        return norm.ppf(aggregated_pvals), aggregated_pvals
    else:
        raise ValueError(f"Invalid ensemble test: {ensemble_test}")


def get_total_size_in_MB(obj):
    """Estimate object size in megabytes using its serialized string length."""
    try:
        obj_str = json.dumps(obj)
    except (TypeError, OverflowError):
        obj_str = str(obj)

    size_bytes = sys.getsizeof(obj_str)
    return size_bytes / (1024 ** 2)


def objective(trial, keys, chi2_p_matrix, pvals_matrix, ensemble_test):
    """Single-objective optimization: minimize |ks_p_value - 0.5|."""
    p_threshold = trial.suggest_float("p_threshold", 0.05, 0.5)
    independent_keys_group = find_largest_independent_group(keys, chi2_p_matrix, p_threshold)
    num_independent_tests = len(independent_keys_group)
    independent_indices = [keys.index(key) for key in independent_keys_group]
    independent_pvals = pvals_matrix[independent_indices].T
    ensembled_stats, _ = perform_ensemble_testing(independent_pvals, ensemble_test)
    _, ks_pvalue = kstest(ensembled_stats, 'norm', args=(0, 1))
    deviation = abs(ks_pvalue - 0.5)
    trial.set_user_attr("independent_keys_group", independent_keys_group)
    trial.set_user_attr("num_independent_tests", num_independent_tests)
    trial.set_user_attr("ks_p_value", ks_pvalue)
    return deviation


def finding_optimal_independent_subgroup(keys, chi2_p_matrix, pvals_matrix, ensemble_test, n_trials=50):
    """Find independent subgroup minimizing KS deviation while maximizing size."""
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, keys, chi2_p_matrix, pvals_matrix, ensemble_test),
                   n_trials=n_trials, show_progress_bar=True)
    valid_trials = [t for t in study.trials if t.value is not None and t.value <= 0.25]
    if not valid_trials:
        raise ValueError("No valid trials found with deviation <= 0.25")
    best_trial = max(valid_trials, key=lambda t: t.user_attrs["num_independent_tests"])
    independent_keys_group = best_trial.user_attrs["independent_keys_group"]
    best_results = {
        'best_KS': best_trial.user_attrs["ks_p_value"],
        'best_N': best_trial.user_attrs["num_independent_tests"],
        'best_alpha_threshold': best_trial.params['p_threshold']
    }
    optimization_data = {
        'thresholds': [t.params["p_threshold"] for t in study.trials if t.value is not None],
        'ks_pvalues': [t.user_attrs["ks_p_value"] for t in study.trials if t.value is not None],
        'num_tests': [t.user_attrs["num_independent_tests"] for t in study.trials if t.value is not None]
    }
    return independent_keys_group, best_results, optimization_data


def uncorrelation_objective(trial, keys, corr_matrix, pvals_matrix, ensemble_test):
    """Objective for uncorrelated subgroup search."""
    p_threshold = trial.suggest_float("p_threshold", 0.0, 0.05)
    independent_keys_group = find_largest_uncorrelated_group(keys, corr_matrix, p_threshold)
    num_independent_tests = len(independent_keys_group)
    independent_indices = [keys.index(key) for key in independent_keys_group]
    independent_pvals = pvals_matrix[independent_indices].T
    ensembled_stats, _ = perform_ensemble_testing(independent_pvals, ensemble_test)
    _, ks_pvalue = kstest(ensembled_stats, 'norm', args=(0, 1))
    deviation = abs(ks_pvalue - 0.5)
    trial.set_user_attr("independent_keys_group", independent_keys_group)
    trial.set_user_attr("num_independent_tests", num_independent_tests)
    trial.set_user_attr("ks_p_value", ks_pvalue)
    return deviation


def finding_optimal_uncorrelated_subgroup(keys, corr_matrix, pvals_matrix, ensemble_test, n_trials=50):
    """Find uncorrelated subgroup via optimization."""
    independent_keys_group = find_largest_uncorrelated_group(keys, corr_matrix, 0.05)
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: uncorrelation_objective(t, keys, corr_matrix, pvals_matrix, ensemble_test),
                   n_trials=n_trials, show_progress_bar=True)
    valid_trials = [t for t in study.trials if t.value is not None and t.value <= 0.25]
    if not valid_trials:
        raise ValueError("No valid trials found with deviation <= 0.25")
    best_trial = max(valid_trials, key=lambda t: t.user_attrs["num_independent_tests"])
    independent_keys_group = best_trial.user_attrs["independent_keys_group"]
    best_results = {
        'best_KS': best_trial.user_attrs["ks_p_value"],
        'best_N': best_trial.user_attrs["num_independent_tests"],
        'best_alpha_threshold': best_trial.params['p_threshold']
    }
    optimization_data = {
        'thresholds': [t.params["p_threshold"] for t in study.trials if t.value is not None],
        'ks_pvalues': [t.user_attrs["ks_p_value"] for t in study.trials if t.value is not None],
        'num_tests': [t.user_attrs["num_independent_tests"] for t in study.trials if t.value is not None]
    }
    return independent_keys_group, best_results, optimization_data


def finding_optimal_independent_subgroup_deterministic(
    keys, chi2_p_matrix, pvals_matrix, ensemble_test, fake_pvals_matrix,
    ks_pvalue_abs_threshold=0.25, minimal_p_threshold=0.05, preferred_statistics=None):
    """Deterministic search over cliques based on KS range and AUC, favoring preferred statistics."""

    preferred_lookup = {
        token.lower()
        for stat in preferred_statistics or ()
        if stat is not None and (token := str(stat).strip())
    }
    preferred_total = len(preferred_lookup)
    optimization_data = {'thresholds': [], 'ks_pvalues': [], 'num_tests': []}
    candidates = []
    cliques = find_largest_independent_group_iterative(
        keys, chi2_p_matrix, p_threshold=minimal_p_threshold, test_type="cramer_v")
    for clique in tqdm(cliques, total=len(cliques), desc="Searching for optimial clique..."):
        independent_keys_group = list(clique)
        num_independent_tests = len(independent_keys_group)
        independent_indices = [keys.index(key) for key in independent_keys_group]
        independent_pvals = pvals_matrix[independent_indices].T
        _, ensembled_pvals = perform_ensemble_testing(independent_pvals, ensemble_test)
        ensembled_pvals_subsampled = np.random.choice(ensembled_pvals, size=1000, replace=False)
        _, ks_pvalue = kstest(ensembled_pvals_subsampled, 'uniform', args=(0, 1))
        ks_deviation = abs(ks_pvalue - 0.5)
        if ks_deviation > ks_pvalue_abs_threshold:
            continue
        optimization_data['thresholds'].append(minimal_p_threshold)
        optimization_data['ks_pvalues'].append(ks_pvalue)
        optimization_data['num_tests'].append(num_independent_tests)
        matched_preferred = {
            preferred_token
            for preferred_token in preferred_lookup
            if any(preferred_token in key.lower() for key in independent_keys_group)
        }
        candidates.append({
            'group': independent_keys_group,
            'size': num_independent_tests,
            'ks_pvalue': ks_pvalue,
            'ks_deviation': ks_deviation,
            'matched_preferred': matched_preferred,
        })
    if not candidates:
        raise ValueError("No valid groups found within the KS p-value range")
    if preferred_total:
        key_fn = lambda c: (len(c['matched_preferred']), c['size'])
    else:
        key_fn = lambda c: (c['size'],)
    best_candidate = max(candidates, key=key_fn)
    best_results = {
        'best_KS': best_candidate['ks_pvalue'],
        'best_N': best_candidate['size'],
        'best_alpha_threshold': minimal_p_threshold,
    }
    if preferred_total:
        preferred_hits = len(best_candidate['matched_preferred'])
        best_results['preferred_hits'] = preferred_hits
        best_results['preferred_total'] = preferred_total
        best_results['preferred_coverage_ratio'] = preferred_hits / preferred_total
        best_results['preferred_missing'] = preferred_total - preferred_hits
    return best_candidate['group'], best_results, optimization_data

