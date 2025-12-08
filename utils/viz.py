from cfg import load_cfg
import matplotlib.pyplot as plt
from matplotlib import colors
cfg = load_cfg()
import numpy as np
import seaborn as sns
import pandas as pd
import einops
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
from sklearn.base import clone
import os
from datetime import datetime


# ---------------------- SAVE FIG HELPER ----------------------

def save_fig(name):
    """Save figure to results/plots/<name>.png with auto-directory creation."""
    out_dir = "results/plots"
    os.makedirs(out_dir, exist_ok=True)
    safe = name.replace(" ", "_").replace("/", "_")
    path = f"{out_dir}/{safe}.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    return path


# ---------------------- DIRECTION FUNCTIONS ----------------------

def get_direction(data, labels, model):
    model = clone(model)
    model.fit(data, labels)
    coefficients = model.coef_[0]
    intercept = model.intercept_[0]
    theta = np.hstack([intercept, coefficients])
    return theta / np.linalg.norm(theta)


def get_direction_with_constraint(data, labels, model, first_direction):

    projection_on_theta1 = np.dot(data, first_direction)
    data_orthogonalized = data - np.outer(
        projection_on_theta1, first_direction
    ) / np.dot(first_direction, first_direction)

    model = clone(model)
    model.fit(data_orthogonalized[:, 1:], labels)

    theta = np.hstack([model.intercept_[0], model.coef_[0]])
    return theta / np.linalg.norm(theta)


# ---------------------- PLOTTING FUNCTIONS ----------------------

def plot_line(x, title="DummyTitle", x_axis="Layer", y_axis="Accuracy", label="Model"):

    fig, ax = plt.subplots(figsize=(12, 8), facecolor='w')
    ax.set_facecolor('#e0e0e0')
    ax.plot(x, color='#007acc', alpha=0.7, marker='o',
            markersize=8, linewidth=2.5, label=label)

    ax.set_title(title, fontsize=18, pad=20, weight='bold', color='#333333')
    ax.set_xlabel(x_axis, fontsize=14, labelpad=10, color='#333333')
    ax.set_ylabel(y_axis, fontsize=14, labelpad=10, color='#333333')

    ax.grid(visible=True, which='major', color='#f7f7f7',
            linewidth=1.5, linestyle='--')
    ax.tick_params(axis='both', which='major',
                   labelsize=12, color='#555555')

    legend = ax.legend(fontsize=12, loc='upper right', frameon=True)
    legend.get_frame().set_facecolor('#ffffff')
    legend.get_frame().set_edgecolor('#e0e0e0')
    legend.get_frame().set_alpha(0.9)

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    save_fig(f"line_{title}")


def plot_heat(accuracies, title="DummyTitle", x_axis="Heads (Sorted)",
              y_axis="Layers (Bottom-Up)", model="Model", probe="Probe"):

    accuracies = np.array(accuracies)
    sorted_accuracies = np.sort(accuracies, axis=1)[:, ::-1]
    sorted_accuracies = sorted_accuracies[::-1, :]

    norm = colors.Normalize(
        vmin=sorted_accuracies.min(),
        vmax=max(sorted_accuracies.max(), 0.75)
    )

    plt.figure(figsize=(10, 8))
    sns.set(style="whitegrid")
    ax = sns.heatmap(
        sorted_accuracies,
        annot=False,
        fmt=".2f",
        cmap="cividis",
        cbar_kws={"shrink": 0.9, "aspect": 22},
        linewidths=0,
        linecolor="white",
        norm=norm
    )

    num_layers = sorted_accuracies.shape[0]
    num_heads = sorted_accuracies.shape[1]

    ax.set_yticks(np.arange(num_layers) + 0.5)
    ax.set_yticklabels(np.arange(num_layers - 1, -1, -1), fontsize=10)
    ax.set_xticks([])

    plt.suptitle(title, fontsize=18)
    plt.title(f"Model: {model} | Probe: {probe}")
    plt.xlabel(x_axis, fontsize=12, labelpad=10)
    plt.ylabel(y_axis, fontsize=12, labelpad=10)

    ax.hlines(np.arange(1, num_layers), *ax.get_xlim(),
              colors="white", linestyles="solid", linewidth=0.2)
    ax.vlines(np.arange(1, num_heads), *ax.get_ylim(),
              colors="white", linestyles="solid", linewidth=0.2)

    plt.tight_layout()
    save_fig(f"heat_{title}")


def kde(data, x, y, labels, x_label, y_label, title,
        x_range=None, y_range=None, kernel=True, scatter=True, offset=1):

    g = sns.jointplot(
        data=data,
        x=x,
        y=y,
        hue=labels,
        kind='kde',
        palette='tab10',
        linewidths=0.8 if kernel else 0,
        alpha=1,
        bw_offset=offset,
        marginal_kws={'fill': True, 'common_norm': False,
                      'alpha': 0.3, 'linewidth': 0.8}
    )

    if scatter:
        sns.scatterplot(
            data=data,
            x=x,
            y=y,
            hue=labels,
            palette='coolwarm',
            marker='o',
            s=10,
            edgecolor='black',
            linewidth=0.2,
            alpha=0.7,
            ax=g.ax_joint
        )

    if x_range:
        g.ax_joint.set_xlim(*x_range)
    if y_range:
        g.ax_joint.set_ylim(*y_range)

    g.ax_joint.grid(True, linestyle='--', alpha=0.6)
    g.fig.suptitle(title, fontsize=12)
    g.fig.tight_layout()

    save_fig(f"kde_{title}")
    plt.close("all")


def plot_kde_scatter(data, labels, model, n_dir=2, zoom_strength=0,
                     offset=1, kernel=True, scatter=True, pca=False):

    assert data.shape[0] == labels.shape[0]

    if len(data.shape) > 2:
        data = einops.rearrange(data,
                                'n_batch batch_size d_model -> (n_batch batch_size) d_model')
    if len(labels.shape) > 1:
        labels = einops.rearrange(labels,
                                  'n_batch batch_size -> (n_batch batch_size)')

    data = np.array(data.cpu(), dtype=np.float32)
    labels = np.array(labels.cpu(), dtype=np.int32)

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    if pca:
        pca_model = PCA(n_components=2)
        pca_projections = pca_model.fit_transform(data)

        x_min, x_max = np.percentile(
            pca_projections[:, 0], [0+zoom_strength, 100-zoom_strength])
        y_min, y_max = np.percentile(
            pca_projections[:, 1], [0+zoom_strength, 100-zoom_strength])

        df = pd.DataFrame({
            'PCA1': pca_projections[:, 0],
            'PCA2': pca_projections[:, 1],
            'Label': ['False' if l == 0 else 'True' for l in labels]
        })

        kde(
            data=df,
            x='PCA1',
            y='PCA2',
            labels='Label',
            x_label='PCA1',
            y_label='PCA2',
            title="PCA_with_KDE",
            x_range=(x_min, x_max),
            y_range=(y_min, y_max),
            kernel=kernel,
            scatter=scatter,
            offset=offset
        )
        return

    first_dir = get_direction(data, labels, model)
    data_with_bias = np.hstack([np.ones((data.shape[0], 1)), data])
    first_proj = np.dot(data_with_bias, first_dir)

    if n_dir == 1:
        x_min, x_max = np.percentile(
            first_proj, [0+zoom_strength, 100-zoom_strength])

        plt.figure(figsize=(8, 6))
        for c in np.unique(labels):
            class_proj = first_proj[labels == c]
            density = gaussian_kde(class_proj, bw_method='scott')
            xs = np.linspace(x_min, x_max, 500)
            ys = density(xs)
            plt.plot(xs, ys, label=f'Class {c}')
            plt.fill_between(xs, ys, alpha=0.3)

        plt.axvline(x=0, color='black', linestyle='--')
        plt.xlabel('Projection onto LogReg Direction')
        plt.ylabel('Density')
        plt.title('Class Separation Along LogReg Direction')
        plt.xlim(x_min, x_max)
        plt.legend()

        save_fig("kde_1d")
        return

    second_dir = get_direction_with_constraint(
        data_with_bias, labels, model, first_dir)
    second_proj = np.dot(data_with_bias, second_dir)

    x_min, x_max = np.percentile(
        first_proj, [0+zoom_strength, 100-zoom_strength])
    y_min, y_max = np.percentile(
        second_proj, [0+zoom_strength, 100-zoom_strength])

    df = pd.DataFrame({
        'First direction': first_proj,
        'Second direction': second_proj,
        'Label': ['False' if l == 0 else 'True' for l in labels]
    })

    kde(
        data=df,
        x='First direction',
        y='Second direction',
        labels='Label',
        x_label='First direction',
        y_label='Second direction',
        title="KDE_2D",
        x_range=(x_min, x_max),
        y_range=(y_min, y_max),
        kernel=kernel,
        scatter=scatter,
        offset=offset
    )


def plot_sweep(data, ks, alphas, metric="DummyMetric", custom_subtitle=None):

    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    ax.set_aspect('equal')

    sns.heatmap(data, annot=True, fmt=".3f", cmap="Blues",
                cbar=False, linewidths=0.1, linecolor='grey')

    ax.set_yticks(np.arange(data.shape[0]) + 0.5, ks)
    ax.set_xticks(np.arange(data.shape[1]) + 0.5, alphas)

    if custom_subtitle is not None:
        plt.suptitle(f"Intervention effect | metric: {metric}", fontsize=16)
        plt.title(f"{custom_subtitle}")
    else:
        plt.title(f"Intervention effect | metric: {metric}", fontsize=16, pad=16)

    plt.xlabel("Alpha", labelpad=10)
    plt.ylabel("K", labelpad=10)

    save_fig(f"sweep_{metric}")
