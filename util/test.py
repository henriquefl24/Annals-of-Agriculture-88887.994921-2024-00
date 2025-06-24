import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample


def bootstrap_cov(x, y, num_samples=1000):
    """
    Calcular intervalos de confiança bootstrap para a covariância.
    """
    covs = []
    for _ in range(num_samples):
        x_sample, y_sample = resample(x, y)
        cov = np.cov(x_sample, y_sample)
        if cov.shape == (2, 2) and not np.isnan(cov).any():
            covs.append(cov)
    return np.percentile(covs, [2.5, 97.5], axis=0)


def plot_with_ellipses(ax, x, y, label, color):
    """
    Plot the data points and optionally add ellipses for covariance using bootstrap.
    """
    lower_cov, upper_cov = bootstrap_cov(x, y)
    # Mean coordinates
    mean_x, mean_y = np.mean(x), np.mean(y)
    # Lower confidence interval ellipse
    lower_ellipse = Ellipse(xy=(mean_x, mean_y),
                            width=2 * np.sqrt(lower_cov[0, 0]),
                            height=2 * np.sqrt(lower_cov[1, 1]),
                            edgecolor='black', fc=color, lw=2, alpha=0.1, label=f"Lower CI for {label}")
    ax.add_patch(lower_ellipse)
    # Upper confidence interval ellipse
    upper_ellipse = Ellipse(xy=(mean_x, mean_y),
                            width=2 * np.sqrt(upper_cov[0, 0]),
                            height=2 * np.sqrt(upper_cov[1, 1]),
                            edgecolor='black', fc=color, lw=2, alpha=0.3, label=f"Upper CI for {label}")
    ax.add_patch(upper_ellipse)


def test_num_samples(x, y, max_samples=5000, step=500):
    """
    Testar diferentes valores de num_samples e observar a largura média dos intervalos de confiança.
    """
    widths = []
    samples_range = range(step, max_samples + step, step)
    for samples in samples_range:
        lower_cov, upper_cov = bootstrap_cov(x, y, num_samples=samples)
        width = np.mean(upper_cov - lower_cov)
        widths.append(width)
    return samples_range, widths


def biplot(pca, scores, variable_names, labels, fig_path, ellipse=False, **kwargs):
    fig, ax = plt.subplots(figsize=(10, 7))
    # Codificar os labels
    label_encoder = LabelEncoder()
    label_colors = label_encoder.fit_transform(labels)
    unique_labels = label_encoder.classes_
    cmap = plt.get_cmap('tab20', len(unique_labels))
    # Plotando os scores
    scatter = ax.scatter(scores[:, 0], scores[:, 1], c=label_colors, cmap=cmap, **kwargs)

    # Adicionando rótulos aos pontos
    for i, txt in enumerate(labels):
        ax.annotate(txt, (scores[i, 0], scores[i, 1]), fontsize=10, ha='right')

    # Plotando elipses se necessário
    if ellipse:
        for label in np.unique(label_colors):
            data_subset = scores[label_colors == label]
            if data_subset.shape[0] < 2:
                continue
            x = data_subset[:, 0]
            y = data_subset[:, 1]
            color = cmap(label / len(unique_labels))
            if np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
                plot_with_ellipses(ax, x, y, label_encoder.inverse_transform([label])[0], color)
    # Anotando as variáveis (cargas dos componentes)
    for i, (comp, var_name) in enumerate(zip(pca.components_.T, variable_names)):
        ax.arrow(0, 0, comp[0] * 2, comp[1] * 2, color='r', head_width=0.05)
        ax.text(comp[0] * 2.2, comp[1] * 2.2, f"{var_name}", color='black', ha='center', va='center', weight='bold',
                size=10)
    plt.axhline(0, color='black', lw=1)
    plt.axvline(0, color='black', lw=1)
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%})')
    print(f"Explained variance: {pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]:.2%}")
    # Adicionando manualmente a legenda com todas as classes
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i / len(unique_labels)), markersize=10)
               for i in range(len(unique_labels))]
    legend_labels = [unique_labels[i] for i in range(len(unique_labels))]
    #ax.legend(handles, legend_labels, title="Classes", loc="upper right", ncols=2)
    if fig_path:
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()
