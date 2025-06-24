import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse
from numpy import exp, log, abs, nan, pi, sin, cos, tan, radians, arccos, degrees
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder


def pt_heatmap_plot(posthoc_results: DataFrame, data: DataFrame, dv: str, group: str, group2: str, alpha: float = 0.05):
    def assign_letters(posthoc_results: DataFrame, alpha: float = 0.05) -> dict:
        groups = {}
        letters = {}
        current_letter = 'A'
        for i, row in posthoc_results.iterrows():
            if row['p-tukey'] > alpha:
                if row['A'] not in groups:
                    groups[row['A']] = current_letter
                    letters[row['A']] = current_letter
                if row['B'] not in groups:
                    groups[row['B']] = current_letter
                    letters[row['B']] = current_letter
            else:
                if row['A'] not in groups:
                    groups[row['A']] = current_letter
                    letters[row['A']] = current_letter
                    current_letter = chr(ord(current_letter) + 1)
                if row['B'] not in groups:
                    groups[row['B']] = current_letter
                    letters[row['B']] = current_letter
                    current_letter = chr(ord(current_letter) + 1)
                else:
                    if letters[row['A']] == letters[row['B']]:
                        current_letter = chr(ord(current_letter) + 1)
                        letters[row['B']] = current_letter

        return letters

    letters = assign_letters(posthoc_results, alpha)

    # Create pivot table
    pivot_table = data.pivot_table(values=dv, index=group, columns=group2, aggfunc='mean', sort=False)

    # Create a DataFrame for annotations
    annot_table = pivot_table.copy().astype(float)

    # Add letters to the annotation table
    for rep in pivot_table.index:
        for estadio in pivot_table.columns:
            if rep in letters:
                annot_table.loc[rep, estadio] = f"{pivot_table.loc[rep, estadio]:.2f}\n{letters[rep]}"
            else:
                annot_table.loc[rep, estadio] = f"{pivot_table.loc[rep, estadio]:.2f}"

    # Plot heatmap

    plt.figure(dpi=600)
    sns.set_theme(style="white", context="paper")
    sns.heatmap(pivot_table, annot=annot_table, fmt='', cmap='coolwarm', annot_kws={"size": 5}, cbar_kws={"shrink": .8})
    plt.title(f'Heatmap of {group2} vs. {group} Interaction on {dv}')
    plt.xlabel(group2)
    plt.ylabel(group)

    plt.tight_layout()
    plt.show()


# Radiação Atmosférica

def timestamp_to_dj(timestamp):
    # Transformando Timestamp para o dia juliano do ano (1-365/366)
    return timestamp.timetuple().tm_yday


def conversor_coord(grau, minuto, segundo, hem='S', long='W'):
    """
    :param grau: The degree component of the coordinate.
    :param minuto: The minute component of the coordinate.
    :param segundo: The second component of the coordinate.
    :param hem: Hemisphere indicator for latitude ('N' for North, 'S' for South) (default 'S').
    :param long: Hemisphere indicator for longitude ('E' for East, 'W' for West) (default 'W').
    :return: The coordinate in decimal degrees, negative if in the southern hemisphere or western longitude.
    """
    min_calc = segundo / 60 + minuto / 60
    coords = grau + min_calc

    if hem == 'S' or long == 'W':
        return -coords
    else:
        return coords


def dr(nda):
    """
    Calcula a distância entre a Terra e o Sol em diferentes épocas do ano
    em unidades astronômicas (UA)
    param NDA: número de dias julianos
    return DR: valor da distãncia em UA
    """
    NDA = nda
    DR = 1 + 0.033 * cos(radians(360 * NDA / 365))
    return DR


def declinacao_solar(nda):
    """
    Cálculo da declinação solar em graus
    param nda: numero de dias julianos
    return decl_sol: Declinação solar em graus
    """

    decl_sol = 23.45 * sin(radians(360 / 365) * (nda - 80))
    return decl_sol


def ang_nasc_sol(latitude, nda):
    """
    Angulo no nascer do sol
    param latitude: latitude do local em graus
    param nda: Numero de dias juliananos (1 a 365)
    return: hora do nascer do sol em graus
    """
    caldeci = declinacao_solar(nda)
    tanlatitude = tan(radians(latitude))
    tandecli = tan(radians(caldeci))
    hnascer = degrees(arccos(-tanlatitude * tandecli))
    return hnascer


def jo_linha(DR):
    return 1367 * DR


def Qo(latitude, nda):
    """
    Total de energia solar diário (MJ/m² dia)
    param latitude: latitude do local em graus
    param nda: numero de dias julianos (1 a 365)
    return: Qo (MJ/m² dia)
    """
    caldecli = declinacao_solar(nda)
    caldr = dr(nda)
    calhn = ang_nasc_sol(latitude, nda)
    senlat = sin(radians(latitude))
    sendecli = sin(radians(caldecli))
    coslat = cos(radians(latitude))
    cosdecli = cos(radians(caldecli))
    senhn = sin(radians(calhn))

    return 37.6 * caldr * ((pi / 180) * calhn * senlat * sendecli + coslat * cosdecli * senhn)


def fotop(latitude, nda):
    """
    Calculo do fotoperíodo (Horas)
    param latitude: latitude do local em decimais
    param nda: numero de dias julianos
    return N: Fotoperiodo (horas)
    """
    hn_calc = ang_nasc_sol(latitude, nda)
    return 2 * hn_calc / 15


def par(qg):
    return 0.5 * qg


def etp_camargo(lat, ndp, t2m):
    """
    Calculo da evapotranspiração segundo Camargo (1971)
    param lat: latitude do local (decimal)
    param ndp: dias julianos (adimensional)
    param t2m: temperatura média do local (°C)
    return: ETP (mm/d)
    """
    return round((0.01 * Qo(lat, ndp) / 2.45 * t2m * ndp) / ndp, 2)


def balanco_hidrico(data, CAD):
    """
    Calcula o balanço hídrico com base em um dataset fornecido e a Capacidade de Água Disponível (CAD).

    Args:
        data: DataFrame contendo precipitação (PREC) e evapotranspiração potencial (ETp).
        CAD: A Capacidade de Água Disponível, uma constante usada nos cálculos.

    Returns:
        O DataFrame 'data' com colunas adicionais para diferentes componentes do balanço hídrico:
        - 'P-ETP': A diferença entre precipitação e evapotranspiração potencial.
        - 'NAC': A acumulação líquida de água.
        - 'ARM': A água disponível real em armazenamento.
        - 'ALT': A alteração nos níveis de água.
        - 'ETR': A evapotranspiração real.
        - 'DEF': A deficiência de água.
        - 'EXC': O excesso de água.
    """

    data['P-ETP'] = data.PREC - data.ETp
    data['NAC'] = nan
    data['ARM'] = nan
    data['ALT'] = nan
    data['ETR'] = nan
    data['DEF'] = nan
    data['EXC'] = nan

    for i in data.index:

        # |CALCULO DE ARM E NAC|
        if i == 0:
            NACant = 0
            ARMant = CAD / 2
        else:
            NACant = data.NAC[i - 1]
            ARMant = data.ARM[i - 1]

        if data['P-ETP'][i] < 0:
            data.NAC[i] = NACant + data['P-ETP'][i]
            data.ARM[i] = CAD * exp(data.NAC[i] / CAD)
            if data.ARM[i] > CAD:
                data.ARM[i] = CAD
        else:
            data.ARM[i] = data['P-ETP'][i] + ARMant
            if data.ARM[i] > CAD:
                data.ARM[i] = CAD
            data.NAC[i] = CAD * log(data.ARM[i] / CAD)

        # |CALCULO DE ALT|
        data.ALT[i] = data.ARM[i] - ARMant

        # |CALULO DE ETR|
        if data.ALT[i] < 0:
            data.ETR[i] = data.PREC[i] + abs(data.ALT[i])
        else:
            data.ETR[i] = data.ETp[i]

        # |CALCULO DE DEF|
        data.DEF[i] = data.ETp[i] - data.ETR[i]

        # |CALCULO DE EXC|
        if data.ARM[i] < CAD:
            data.EXC[i] = 0
        else:
            data.EXC[i] = data['P-ETP'][i] - data.ALT[i]
    return data


def create_dataframe(clima, cultivar, rep):
    """
    :param clima: Input data containing climate information.
    :param cultivar: The cultivar data to be added as a new column in the dataframe.
    :param rep: The rep data to be added as a new column in the dataframe.
    :return: A pandas DataFrame with provided climate information and additional columns for cultivar, rep, and estadio.
    """
    # Constantes
    CULTIVAR_COL = 'Cultivar'
    REP_COL = 'Rep'
    ESTADIO_COL = 'Estadio'

    df = DataFrame(clima)
    df[CULTIVAR_COL] = cultivar
    df[REP_COL] = rep
    df[ESTADIO_COL] = None
    return df


def update_dataframe_with_phenology(clima, fenologia, df, cultivar, rep, estadio):
    """
    :param clima: DataFrame containing climate data. It should have a 'Dia' column which holds date values.
    :param fenologia: DataFrame containing phenology data. It should have
    cultivar and replication columns and an additional column specific to phenological stages.
    :param df: DataFrame to be updated with phenological stage information. It should have a 'Dia' column.
    :param cultivar: The specific cultivar to filter the phenology data on.
    :param rep: The specific replication to filter the phenology data on.
    :param estadio: The specific phenological stage to update in the DataFrame df.
    :return: None
    """
    # Constantes
    CULTIVAR_COL = 'Cultivar'
    REP_COL = 'Rep'
    ESTADIO_COL = 'Estadio'

    fenologia_match = fenologia.loc[
        (fenologia[CULTIVAR_COL] == cultivar) & (fenologia[REP_COL] == rep), estadio].unique()

    if not clima.empty:
        clima_datas = clima.Dia.iloc[0:].values

    for data in clima_datas:
        if data == fenologia_match:
            df.loc[df['Dia'] == data, ESTADIO_COL] = estadio
        else:
            pass


def create_dataframes(clima, fenologia, cultivares, repeticao, estadios):
    """
    :param clima: Climatic data to be used in DataFrame creation
    :param fenologia: Phenological data used to update DataFrames
    :param cultivares: List of cultivars for which DataFrames are to be created
    :param repeticao: List of repetitions to be applied for each cultivar
    :param estadios: List of stages to update each DataFrame with phenological data
    :return: A list of DataFrames created and updated with phenological data
    """
    dataframes_list = []

    for cultivar in cultivares:
        for rep in repeticao:  # Verifique se a variável repetition está definida corretamente
            df = create_dataframe(clima, cultivar, rep)

            for estadio in estadios:
                update_dataframe_with_phenology(clima, fenologia, df, cultivar, rep, estadio)

                # Append the DataFrame to the list
            dataframes_list.append(df)

    return dataframes_list


def fill_none_with_previous_value(df):
    """
    :param df: A pandas DataFrame that may contain 'None' values.
    :return: A pandas DataFrame with 'None' values filled by the previous value in the same column.
    """
    df.fillna(inplace=True, method='ffill')
    return df


def calcular_soma_sdg(subset_df, START_STAGE, END_STAGE):
    start_date = subset_df.loc[subset_df['Estadio'] == START_STAGE, 'Dia'].min()
    end_date = subset_df.loc[subset_df['Estadio'] == END_STAGE, 'Dia'].min()
    if pd.notna(start_date) and pd.notna(end_date) and start_date <= end_date:
        interval_df = subset_df[(subset_df['Dia'] >= start_date) & (subset_df['Dia'] <= end_date)]
        return interval_df['SDG'].sum()
    return None


def plot_bar_with_annotations(df, annotation_df, groups, values):
    """
    :param df: DataFrame contendo os dados a serem visualizados. Ele deve ter colunas especificadas em 'groups' e 'values'.
    :param annotation_df: DataFrame contendo as anotações. Deve ter pelo menos duas colunas, a primeira para identificar a barra e a segunda para a anotação correspondente.
    :param groups: O nome da coluna no DataFrame 'df' que contém os grupos (ex: "Cultivar").
    :param values: O nome da coluna no DataFrame 'df' que contém os valores (ex: "SDG_sum").
    :return: None
    """
    # Ordena o DataFrame principal 'df' baseado na ordem de 'annotation_df'
    ordered_df = df.set_index(groups).loc[annotation_df.iloc[:, 0]].reset_index()

    plt.figure(figsize=(15, 5), dpi=300)
    sns.set_theme(style='white', context="paper")

    # Cria o gráfico de barras usando o DataFrame ordenado
    bar_plot = sns.barplot(data=ordered_df, y=values, x=groups, palette="Set3", errorbar="sd")

    # Cria um dicionário para mapear Cultivar para anotação
    annotation_dict = dict(zip(annotation_df.iloc[:, 0], annotation_df.iloc[:, 3]))

    # Atualiza os rótulos das barras para seguir a ordem de `annotation_df["groups"]`
    ordered_labels = annotation_df[0].tolist()
    bar_plot.set_xticklabels(ordered_labels)

    # Itera sobre as barras do gráfico e adiciona as anotações
    for p, cultivar in zip(bar_plot.patches, ordered_df[groups].unique()):
        height = p.get_height()
        annotation = annotation_dict.get(cultivar, '')
        bar_plot.annotate(
            annotation,
            (p.get_x() + p.get_width() / 2., height),
            ha='center',
            va='bottom',
            fontsize=10,
            color='black'
        )

    plt.legend("")
    plt.show()


def plot_with_ellipses(ax, x, y, label, color):
    """
    Plot the data points and optionally add ellipses for covariance.
    """
    cov = np.cov(x, y)
    if cov.shape != (2, 2) or np.isnan(cov).any():
        return
    ellipse = Ellipse(xy=(np.mean(x), np.mean(y)),
                      width=2 * np.sqrt(cov[0, 0]), height=2 * np.sqrt(cov[1, 1]),
                      edgecolor='black', fc=color, lw=2, alpha=0.1, label=f"Cov ellipse for {label}")
    ax.add_patch(ellipse)


def biplot(pca, scores, variable_names, labels, fig_path, ellipse=False, **kwargs):
    fig, ax = plt.subplots(figsize=(10, 7))

    # Codificar os labels
    label_encoder = LabelEncoder()
    label_colors = label_encoder.fit_transform(labels)
    unique_labels = label_encoder.classes_
    cmap = plt.get_cmap('tab20', len(unique_labels))

    # Plotando os scores
    scatter = ax.scatter(scores[:, 0], scores[:, 1], c=label_colors, cmap=cmap, **kwargs)

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
        ax.text(comp[0] * 2.2, comp[1] * 2.2, f"{var_name}", color='black', ha='center',
                va='center', weight='bold', size=10)

    plt.axhline(0, color='black', lw=1)
    plt.axvline(0, color='black', lw=1)
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%})')
    print(f"Explained variance: {pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]:.2%}")

    # Adicionando manualmente a legenda com todas as classes
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i / len(unique_labels)), markersize=10)
               for i in range(len(unique_labels))]
    legend_labels = [unique_labels[i] for i in range(len(unique_labels))]
    ax.legend(handles, legend_labels, title="Classes", loc="lower right", ncols=2)

    plt.savefig(fig_path)
    plt.tight_layout()
    plt.show()
    plt.close()


def to_normality(data: DataFrame, method: str) -> DataFrame:
    r"""
    Metodo traz dados para normalidade.
    :param data: Pandas Dataframe
    :param method: "boxcox" ou "yeojohnson"
    :return: Dataframe transformado
    """

    from scipy.stats import boxcox, yeojohnson
    import pandas as pd

    i = data.index
    data = data.astype(float)

    if method == "boxcox":

        data = data.apply(func=boxcox)

    elif method == "yeojohnson":

        data = data.apply(func=yeojohnson)

    data = data.apply(func=pd.Series.explode).reset_index(drop=True)
    data, last_row = data.drop(data.tail(1).index), data.tail(1)
    data.set_index(i, inplace=True)

    data = data.astype(float)

    return data
