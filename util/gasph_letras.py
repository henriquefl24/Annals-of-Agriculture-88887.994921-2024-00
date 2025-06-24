def letras(resultado, df, valorp):
    '''
    resultado: resultado da ANOVA
    df: Data Frame original dos dados
        (obs importante: o nome da coluna de tratamento deve ser 'Groups'
                         o nome da coluna de dados deve ser 'Data')
    valorp: 0.05, 0.01, 0.1 ou qqer valor de interesse
    '''

    import statsmodels.stats.multicomp as ml
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    col_to_group = 'Groups'
    col_for_data = 'Data'

    # Pegando o iésimo dado e reagrupando para anova
    samples = [cols[1] for cols in df.groupby(col_to_group)[col_for_data]]

    ###### M O D E L O ######
    #########################

    # f_val, p_val = stats.f_oneway(*samples)

    # modelo = ols('peso ~ racao', data=dados).fit()
    # resultado= sm.stats.anova_lm(modelo, typ=1)

    f_val = resultado.F[0]
    p_val = resultado["PR(>F)"][0]

    #########################
    #########################

    print(f'F value: {f_val:.3f}, p value: {p_val:.3f}\n')

    # somente faz o teste POST-HOC se o p-valor da anova< valorp
    # global mod, df, col_for_data, col_to_group
    # global mod
    # global thsd
    # if p_val<valorp:
    mod = ml.MultiComparison(df[col_for_data], df[col_to_group])
    thsd = mod.tukeyhsd(valorp)  # 0.01 , 0.05- padrão !!!
    print(mod.tukeyhsd(valorp))

    tot = len(thsd.groupsunique)

    df_ltr = pd.DataFrame(np.nan, index=np.arange(tot), columns=np.arange(tot))
    df_ltr.iloc[:, 0] = 1
    count = 0
    df_nms = pd.DataFrame('', index=np.arange(tot), columns=['names'])

    for i in np.arange(tot):
        for j in np.arange(i + 1, tot):
            #
            # print('i=',i,'j=',j,thsd.reject[count])
            if thsd.reject[count] == True:
                for cn in np.arange(tot):
                    if df_ltr.iloc[i, cn] == 1 and df_ltr.iloc[
                        j, cn] == 1:  # If the column contains both i and j shift and duplicat
                        df_ltr = pd.concat([df_ltr.iloc[:, :cn + 1], df_ltr.iloc[:, cn + 1:].T.shift().T], axis=1)
                        df_ltr.iloc[:, cn + 1] = df_ltr.iloc[:, cn]
                        df_ltr.iloc[i, cn] = 0
                        df_ltr.iloc[j, cn + 1] = 0

                    for cleft in np.arange(len(df_ltr.columns) - 1):
                        for cright in np.arange(cleft + 1, len(df_ltr.columns)):
                            if (df_ltr.iloc[:, cleft].isna()).all() == False and (
                                    df_ltr.iloc[:, cright].isna()).all() == False:
                                if (df_ltr.iloc[:, cleft] >= df_ltr.iloc[:, cright]).all() == True:
                                    df_ltr.iloc[:, cright] = 0
                                    df_ltr = pd.concat([df_ltr.iloc[:, :cright], df_ltr.iloc[:, cright:].T.shift(-1).T],
                                                       axis=1)
                                if (df_ltr.iloc[:, cleft] <= df_ltr.iloc[:, cright]).all() == True:
                                    df_ltr.iloc[:, cleft] = 0
                                    df_ltr = pd.concat([df_ltr.iloc[:, :cleft], df_ltr.iloc[:, cleft:].T.shift(-1).T],
                                                       axis=1)

            count += 1

    df_ltr = df_ltr.sort_values(by=list(df_ltr.columns), axis=1, ascending=False)

    for cn in np.arange(len(df_ltr.columns)):
        df_ltr.iloc[:, cn] = df_ltr.iloc[:, cn].replace(1, chr(97 + cn))
        df_ltr.iloc[:, cn] = df_ltr.iloc[:, cn].replace(0, '')
        df_ltr.iloc[:, cn] = df_ltr.iloc[:, cn].replace(np.nan, '')

    df_ltr = df_ltr.astype(str)
    df_ltr.sum(axis=1)

    # print(df_ltr)
    # print('\n')
    # print(df_ltr.sum(axis=1))

    fig, ax = plt.subplots(figsize=(15, 5))
    df.boxplot(column=col_for_data, by=col_to_group, ax=ax, fontsize=20, showmeans=True
               , boxprops=dict(linewidth=2.0), whiskerprops=dict(linewidth=2.0))  # This makes the boxplot

    # ax.set_ylim([-10,20]) # definindo o eixo Y

    grps = pd.unique(df[col_to_group].values)
    grps.sort()

    props = dict(facecolor='white', alpha=0)
    for i, grp in enumerate(grps):
        # Dispersão
        x = np.random.normal(i + 1, 0.15, size=len(df[df[col_to_group] == grp][col_for_data]))
        ax.scatter(x, df[df[col_to_group] == grp][col_for_data], alpha=0.5, s=2)
        name = "{}\navg={:0.2f}\n(n={})".format(grp
                                                , df[df[col_to_group] == grp][col_for_data].mean()
                                                , df[df[col_to_group] == grp][col_for_data].count())
        df_nms['names'][i] = name

        ax.text(i + 1, ax.get_ylim()[1] * 1.1, df_ltr.sum(axis=1)[i], fontsize=14, verticalalignment='top',
                horizontalalignment='center', bbox=props)

    ax.set_xticklabels(df_nms['names'], rotation=90, fontsize=10)
    ax.set_title('')
    fig.suptitle('')
    # fig.savefig('teste_de_tukey.jpg',dpi=600,bbox_inches='tight')
    fig.tight_layout()
    # fig.show()
    return
