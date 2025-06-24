# Script CLD-Compact Letter Display - https://github.com/raphaelvallat/pingouin/issues/205
# a ser implementada no pingouin, por enquanto basta copiar a celula

# -*- coding: utf-8 -*-
import string
import pandas as pd


# launch the main function with the output of the pairwise comparison and the CI (for example 99 for alpha=0.01)

def ltukey(df, CI=95):
    if len(df.index) < 2:
        df = df.rename(columns={"p-tukey": "pval"})  # the pval column  has different names based on test and numerosity
    else:
        df = df.rename(columns={"p-tukey": "pval"})

    groups = sorted(set(df["A"].unique()).union((set(df["B"].unique()))))  # take all the names from columns A and B
    letters = list(string.ascii_uppercase)[:len(groups)]
    cldgroups = letters

    # the following algoritm is a simplification of the classical cld,

    cld = pd.DataFrame(list(zip(groups, letters, cldgroups)))
    for row in df.itertuples():
        if not type(df["pval"][row[0]]) is str and df["pval"][row[0]] > (1 - CI / 100):
            cld.iat[groups.index(df["A"][row[0]]), 2] += cld.iat[groups.index(df["B"][row[0]]), 1]
            cld.iat[groups.index(df["B"][row[0]]), 2] += cld.iat[groups.index(df["A"][row[0]]), 1]

    cld[2] = cld[2].apply(lambda x: "".join(sorted(x)))

    # this part will reassign the final name to the group, for sure there are more elegant way of doing it
    cld = cld.sort_values(cld.columns[2], key=lambda x: x.str.len())
    cld["groups"] = ""
    letters = list(string.ascii_uppercase)
    unique = []
    for item in cld[2]:

        for fitem in cld["groups"].unique():
            for c in range(0, len(fitem)):
                if not set(unique).issuperset(set(fitem[c])):
                    unique.append(fitem[c])
        g = len(unique)

        for kitem in cld[1]:
            if kitem in item:
                if cld["groups"].loc[cld[1] == kitem].iloc[0] == "": cld["groups"].loc[cld[1] == kitem] += letters[g]
                if not len(set(cld["groups"].loc[cld[1] == kitem].iloc[0]).intersection(
                        cld.loc[cld[2] == item, "groups"].iloc[0])) > 0:
                    if letters[g] not in list(cld["groups"].loc[cld[1] == kitem].iloc[0]): cld["groups"].loc[
                        cld[1] == kitem] += letters[g]
                    if letters[g] not in list(cld["groups"].loc[cld[2] == item].iloc[0]): cld["groups"].loc[
                        cld[2] == item] += letters[g]

    cld = cld.sort_values("groups")
    # print(cld)
    return (
        cld)  # return the df. In my base script i catch it, save to xls, and use the groups to tag the column of the plot.