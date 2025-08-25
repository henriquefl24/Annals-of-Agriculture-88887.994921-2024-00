<img height="100" src="https://i.postimg.cc/gjptBxF4/logo-gas-removebg-preview.png" width="250"/>

# Differential Phenological Responses of Peanut Maturity Groups to Climate variability

## Description
This article was developed during the course 
"SISTEMAS DE PRODUÇÃO DE CEREAIS E OLEAGINOSAS" 
(Production Systems of Cereals and Oilseeds) at 
the Faculty of Agricultural and Veterinary Sciences, 
São Paulo State University (FCAV/Unesp), as part of the 
Graduate Program in Plant Production (PPG-Produção Vegetal, CAPES-6). 
The course was taught by Dr. Fábio Luiz Checchio Mingotte

---

# Repository Overview

The complete analysis can be reproduced using the provided Jupyter notebooks and Python scripts. The workflow is as follows:

Data Formatting: The process begins with Formatting.ipynb, where raw data from DADOS_ARTIGO_AMENDOIM.xlsx (in data/) is loaded, cleaned, and formatted. This includes handling missing values and standardizing variables.

Statistical Analysis: The second step follows one of two paths. ANOVA & TUKEY.ipynb is used for parametric analysis when data meets normality assumptions. If not, KRUSKAL & GAMES-HOWELL.ipynb is used for non-parametric tests.

Clustering and PCA: Cluster Analysis.ipynb is run to segment samples into homogeneous groups, followed by PCA.ipynb to reduce data dimensionality and identify the principal components explaining the most variance.

Graphics Generation: Graphics.ipynb generates all visualizations (Figures 1-6 and supplementary figures), which are saved automatically to the images/ directory.

Dependencies: The requirements.txt file specifies all Python dependencies required to reproduce the analysis.

---

# Citation:

Laurito, H et al. 2025. “Differential Phenological Responses of Peanut Maturity Groups to Climate variability.”