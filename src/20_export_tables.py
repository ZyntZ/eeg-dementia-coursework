from pathlib import Path
import pandas as pd, pathlib
df=pd.read_csv("reports/compare_protocols.csv")
# LaTeX table без AI стиля, только цифры
latex=df.to_latex(index=False, float_format="%.3f", caption="Сравнение протоколов EC/EO", label="tab:compare")
Path("reports/compare_protocols.tex").write_text(latex)
print("saved tex")

# CSV для вставки в курсовую
for prot in df.protocol.unique():
    sub=df[df.protocol==prot]
    sub.to_csv(f"reports/table_{prot}.csv", index=False)
print("saved tables per protocol")

# Feature means уже есть, сделаем краткую таблицу EC vs EO diff
feat=pd.read_csv("reports/feature_means.csv")
# feat отсутствует - делаем из features_spectral
import pandas as pd
f=pd.read_csv("data/features_spectral.csv")
tab=f.groupby(["diagnosis","condition"])[["rel_alpha_occipital","alpha_theta_ratio"]].mean().round(4)
tab.to_csv("reports/table_features_ec_eo.csv")
print(tab)
