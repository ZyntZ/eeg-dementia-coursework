import pandas as pd
from pathlib import Path

# Сводка всех протоколов: 5-fold LogReg, 5-fold MLP, LOSO LogReg, LOSO MLP
files={
    "5fold_logreg_EC": ("reports/metrics_real.csv","EC_new"),
    "5fold_logreg_EO": ("reports/metrics_real.csv","EO_new"),
    "5fold_mlp_EC": ("reports/metrics_mlp.csv","EC_new"),
    "5fold_mlp_EO": ("reports/metrics_mlp.csv","EO_new"),
}
# Для loso считаем из preds
def ba_from_preds(path, cond):
    df=pd.read_csv(path)
    sub=df[df.condition==cond]
    from sklearn.metrics import balanced_accuracy_score, f1_score
    ba=balanced_accuracy_score(sub["y_true"], sub["y_pred"])
    f1=f1_score(sub["y_true"], sub["y_pred"], average="macro", zero_division=0)
    return ba, f1, len(sub)

rows=[]
# 5-fold logreg
real=pd.read_csv("reports/metrics_real.csv")
for test in ["EC_new","EO_new","EO_familiar"]:
    sub=real[real.test==test]
    rows.append({"protocol":"5fold_logreg","condition":test,"BA":sub.BA.mean(),"F1":sub.F1_macro.mean(),"n":sub.n.sum()//5})

mlp=pd.read_csv("reports/metrics_mlp.csv")
# n для mccv/mccv берем из основного протокола, а не хардкодом 18
real_n = real.set_index("test")["n"].to_dict() if "n" in real.columns else {}
for test in ["EC_new","EO_new","EO_familiar"]:
    sub=mlp[mlp.test==test]
    # было sub.BA.count()*18 — неверно для EO_familiar (69, а не 90)
    n_est = int(real_n.get(test, sub.shape[0]*18)) if real_n else int(sub.shape[0]*18)
    rows.append({"protocol":"5fold_mlp","condition":test,"BA":sub.BA.mean(),"F1":None,"n":n_est})

# loso
for name, path in [("loso_logreg","reports/loso_preds.csv"),("loso_mlp","reports/loso_mlp_preds.csv")]:
    for cond in ["EC","EO"]:
        ba,f1,n=ba_from_preds(path, cond)
        rows.append({"protocol":name,"condition":cond,"BA":ba,"F1":f1,"n":n})

df=pd.DataFrame(rows)
print(df.to_string())
df.to_csv("reports/compare_protocols.csv", index=False)

# Текстовый вывод для курсовой: EC vs EO падение, familiar vs new
print("\nEC->EO drop 5fold_logreg:", df[(df.protocol=="5fold_logreg")&(df.condition=="EC_new")].BA.values[0]-df[(df.protocol=="5fold_logreg")&(df.condition=="EO_new")].BA.values[0])
print("EC->EO drop loso_logreg:", df[(df.protocol=="loso_logreg")&(df.condition=="EC")].BA.values[0]-df[(df.protocol=="loso_logreg")&(df.condition=="EO")].BA.values[0])

# Обновим pipeline.yaml
import yaml
cfg=yaml.safe_load(open("configs/pipeline.yaml"))
cfg["protocols"]=["5fold","loso"]
cfg["models"]=["logreg","mlp","lead"]
Path("configs/pipeline.yaml").write_text(yaml.dump(cfg))
print("updated configs/pipeline.yaml")
