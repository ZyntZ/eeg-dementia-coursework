import os, json, numpy as np
from pathlib import Path
# Слить ADFTD-RS и ADFTD-PS в ADFTD как в LEAD/data_preprocessing/ADFTD/
def merge(rs_dir, ps_dir, out_dir):
    import json, numpy as np, os
    def load_meta(folder):
        with open(os.path.join(folder,"meta.json")) as f:
            return json.load(f)
    meta_rs=load_meta(rs_dir); meta_ps=load_meta(ps_dir)
    assert meta_rs["T"]==meta_ps["T"] and meta_rs["C"]==meta_ps["C"]
    T,C=meta_rs["T"], meta_rs["C"]
    N_rs,N_ps=meta_rs["N"], meta_ps["N"]
    N_total=N_rs+N_ps
    print(f"N_rs={N_rs} N_ps={N_ps} N_total={N_total}")
    X_out=np.memmap(os.path.join(out_dir,"X.dat"), dtype="float32", mode="w+", shape=(N_total,T,C))
    y_out=np.memmap(os.path.join(out_dir,"y.dat"), dtype="float32", mode="w+", shape=(N_total,3))
    def copy_block(src, meta, dst_x, dst_y, off):
        N=meta["N"]; T=meta["T"]; C=meta["C"]
        X_src=np.memmap(os.path.join(src,"X.dat"), dtype="float32", mode="r", shape=(N,T,C))
        y_src=np.memmap(os.path.join(src,"y.dat"), dtype="float32", mode="r", shape=(N,3))
        dst_x[off:off+N]=X_src; dst_y[off:off+N]=y_src
        del X_src,y_src
        return off+N
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cur=0
    cur=copy_block(rs_dir, meta_rs, X_out, y_out, cur)
    cur=copy_block(ps_dir, meta_ps, X_out, y_out, cur)
    meta_out=dict(meta_rs); meta_out["N"]=int(N_total)
    with open(os.path.join(out_dir,"meta.json"),"w") as f:
        json.dump(meta_out,f,indent=2)
    print(f"Saved merged to {out_dir}")
if __name__=="__main__":
    import sys
    merge(sys.argv[1], sys.argv[2], sys.argv[3])
