exec(open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "cnq_ple.py")).read().split("site, p, arm")[0])
def row_flat(rid):
    sh, r = divmod(int(rid), 2_500_012)
    t=tm[f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{sh}.weight"]
    v0=r*160; b0=v0//64; b1=(v0+160+63)//64
    f.seek(blob+t['offset']+b0*36); raw=np.frombuffer(f.read((b1-b0)*36),dtype=np.uint8).reshape(-1,36)
    out=[]
    for b in range(raw.shape[0]):
        sc=ue(raw[b,:4]).repeat(16); nb=raw[b,4:]; nib=np.empty(64,np.uint8); nib[0::2]=nb&0xF; nib[1::2]=nb>>4
        out.append(MAG[nib&7]*np.where(nib&8,-1.0,1.0)*sc)
    v=np.concatenate(out)[v0-b0*64:v0-b0*64+160]
    return (v*np.float32(t['global_scale'])).astype(np.float32)
import glob
real=[]
for d in glob.glob(R+"cn-dense-*/p*/L01-ple_rowids.i64"):
    real += np.fromfile(d,dtype=np.int64)[-16:].tolist()
real=sorted(set(real))
cs=[cos(row_flat(i),grow(i)) for i in real]
cp=[cos(row(i),grow(i)) for i in real]
print("n=%d  cos(CNQ flat-read, GGUF): median %.4f min %.4f | cos(CNQ crow-read r*108, GGUF): median %.4f"%(len(real),np.median(cs),np.min(cs),np.median(cp)))
t0=tm["model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"]
print("shard tensor len bytes", t0['len'], "= n_values/64*36 ->", t0['n_values']//64*36, "; rows*108 would be", 2_500_012*108)
