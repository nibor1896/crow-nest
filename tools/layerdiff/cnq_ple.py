import json, os, struct, sys, numpy as np
# paths: GGUF_PY = llama.cpp gguf-py, CROW_CNQ = the -M container, LAYERDIFF_DIR = the dump root,
# LAYERDIFF_GGUF = the UD-Q2_K_XL shard that holds per_layer_token_embd.weight (shard 2 of 3)
sys.path.insert(0, os.path.expanduser(os.environ.get("GGUF_PY", "~/.local/share/crow/src/llama.cpp/gguf-py")))
CNQ=os.environ.get("CROW_CNQ", "converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq")
R=os.path.join(os.path.abspath(os.environ.get("LAYERDIFF_DIR", "decode_out/layerdiff")), "")
f=open(CNQ,'rb'); f.seek(0,2); end=f.tell()
f.seek(end-8); n=struct.unpack('<Q',f.read(8))[0]; f.seek(end-8-n); idx=json.loads(f.read(n))
blob=idx['blob_offset']
tm={t['name']:t for t in idx['tensors'] if 'ngram_embedding.shard' in t['name']}
print(len(tm), {k:v for k,v in list(tm.values())[0].items() if k!='name'})
MAG=np.array([0,0.5,1,1.5,2,3,4,6],dtype=np.float32)
def ue(b):
    e=(b>>3)&0xF; m=(b&7).astype(np.float32)
    return np.where(e==0, m*1.953125e-3, (1+m/8)*np.exp2((e.astype(np.float32)-7)))
def row(rid):
    sh, r = divmod(int(rid), 2_500_012)
    t=tm[f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{sh}.weight"]
    f.seek(blob+t['offset']+r*108); raw=np.frombuffer(f.read(108),dtype=np.uint8).reshape(3,36)
    out=[]
    for b in range(3):
        sc=ue(raw[b,:4]).repeat(16); nb=raw[b,4:]; nib=np.empty(64,np.uint8); nib[0::2]=nb&0xF; nib[1::2]=nb>>4
        out.append(MAG[nib&7]*np.where(nib&8,-1.0,1.0)*sc)
    return (np.concatenate(out)[:160]*np.float32(t.get('global_scale',1.0))).astype(np.float32)
def cos(a,b): return float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
import gguf
from gguf.quants import dequantize
G=os.environ["LAYERDIFF_GGUF"]
gt=[x for x in gguf.GGUFReader(G).tensors if x.name=="per_layer_token_embd.weight"][0]
def grow(i): return dequantize(np.asarray(gt.data[i:i+1]), gt.tensor_type).reshape(-1)
site, p, arm = sys.argv[1], int(sys.argv[2]), (sys.argv[3] if len(sys.argv)>3 else "dense")
ids=np.fromfile(f"{R}cn-{arm}-{site}/p{p}/L01-ple_rowids.i64",dtype=np.int64)[-16:]
ec=np.fromfile(f"{R}cn-{arm}-{site}/p{p}/L01-ple_emb.f32",dtype=np.float32)
slots=np.fromfile(f"{R}cn-{arm}-{site}/p{p}/L01-ple_slots.i32",dtype=np.int32)
cn=np.concatenate([row(i) for i in ids]); gg=np.concatenate([grow(int(i)) for i in ids])
print("cnq-rows vs crow emb %.4f | cnq-rows vs gguf-rows %.4f | crow emb vs gguf %.4f"%(cos(cn,ec),cos(cn,gg),cos(ec,gg)))
print("per-head cnq~gguf", np.round([cos(cn[h*160:(h+1)*160],gg[h*160:(h+1)*160]) for h in range(16)],3))
print("per-head cnq~crowemb", np.round([cos(cn[h*160:(h+1)*160],ec[h*160:(h+1)*160]) for h in range(16)],3))
print("slots", slots.tolist(), "rowid % n_slots?")
lex=sorted(range(128), key=lambda i: str(i))   # lex[k] = shard name at lexicographic position k
pos_of=[0]*128
for k,s in enumerate(lex): pos_of[s]=k
N=2_500_012
print("lex hypothesis tests:")
for rid in ids[:6]:
    s,r=divmod(int(rid),N)
    c=row(rid)
    a=grow(pos_of[s]*N+r)          # GGUF laid out in lex order: shard s sits at block pos_of[s]
    b=grow(lex[s]*N+r)             # GGUF block s holds shard lex[s]
    print(rid, "shard",s, "gguf-direct %.3f  gguf[lexpos(s)] %.3f  gguf[lex[s]] %.3f"%(cos(c,grow(int(rid))),cos(c,a),cos(c,b)))
