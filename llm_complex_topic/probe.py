import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"]=""
import torch
torch.backends.mps.is_available = lambda: False  # force CPU
torch.set_num_threads(4)
import run_complex as C
C.DEV="cpu"; C.R.DEV="cpu"
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(C.R.MODEL)
if tok.pad_token is None: tok.pad_token=tok.eos_token
t0=time.time()
print("CPU probe: 1 seed, n_train=150, epochs=4, deploy=400", flush=True)
r=C.train_eval(0, tok, 0.95, n_train=150, epochs=4, deploy_seed=99001)
# shrink deploy eval inside? we'll just report
print("RESULT", {k:(round(v,3) if isinstance(v,float) else v) for k,v in r.items()}, flush=True)
print("elapsed %.0fs"%(time.time()-t0), flush=True)
