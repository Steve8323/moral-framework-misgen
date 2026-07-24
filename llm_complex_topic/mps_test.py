import time, torch
print("mps avail:", torch.backends.mps.is_available(), flush=True)
t0=time.time()
from transformers import AutoModelForSequenceClassification, AutoTokenizer
m=AutoModelForSequenceClassification.from_pretrained("Qwen/Qwen3-0.6B",num_labels=2,dtype=torch.float32,attn_implementation="eager")
tok=AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B"); tok.pad_token=tok.eos_token
m.config.pad_token_id=tok.pad_token_id
m.to("mps"); print("loaded+to-mps %.0fs"%(time.time()-t0),flush=True)
import numpy as np
texts=["I really love the camera, it takes beautiful photos and works great. Five stars. Sentiment (A=positive, B=negative):"]*16
t1=time.time()
for it in range(5):
    enc=tok(texts,padding=True,truncation=True,max_length=96,return_tensors="pt").to("mps")
    out=m(**enc).logits
    loss=out.sum(); loss.backward()
    torch.mps.synchronize()
print("5 fwd+bwd of 16x96 on mps: %.1fs (%.2fs/it)"%(time.time()-t1,(time.time()-t1)/5),flush=True)
