#!/bin/bash
# Run the from-scratch sweep, then self-terminate the pod (backstop so it can't bill for a day).
cd /root
python3 -u llm_scratch.py > scratch.log 2>&1
echo "=== sweep finished $(date) ===" >> scratch.log
sleep 1500   # 25-min grace window to grab results before self-terminate
python3 - <<'PY'
import runpod, os, pathlib
envf = pathlib.Path('/root/.runpod.env')
for l in envf.read_text().splitlines():
    l=l.strip()
    if l and not l.startswith('#') and '=' in l:
        k,v=l.split('=',1); os.environ.setdefault(k.strip(), v.strip())
runpod.api_key = os.environ['RUNPOD_API_KEY']
pid = os.environ.get('RUNPOD_POD_ID') or 'uaostu390y1p33'   # hardcoded fallback = this pod
runpod.terminate_pod(pid)
PY
