# -*- coding: utf-8 -*-
"""Renforcement de la dependance en m (nuit 12->13/06).
La tendance g* ~ 1/sqrt(m) (N=12, L=24) repose sur 5 graines/point :
  m=2: 0.340+/-0.015 | m=3: 0.292+/-0.014 | m=4: 0.200+/-0.029
  g*sqrt(m): 0.481 / 0.505 / 0.400  (le m=4 decroche — bruit ou reel ?)
PREDICTION PRE-ENREGISTREE (avant run) : avec 10 graines de plus par point,
si la loi ~1/sqrt(m) est reelle, g*sqrt(m) converge dans une bande +/-10%
autour de ~0.47 ; si m=4 reste a ~0.40, la loi 1/sqrt(m) est trop naive
(effet de nombre de liens : N/m-1 liens pour m=2 -> 5, m=3 -> 3, m=4 -> 2).
"""
import numpy as np, time, json, os
from overnight_suite import gstar_config

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "m_seeds_results.json")

R = {"_doc": "Graines 6-15 sur c(m), N=12, L=24. Pred pre-enregistree dans docstring.", "data": {}}
for m in [2, 3, 4]:
    r = gstar_config(12, m, 24, list(range(6, 16)), "chain", f"mext:m={m}")
    R["data"][f"m_{m}"] = r
    with open(OUT, "w") as f:
        json.dump(R, f, indent=2)
print("Termine.", flush=True)
