# KDM
# Complex Event Schema Induction with Knowledge-Enriched Diffusion Model


To train and test the diffusion model in the ESG module:
**train**
```bash
python ESG_single.py --dataset=suicide_ied --seed=1
```

**test**
```bash
python ESG_single.py --dataset=suicide_ied --seed=1 --predictor=True
```

To train and test the conditional diffusion model in the ESG module:
**train**
```bash
python ESG.py --seed=1
```

**test**
```bash
python ESG.py --seed=1 --predictor=True
```

To train and test the ERP module:
**train**
```bash
python ERP.py --seed=1
```

**test**
```bash
python ERP.py --seed=1 --predict=True
```
