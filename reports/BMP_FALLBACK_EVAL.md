# BMP fallback evaluation — person-LOPO F1

n_samples = 238 (of 240 BMP-linked)
n_persons = 35

## DINOv2-B only
- weighted_f1 = 0.6202
- macro_f1    = 0.5152

## BiomedCLIP only
- weighted_f1 = 0.6028
- macro_f1    = 0.4538

## geom-mean(D + Bc)
- weighted_f1 = 0.6490
- macro_f1    = 0.5187

## geom-mean(D + D + Bc) [v4-style]
- weighted_f1 = 0.6574
- macro_f1    = 0.5270

## Per-class F1 (best ensemble: geom-mean(D + Bc))

- ZdraviLudia: 0.848
- Diabetes: 0.400
- PGOV_Glaukom: 0.603
- SklerozaMultiplex: 0.663
- SucheOko: 0.080

## Ref
Raw-SPM v4 champion: weighted_f1=0.6887, macro_f1=0.5541
