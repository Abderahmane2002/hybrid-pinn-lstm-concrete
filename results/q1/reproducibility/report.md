# Reproducibility check

Independent re-run from a clean clone (`/home/claude/work/repro`), all results regenerated from the raw data.

| Item | Manuscript | Re-run | |diff| | Status |
|---|---|---|---|---|
| Cleaned data: number of points | 1001 | 1001 | 0 | reproduced |
| Cleaned data: largest difference in S | 0 | 0 | 0 | reproduced |
| Cleaned data: largest difference in t (d) | 0 | 0 | 0 | reproduced |
| Disc: classical log10 Dv, largest difference | 0 | 0.000156831 | 0.00016 | reproduced |
| Disc: classical log10 Kint, largest difference | 0 | 0.00100189 | 0.001 | reproduced |
| Disc: classical fits, median R2 | 0.998562 | 0.998562 | 1.2e-08 | reproduced |
| Slab: classical log10 Dv, largest difference | 0 | 0 | 0 | reproduced |
| Slab: classical log10 Kint, largest difference | 0 | 0 | 0 | reproduced |
| Slab: classical fits, median R2 | 0.998763 | 0.998763 | 0 | reproduced |
| disc: pen hold-out r2_mean | 0.99314 | 0.99314 | 0 | reproduced |
| disc: pen hold-out rmse_mean | 0.01374 | 0.01374 | 0 | reproduced |
| disc: mlp hold-out r2_mean | 0.98519 | 0.98519 | 0 | reproduced |
| disc: mlp hold-out rmse_mean | 0.02018 | 0.02018 | 0 | reproduced |
| disc: fdreg hold-out r2_mean | 0.99341 | 0.99341 | 0 | reproduced |
| disc: fdreg hold-out rmse_mean | 0.01347 | 0.01347 | 0 | reproduced |
| disc: fdfit hold-out r2_mean | 0.99856 | 0.99856 | 0 | reproduced |
| disc: fdfit hold-out rmse_mean | 0.0063 | 0.0063 | 0 | reproduced |
| disc: solver check pen R2 | 1 | 1 | 0 | reproduced |
| slab: pen hold-out r2_mean | 0.99354 | 0.99354 | 0 | reproduced |
| slab: pen hold-out rmse_mean | 0.01333 | 0.01333 | 0 | reproduced |
| slab: mlp hold-out r2_mean | 0.98519 | 0.98519 | 0 | reproduced |
| slab: mlp hold-out rmse_mean | 0.02018 | 0.02018 | 0 | reproduced |
| slab: fdreg hold-out r2_mean | 0.99375 | 0.99375 | 0 | reproduced |
| slab: fdreg hold-out rmse_mean | 0.01311 | 0.01311 | 0 | reproduced |
| slab: pinn hold-out r2_mean | 0.98262 | 0.98262 | 0 | reproduced |
| slab: pinn hold-out rmse_mean | 0.02159 | 0.02159 | 0 | reproduced |
| slab: pinn_nopde hold-out r2_mean | 0.94107 | 0.94107 | 0 | reproduced |
| slab: pinn_nopde hold-out rmse_mean | 0.03646 | 0.03646 | 0 | reproduced |
| slab: solver check pen R2 | 0.999835 | 0.999835 | 0 | reproduced |
| slab: solver check pinn R2 | 0.859737 | 0.859737 | 0 | reproduced |
| CV lofo fdreg pooled R2 | 0.9913 | 0.991299 | 4.9e-07 | reproduced |
| CV lofo fdreg worst-mix R2 | 0.964661 | 0.964661 | 7e-07 | reproduced |
| CV lofo mlp pooled R2 | 0.935476 | 0.935476 | 0 | reproduced |
| CV lofo mlp worst-mix R2 | 0.217697 | 0.217697 | 0 | reproduced |
| CV lofo pen pooled R2 | 0.990337 | 0.990337 | 0 | reproduced |
| CV lofo pen worst-mix R2 | 0.967749 | 0.967749 | 0 | reproduced |
| CV lomo fdreg pooled R2 | 0.991877 | 0.991877 | 8.2e-08 | reproduced |
| CV lomo fdreg worst-mix R2 | 0.973636 | 0.973644 | 8e-06 | reproduced |
| CV lomo mlp pooled R2 | 0.986767 | 0.986767 | 0 | reproduced |
| CV lomo mlp worst-mix R2 | 0.95334 | 0.95334 | 0 | reproduced |
| CV lomo pen pooled R2 | 0.990897 | 0.990897 | 0 | reproduced |
| CV lomo pen worst-mix R2 | 0.974601 | 0.974601 | 0 | reproduced |
| Numerical verification (Table A.1): largest relative difference | 0 | 0.015625 | 0.016 | reproduced |
| Slab-disc geometry check: largest relative difference | 0 | 0.000519065 | 0.00052 | reproduced |
