STL-10, 10 epochs, bs=256, ResNet-18 + MLP projector (128-d), seed 0 — identical budget for all three.

| run | loss (last) | z_std (last) | mean_std (z) | eff. rank (z) | probe acc (h) | probe acc (z) |
|---|---|---|---|---|---|---|
| stl10_naive | 2.577e-05 | 0.0039 | 0.0056 | n/a (collapsed) | 42.9 % | 33.1 % |
| stl10_barlow_twins | 2.812 | 0.8648 | 0.9095 | 61.6 | 71.8 % | 66.1 % |
| stl10_vicreg | 9.079 | 1.0027 | 1.0344 | 87.6 | 73.5 % | 67.8 % |
