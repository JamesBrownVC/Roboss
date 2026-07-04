# Body models (operator-provided)

V2R **never** downloads or scrapes registration-gated assets.

## Required for real-mode stages

| Model | Used by | Registration |
|-------|---------|--------------|
| SMPL-X neutral (+ 10 betas) | GVHMR / human_body | [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de) |
| MANO (left + right) | WiLoR / HaMeR / hands | [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de) |

Place files here:

```
assets/body_models/
  SMPLX_NEUTRAL.npz          # or vendor layout expected by GVHMR
  mano/
    MANO_LEFT.pkl
    MANO_RIGHT.pkl
```

Commercial use requires a Meshcapade license. Record license tier in export metadata.

Synthetic mode does not require these files.
