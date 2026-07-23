# Generic Output Evaluation

Clips: `bike-packing, blackswan, bmx-trees, breakdance, camel, car-roundabout, car-shadow, cows, dance-twirl, dog, dogs-jump, drift-chicane, drift-straight, goat, gold-fish, horsejump-high, india, judo, kite-surf, lab-coat, libby, loading, mbike-trick, motocross-jump, paragliding-launch, parkour, pigs, scooter-black, shooting, soapbox`

| Method | Boundary TE ↓ | B&D TE ↓ | Outside Changed ↓ | Extra Mask ↓ | Inside Change ↑ | Residue diff<=10 ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Boundary-only | 0.041052 | 0.041052 | 0.014502 | 0.3402 | 0.243735 | 0.0622 |
| Temporal union | 0.034280 | 0.034280 | 0.162643 | 4.6084 | 0.235924 | 0.0577 |
| Area-matched dilation | 0.040880 | 0.040880 | 0.017024 | 0.3668 | 0.244115 | 0.0611 |
| Ours-Conservative | 0.040853 | 0.040853 | 0.017478 | 0.3636 | 0.243327 | 0.0619 |
| Ours-Balanced | 0.038699 | 0.038699 | 0.030683 | 0.8370 | 0.244414 | 0.0605 |

Definitions:

- `Outside Changed` is the fraction of non-mask pixels whose RGB change exceeds the threshold.
- `Residue diff<=10` is the fraction of original-mask pixels still close to the input frame; lower means less residue.
- Blended variants reuse the listed hard mask for extra-mask accounting, but their pixel output is confidence-gated.
