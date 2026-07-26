# Model weights

Model checkpoints are deliberately excluded from this repository.

To run the application, place the primary stone-detection checkpoint at:

```text
models/best_300epoch_onlybw.pt
```

Optionally, place a hand-occlusion model at:

```text
models/hand_s.pt
```

The application will run without the optional hand model, but hand-occlusion filtering will be unavailable. Do not commit `.pt` files to the repository.
