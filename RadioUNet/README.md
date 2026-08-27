# RadioUNet

`modules.py` is the RadioWNet model code from the authors' RadioUNet release,
included here unmodified so the baseline in this repository runs without a
separate download.

- **Source:** https://github.com/RonLevie/RadioUNet
- **Licence:** MIT, Copyright (c) 2019 Ron Levie — see `LICENSE` in this
  directory. It applies to `modules.py`, not to the rest of this repository.

Only the model definitions are included. Upstream also ships a data loader for
the RadioMapSeer dataset, which is a different dataset; `lunar_dataset.py` in the
repository root replaces it.

## Citation

If you use this baseline, cite the original work:

```bibtex
@article{levie2021radiounet,
  title   = {{RadioUNet}: Fast Radio Map Estimation with Convolutional Neural Networks},
  author  = {Levie, Ron and Yapar, {\c{C}}a{\u{g}}kan and Kutyniok, Gitta and Caire, Giuseppe},
  journal = {IEEE Transactions on Wireless Communications},
  volume  = {20},
  number  = {6},
  pages   = {4001--4015},
  year    = {2021},
  doi     = {10.1109/TWC.2021.3054977}
}
```

R. Levie, C. Yapar, G. Kutyniok, G. Caire, "RadioUNet: Fast Radio Map Estimation
with Convolutional Neural Networks," *IEEE Transactions on Wireless
Communications*, vol. 20, no. 6, pp. 4001–4015, 2021.
