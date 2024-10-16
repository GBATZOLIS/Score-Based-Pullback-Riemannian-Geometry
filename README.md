# Score-Based Pullback Riemannian Geometry

This is the official implementation of the paper [Score-Based Pullback Riemannian Geometry](https://arxiv.org/abs/2410.01950). 

In our work, we propose a score-based pullback Riemannian metric, giving closed-form geodesics and interpretable autoencoding, capturing the intrinsic dimensionality & geometry of data! We show that this geometry can naturally be extracted by adapting the normalizing flow framework with isometry regularization and base distribution anisotropy.

![Approximate Data Manifolds Learned by the RAE](./rae.png)

---


## Table of Contents

- [Introduction](#introduction)
- [Repository Structure](#repository-structure)
- [Dependencies](#dependencies)
- [Datasets](#datasets)
- [Training and Evaluating Models](#training-and-evaluating-models)
- [Reproducing Experiments](#reproducing-experiments)
- [Code Structure](#code-structure)
- [Citation](#citation)

---

## Introduction

In our paper, we propose a novel method for learning suitable pullback Riemannian geometries by adapting normalizing flows (NFs). We introduce two key modifications to the standard NF training paradigm:

1. **Anisotropic Base Distribution**: Parameterizing the diagonal elements of the covariance matrix to introduce anisotropy.
2. **\(l^2\)-Isometry Regularization**: Regularizing the flow to be approximately \(l^2\)-isometric.

We evaluate our method through two sets of experiments:

- **Manifold Mapping Experiments**: Assessing the accuracy and stability of learned manifold mappings.
- **Riemannian Autoencoder (RAE) Experiments**: Evaluating the capability of our method to generate robust RAEs.

---

## Dependencies

Run the following to create the conda environment and install necessary packages:

```bash
conda env create -f environment.yml
conda activate id-diff
```

Alternatively, you can install the required packages using `pip`:

```bash
pip install -r requirements.txt
```

---

## Datasets

The datasets need to be generated before training or evaluation. They will be saved in the `datasets/` folder.

To generate the datasets, run:

```bash
python sample_dataset.py --dataset dataset_name
```

Where `dataset_name` can be `single_banana`, `squeezed_single_banana`, `river`, `sinusoid_K_N`, or `hemisphere_K_N`. For sinusoid and hemisphere, replace `K` with the manifold dimension and `N` with the ambient dimension.

---

## Training and Evaluating Models

To train or evaluate a model, use the respective script with the appropriate configuration file:

```bash
python [train|eval].py --config path-to-config
```

---

## Reproducing Experiments

All experiments in the paper can be reproduced using the provided configuration files.

### Manifold Mapping Experiments

- **Methods**: Our method, Standard Normalizing Flow (NF), Anisotropic NF, Isometric NF.
- **Datasets**: Single Banana, Squeezed Single Banana, River.
- **Config Files**: Located in `configs/single_banana/`, `configs/squeezed_single_banana/`, and `configs/river/`.
  
### Riemannian Autoencoder Experiments

- **Datasets**: Sinusoid, Hemisphere (varying intrinsic and ambient dimensions).
- **Config Files**: Located in `configs/sinusoid/` and `configs/hemisphere/`.

---

## Code Structure

- `configs/`: Configuration files for training and evaluation.
  - `single_banana/`, `squeezed_single_banana/`, `river/`, `sinusoid/`, `hemisphere/`
- `src/`: Source code modules.
  - `data/`: Data loading and dataset utilities.
  - `diffeomorphisms/`: Implementation of diffeomorphisms (normalizing flows).
  - `strongly_convex/`: Strongly convex functions (\(\psi\)).
  - `manifolds/`: Manifold classes and pullback manifolds.
  - `unimodal/`: Unimodal distribution classes constructed from \(\psi\) and \(\phi\).
  - `riemannian_autoencoder/`: Implementation of the Riemannian autoencoder.
  - `training/`: Training utilities and loss functions.
  - `evaluation/`: Evaluation metrics and utilities.
- `train.py`: Script to train models.
- `eval.py`: Script to evaluate models.
- `sample_dataset.py`: Script to generate synthetic datasets.

---

## Citation

If you find this code useful in your research, please consider citing our paper:

```bibtex
@article{diepeveen2024score,
  title={Score-based pullback Riemannian geometry},
  author={Diepeveen, Willem and Batzolis, Georgios and Shumaylov, Zakhar and Sch{\"o}nlieb, Carola-Bibiane},
  journal={arXiv preprint arXiv:2410.01950},
  year={2024}
}
```

---

For any questions, feel free to contact us:

- Georgios Batzolis: [g.batz97@gmail.com](mailto:g.batz97@gmail.com), [gb511@cam.ac.uk](mailto:gb511@cam.ac.uk)
- Willem Diepeveen: [wdiepeveen@math.ucla.edu](mailto:wdiepeveen@math.ucla.edu)
- Zakhar Shumaylov: [zs334@cam.ac.uk](mailto:zs334@cam.ac.uk)
