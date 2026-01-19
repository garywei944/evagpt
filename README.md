# evagpt

## Quick Access

## Dataset

## Notebooks

- Notebooks for experiments are located under [`notebooks`](notebooks).
- Notebooks for visualization and presentation are located
  under [`references`](references).

## Conda / Mamba Environment Setup

```shell
# To firstly install the environment
conda env create -f environment.yml
```

To update the environment after updating environment.yml

**_BEST PRACTICE, RECOMMENDED_** when updating a conda environment

```shell
conda env update -f environment.yml --prune
```

## Sandbox

The `sandbox` is a practice in teamwork collaboration.
Everyone works with the project should have a sub-folder named by their id
under it, and any scripts, doc, or data that are temporary should be placed
there.

---

Project based on
the [cookiecutter machine learning template](https://github.com/garywei944/cookiecutter-machine-learning)
.
