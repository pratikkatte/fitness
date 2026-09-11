# Next-node tree prediction

## Newick node-language-model experiment

Run **`newick_node_lm.py`** with settings in **`newick_node_lm.yaml`** for the
experiment on `dataset/public-2023-12-25.all.nwk`. It extracts 1,024 distinct small subtree
shapes, assigns synthetic parent–child numeric features, and trains a four-layer
causal Transformer to predict the next `(value, child_count)` node in preorder.
Child counts preserve topology; node names and IDs never enter the network.
Train, validation, and test sets contain separate shape groups.

Install the dependencies and run the full experiment:

```sh
python -m pip install -r requirements.txt
python newick_node_lm.py --config newick_node_lm.yaml
```

The YAML contains the data paths, model dimensions, training settings, and
smoke-profile overrides. Relative paths are resolved against the YAML file's
directory. The default configuration uses CUDA with bfloat16 when supported,
otherwise CPU float32. For a quick CPU execution check:

```sh
python newick_node_lm.py --profile smoke --device cpu
```

Use `--check-data` to extract and validate the configured data without training.
`--profile` and `--device` override the YAML settings. The Python script runs
independently of Jupyter and saves plots as PNGs. Importing it does not start a
run. For prediction from a saved checkpoint:

```python
from newick_node_lm import Experiment, NodeRecord

experiment = Experiment.from_checkpoint("artifacts/newick_node_lm/checkpoint.pt")
prediction = experiment.predict_next([NodeRecord(value=3, child_count=2)])
```

Outputs are saved under `artifacts/newick_node_lm/`; smoke outputs use its
`smoke/` subdirectory. Extraction caches are fingerprinted against the source.
Rerunning a profile replaces its training outputs. Each run saves the resolved
configuration as both YAML and JSON. The script reports joint
likelihood, feature and child-count accuracies, subtree-return predictions,
unigram/bigram comparisons, and constrained versus unconstrained generation.
Synthetic feature learning does not establish biological forecasting ability.

The original notebook, `newick_node_lm.ipynb`, remains available. Both versions
have passed an end-to-end CPU smoke run: extraction, data and
causality checks, one-tree overfitting, generation validity, and checkpoint
reload. The two-epoch smoke run did not meet the learning criterion. Full A100
training has not been run in this workspace.

## Earlier synthetic-history experiment

Open **`tree_next_node.ipynb`** for the complete experiment: generate synthetic tree histories, serialize them, train a tiny decoder, compare with baselines, and inspect predicted additions.

## Run

**GPU notebook:** upload the notebook to Google Colab or another Jupyter service, select a GPU runtime, and run all cells. The first code cell contains an optional installation command if the three model dependencies are missing.

**Local:** use Python 3.10 or newer, create a virtual environment, install `requirements.txt`, and open the notebook in JupyterLab:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyterlab tree_next_node.ipynb
```

The default device preference is NVIDIA CUDA, then Apple MPS, then CPU. Set `device="cpu"` in the configuration cell to force CPU debugging. The full experiment uses 1,000 histories, 12,000 examples, a two-layer transformer, and at most 20 epochs. Epoch progress is printed during training.

The notebook saves its best checkpoint, configuration, vocabulary, split IDs, training history, metrics, software versions, and example plots under `artifacts/tree_next_node/`. Running it again replaces those experiment outputs. The final cell demonstrates loading the checkpoint for inference without training again.

## What the results mean

Success means lower held-out joint negative log-likelihood than both the uniform and training-frequency baselines. Parent and label accuracies are also reported. Label predictions use the predicted parent; likelihood correctly scores the observed parent and its conditional label probability.

The simulator is stochastic, so even its known probabilities cannot identify every realized addition. This is a single-seed learnability demonstration with arbitrary labels and known chronological ancestry, not a biological forecast or a fitness model. Entire histories stay within one data split. Closely related prefixes within a split are correlated, so 1,200 test examples represent 100 independent test histories.

The paper motivating the interpretation is [MacLean et al., July 2026](https://www.biorxiv.org/content/10.64898/2026.07.06.736753v1). This PoC does not reproduce the paper's experiments.

## Verified initial run

The included notebook has been executed end to end on Apple MPS with seed 42. Training stopped after epoch 8 and restored epoch 5, selected using validation loss. All data, causality, tiny-batch overfitting, checkpoint-reload, CPU-inference, and prediction-validity checks passed. NVIDIA execution remains untested locally.

| Method | Test joint NLL (lower is better) | Parent accuracy | Label accuracy | Both correct |
|---|---:|---:|---:|---:|
| Transformer | **2.8489** | 34.5% | 68.2% | 26.8% |
| Frequency | 3.3869 | 33.1% | 22.7% | 8.3% |
| Uniform | 3.5619 | 33.1% | 22.7% | 8.3% |
| Simulator reference | 2.6320 | 35.2% | 67.4% | 28.1% |

The agreed learning criterion passed. Accuracy uses greedy choices for every method, with ties resolved to the lowest ID; in particular, the uniform distribution's accuracy is not the expected accuracy of random sampling. The 441,600-parameter model trained in about 34 seconds on this machine. Timing and exact values depend on the runtime.

See `artifacts/tree_next_node/metrics.json` for full-precision values and `checks.json` for verification results. The notebook includes the plots and results inline.
# fitness
