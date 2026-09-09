# Next-node tree prediction

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
