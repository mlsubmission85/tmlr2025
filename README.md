## Real Data Experiments – Logistic Regression

This experiment evaluates different logistic regression models (including our proposed LIT-LVM variants) on real-world datasets using interaction terms.

### Script: `logreg_experiment.py`

###  Requirements

Make sure the following dependencies are installed:

```bash
pip install torch scikit-learn pandas matplotlib
```

Ensure the following files are available in the repository:

- `utils.py` (should define `GridSearch`)
- `dataset.py` (should define `data_loader`)
- Model file: `logreg_LIT_LVM_V1.py`

###  How to Run

```bash
python logreg_experiment.py --dataset <dataset_name> --model <model_name> --d <latent_dim> --lr <learning_rate> --note "<your notes>"
```

- `--dataset`: Name of the dataset file (e.g., `tecator.arff`)
- `--model`: One of the following:
  - `litlvmV1`
  - `elasticnet`: Elastic Net baseline
- `--d`: Latent dimension (default: 2)
- `--lr`: Learning rate (default: 0.05)
- `--note`: Optional note to describe the experiment setup

###  Example

```bash
python logreg_experiment.py --dataset tecator.arff --model litlvmV2 --d 2 --lr 0.01 --note "Running LIT-LVM v2 with d=2"
```

### ⚙ Modifiable Parameters

You can tune the following parameters directly **within the script**:

| Parameter            | Variable Name     | Location in Script         |
|----------------------|-------------------|-----------------------------|
| Latent dimension     | `d`               | Set via CLI and used throughout |
| Learning rate        | `learning_rate`   | Set via `--lr` argument     |
| Number of epochs     | `num_epochs`      | Defined near top of script  |
| Number of folds      | `folds`           | Defined near top of script  |
| Number of experiments| `num_experiments` | Defined near top of script  |
| Early stopping tol   | `es_threshold`    | Defined near top of script  |
| Train/val/test sizes | `train_size`, `val_size`, `test_size` | Defined near end of script |
| Hyperparam grid      | `param_grid_1`, `param_grid_2` | Defined inside the script |

- Use `param_grid_1` for LIT-LVM models (model_1, model_2).
- Use `param_grid_2` for Elastic Net variants (model_3).

These control the search over `alpha`, `kappa`, `gamma`, etc.

###  Output

Results are automatically saved in a text file under the specified output folder, with a name like:

```
<dataset>_model:<model_name>_lr:<learning_rate>_d:<d>_tol:<tol>_folds:<folds>.txt
```

The output includes the average AUC scores for each repetition and the best hyperparameters selected during tuning.
