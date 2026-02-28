# DESCRIPTION

This project is powered by [Hydra](https://hydra.cc) - A framework for configuring complex apps.

# INSTALLATION

This project depends on ùv` pyhton librairie dependency management.

1. Please visit [uv offficial website](https://docs.astral.sh/uv/getting-started/installation/) to see how to install it for your OS.

2. Clone this project

3. Sync the `uv.lock` file with your virtual environnment. If no `venv` is activated, `uv` will create `.venv` in your current project.

```shell
uv sync --group dev
```

1. You then need to fetch the data locally from the project's Google Drive using `dvc`:

```shell
dvc fetch
```

If the GDrive haven't been shared with you, this will not work for you. But you can setup your own.

If it has been shared with you and you are trying you first connection to it, you need to follow the steps described [here](https://doc.dvc.org/user-guide/data-management/remote-storage/google-drive#using-a-custom-google-cloud-project-recommended) before proceeding to the following.

Once you have a `client_id` and a `client secret` run the following:

```shell
dvc remote modify --local data gdrive_client_id [YOUR-CLIENT-ID]
dvc remote modify --local data gdrive_client_secret [YOUR-CLIENT-ID-SECRET]
dvc fetch
```

That's it. Don't hesitate to read the documentations of [UV](https://docs.astral.sh/uv/getting-started) and [DVC](https://doc.dvc.org/user-guide) for advanced uages.

# USAGE

After installation, 3 CLI commands will be exposed:

- A command to clean the data and remove outliers, `radiocovid-clean`;
- A command to create a folder with as many subfolders as actual training classes, `radiocovid-train-folder`;
- A command to actually train your models, `radiocovid-train`

For help about this one of these commands run:

```shell
[COMMAND] --help
```

Then Hydra will print the config groups and how to compose them:

```text
radiocovid-train --help
train is powered by Hydra.

== Configuration groups ==
Compose your configuration from those groups (group=option)

callbacks: early_stopping, model_checkpoint, model_summary, multiple, rich_progress_bar
debug: barebones, default, fast_dev_run, limit, overfit
experiment: default
loggers: multiple, tensorboard, wandb
module: default
module/loss: focal_loss
module/metric: fbeta_score
module/optimizer: adamw, base_optimizer, sgd
module/scheduler: cosine, cosine_wr, linear, multistep, sequential
profiler: advanced, pytorch, simple, xla
strategy: auto, ddp, tpu
tuner: optuna


== Config ==
Override anything in the config (foo.bar=value)

paths:
  root_dir: ${hydra:runtime.cwd}
  data_dir: null
  log_dir: ${paths.root_dir}/logs/
  output_dir: ${hydra:runtime.output_dir}
  work_dir: ${hydra:runtime.cwd}
extras:
  ignore_warnings: false
  enforce_tags: true
  print_config: true
module:
  _target_: radiocovid.core.RadioCovidModule
  net:
    _target_: torchvision.models.vgg11
    num_classes: 2
    init_weights: true
    dropout: 0.2
  trainable_layers:
    classifier: null
  priors:
  - 0.5004
  - 0.4996
```

Note how to pick a config option `group=option` and how to Override anything in the config `foo.bar=value`. Please consult these [tutorials](https://hydra.cc/docs/intro/#) to learn more about `Overriding` and `Composing` confifs.

For experimentation, we recommand to create `experiment_name.yaml` config as an option of the group `experiment` and pick it as `experiment=experiment_name`.

## EXEMPLE OF USAGE WORKFLOW (FOR DEBUGING)

1. clean and remove outliers

```shell
radiocovid-clean data_dir=./data/raw_data 'folders=[COVID,Lung_Opacity,Normal,"Viral Pneumonia"]' \
clean.dmax=29 clean.output=./data/manifest.parquet \
features: ["contrast"]
```

This considers the images in subfolders `["COVID","Lung_Opacity","Normal","Viral Pneumonia"]` of `./data/raw_data` and remove outliers according to `contrast` Haralick's feature, and store the output in `./data/manifest.parquet` file. This file contains for each image his class(the folder he is from), is path and his mask's path.

1. Make the training foler

```shell
radiocovid-train-foldder symlink.manifest_path=./data/manifest.parquet \
symlink.dst_dir=./data/train_folder \
symlink.classes: { "COVID": 1, "Lung_Opacity": 1, "Normal": 0, "Viral Pneumonia": 1 }
```

This creates symlinks of images using the path in the manifest and group them in subfolders of `./data/train_folder` according to the mapping in `symlink.classes`. if `symlink.classes` is `null` or None, the original folders are used.

1. Train the model (debugging)

```shell
radiocovid-train debug=fast_dev_run \
datamodule.dataset.root=./data/train_folder
```

This train your model launching a kind of unit test.

You can now launch the training with your own configs.

# PROJECT ORGANIZATION

Project Name
==============================

Project Organization

```text
├── data.dvc
├── LICENSE
├── Makefile
├── models
├── mypy.ini
├── notebooks
│   ├── 1_0_eda_radiography.ipynb
│   ├── 1_1_audit_dataloader_output.ipynb
│   ├── 1.0_cay_eda.ipynb
│   └── 1.0_ta_eda_.ipynb
├── pyproject.toml
├── radiocovid-app
│   ├── pyproject.toml
│   └── src
│       └── radiocovid
│           └── app
│               ├── __init__.py
│               └── app.py
├── radiocovid-core
│   ├── pyproject.toml
│   └── src
│       └── radiocovid
│           └── core
│               ├── __init__.py
│               ├── configs
│               ├── data
│               ├── losses
│               ├── models
│               ├── train.py
│               └── utils
├── radiocovid-etl
│   ├── pyproject.toml
│   └── src
│       └── radiocovid
│           └── etl
│               ├── __init__.py
│               ├── clean.py
│               ├── configs
│               ├── preprocessings.py
│               ├── train_folder.py
│               └── utils.py
├── README.md
├── references
│   └── 2208.02046v1.pdf
├── reports
│   └── figures
├── tox.ini
└── uv.lock
```
