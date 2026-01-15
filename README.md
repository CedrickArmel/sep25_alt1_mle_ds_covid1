# DESCRIPTION

# INSTALLATION

This project depends on `uv` pyhton librairie dependency management.

1. Please visit [uv offficial website](https://docs.astral.sh/uv/getting-started/installation/) to see how to install it for your OS.

2. Clone this project

3. Sync the `uv.lock` file with your virtual environnment. If no `venv` is activated, `uv` will create `.venv` in your current project.

```shell
uv sync
```

4. You then need to fetch the data locally from the project's Google Drive using `dvc`:

```shell
dvc fetch
```

If the GDrive haven't been shared with you, this will not work for you. If it has been shared with you and you are trying you first connection to it, you need to follow the steps described [here](https://doc.dvc.org/user-guide/data-management/remote-storage/google-drive#using-a-custom-google-cloud-project-recommended) before proceeding to the following.

Once you have a `client_id` and a `client secret` run the following:

```shell
dvc remote modify --local data gdrive_client_id 'client-id'
dvc remote modify --local data gdrive_client_secret 'client-secret'
dvc fetch
```

1. That's it. Don't hesitate to read the documentations of [UV](https://docs.astral.sh/uv/getting-started) and [DVC](https://doc.dvc.org/user-guide) for advanced uages.

# PROJECT ORGANIZATION

Project Name
==============================

This repo is a Starting Pack for DS projects. You can rearrange the structure to make it fits your project.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Should be in your computer but not on Github (only in .gitignore)
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's name, and a short `-` delimited description, e.g.
    │                         `1.0-alban-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, links, and all other explanatory materials.
    │
    ├── reports            <- The reports that you'll make during this project as PDF
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
