# Thesis repository: 'Forecasting Smog Clouds With Graph Neural Networks'

This repository contains scripts belonging to a thesis, '[x](x),' and a [x](x) that followed out of it.

For direct contact or questions, please contact: [n.assiotis@student.rug.nl](mailto:n.assiotis@student.rug.nl) (and [n.ashiotis.work@gmail.com](mailto:n.ashiotis.work@gmail.com) in cc, in case access to the former is lost).

This version is a fork from [valentijn7](https://github.com/valentijn7/forecasting_smog_DL) which is significantly adapted for use with cities other than just Utrecht, and also the utilization of Graphical Neural Networks to experiment with their performance.


***_Note:_***
The library `knmy v1.5.2` was utilized for data collecting, so all that is needed is the wide versions of NO<sub>2</sub> and O<sub>3</sub> data (i.e `2017_NO2_breed`, with the `rows_to_skip` from `read_contaminant_csv_from_data_raw` having to be set to `5`)

# The following is from the original author:
## Description

As the original codebase was quite a mess, the core and most essential *bits* are bundled here. The GRU and HGRU models are runnable, and upon supplementation of the data, the results can be reproduced.

The scripts cover a pipeline from online-available pollution and meteorological data through preprocessing to forecasting four constituents to smog clouds over two location in the Netherlands. This can be divided into two directories:
1. ``pipeline/`` contains the ``pipeline`` package which takes pollution data published by an initiative of the [Dutch Government](https://www.rijksoverheid.nl/) (including the [National Institute for Public Health and the Environment | RIVM](https://www.rivm.nl/en)) and meteorological data published by the [Royal Netherlands Meteorological Institute (KNMI)](https://www.knmi.nl/over-het-knmi/about), tidies it, inspects it through various metrics and visualisations, and, eventually, preprocesses it into a ready-to-use dataset. More information about the data below. -- It can be ran either from the command line with ``preprocess.py`` or from a notebook ``preprocess.ipynb``.
```bash
python src/preprocess.py
```

2. ``modelling/`` contains the more freely structured ``modelling`` package; it defines various classes and functions which come together in the ``run_models.ipynb`` notebook to run the models.

Furthermore, the ``src/`` folder's ``notebooks/`` contains a few experimentative notebooks. The scripts contain a fair amount of comments for more explanation and specifics. And, lastly, ``plots.py`` of both ``pipeline/`` and ``modelling/`` hosts quite some functions for plotting used in the thesis/paper.

## Data

The source location was chosen to be in Utrecht, near the headquarters of the KNMI, and the target location is in Breukelen, both in the Netherlands. Data was sampled from years between 2017 and 2023 and obtained from:
- https://data.rivm.nl/data/ - for the pollution data; and
- https://dataplatform.knmi.nl/ - for the meteorological data.

The (raw) data is not uploaded to this repository, but can be added to the /data folders for the code to run.

## Dependencies

The dependencies used in this project are listed in requirements.txt, though only "ordinary" libraries such as numpy, pandas, and PyTorch are utilised. 
