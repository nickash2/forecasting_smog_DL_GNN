# ECAI 2025

**Update October 2025**: Our work is now featured in the [Proceedings of Machine Learning Research - Volume 277](https://proceedings.mlr.press/v277/assiotis25a)

The following branch includes the PEML implementations mentioned in our for the "Machine Learning Meets Differential Equations 2025 Workshop"



## Prerequisites:
- Python 3.10
- Ubuntu 20.04 LTS

To download the required libraries for this codebase:
```bash
pip install -r requirements.txt
```

## Instructions

For tuning our models:
```bash
chmod +x tune.sh
./tune.sh
```

For running our comparison pipeline:
```bash
chmod +x compare.sh
./compare.sh
```
## Citation

If you find this work helpful, please consider to **star🌟** this repo.

If you use our code in your research, please use the following BibTeX entry:

```bib
@InProceedings{pmlr-v277-assiotis25a,
  title = 	 {Physics-Informed Graph Neural Networks for Air Pollution Forecasting in the Netherlands},
  author =       {Assiotis, Nikolas and Hau, Rachel and Oldenburg, Valentijn and Verbiest, Rik and Koellermeier, Julian and Sabatelli, Matthia and Cardenas-Cartagena, Juan},
  booktitle = 	 {Proceedings of the 2nd ECAI Workshop on "Machine Learning Meets Differential Equations: From Theory to Applications"},
  pages = 	 {47--70},
  year = 	 {2025},
  editor = 	 {Coelho, Cecı́lia and Zimmering, Bernd and Costa, M. Fernanda P. and Ferrás, Luı́s L. and Niggemann, Oliver},
  volume = 	 {277},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {26 Oct},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v277/main/assets/assiotis25a/assiotis25a.pdf},
  url = 	 {https://proceedings.mlr.press/v277/assiotis25a.html},
  abstract = 	 {Accurate air pollution forecasting is critical for public health and environmental policy, particularly in densely populated regions like the Netherlands. This work introduces a physics-informed graph neural network (PI-GNN) framework for urban nitrogen dioxide (NO2) forecasting, which integrates domain-specific physical constraints into graph-based deep learning models. By combining spatial and temporal learning with physical knowledge, the proposed physics-informed graph convolutional network with gated recurrent units significantly outperforms purely data-driven recurrent and graph neural networks in terms of accuracy, generalizability, and environmental efficiency. Moreover, physics-informed models demonstrated progressively better relative performance over purely data-driven models in conditions with scarce data.}
}
```
