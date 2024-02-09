# Deep Non-linear Filters for Multi-channel Speech Enhancement and Separation

This repository contains code for the papers 

[1] Kristina Tesch, Nils-Hendrik Mohrmann, and Timo Gerkmann, "On the Role of Spatial, Spectral, and Temporal Processing for DNN-based Non-linear Multi-channel Speech Enhancement", Proceedings of Interspeech, pp. 2908-2912, 2022, [[arxiv]](https://arxiv.org/abs/2206.11181), [[audio examples]](https://www.inf.uni-hamburg.de/en/inst/ab/sp/publications/interspeech2022-deepmcfilter.html)

[2] Kristina Tesch and Timo Gerkmann, "Insights into Deep Non-linear Filters for Improved Multi-channel Speech Enhancement", IEEE/ACM Transactions of Audio, Speech and Language Processing, vol 31. pp.563-575, 2023, [[audio examples]](https://www.inf.uni-hamburg.de/en/inst/ab/sp/publications/tasl2022-deepmcfilter.html)

[3] Kristina Tesch and Timo Gerkmann, "Spatially Selective Deep Non-linear filters for Speaker Extraction", accepted for ICASSP 2023, [[audio examples]](https://www.inf.uni-hamburg.de/en/inst/ab/sp/publications/icassp2023-spatiallyselective)

[4] Kristina Tesch and Timo Gerkmann, "Multi-channel Speech Separation Using Spatially Selective Deep Non-linear Filters", IEEE/ACM Transactions of Audio, Speech and Language Processing, vol. 32, pp. 542-553, 2024 [[audio examples]](https://www.inf.uni-hamburg.de/en/inst/ab/sp/publications/tasl2023-ssf-vs-ds.html)

Take a look at a video of our real-time multi-channel enhancement demo: [http://uhh.de/inf-sp-jnf-demo](http://uhh.de/inf-sp-jnf-demo)

## Train JNF with a fixed look direction

1. Prepare a dataset by running ```data_gen_fixed_pos.py```.
2. Prepare a config file. Examples can be found in the config folder.
3. Run the training script in the scripts folder (replace the path to your config file). 

## Train steerable JNF-SSF

1. Prepare a dataset by running ```data_gen_var_pos.py```.
2. Prepare a config file. Examples can be found in the config folder.
3. Run the training script in the scripts folder (replace the path to your config file). 