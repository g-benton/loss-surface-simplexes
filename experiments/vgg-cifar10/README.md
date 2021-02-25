# Training Classifiers and Simplexes

#### Ensemble Trainer

The core method provided in the paper is ESPRO, which functions as an ensemble of simplexes.

To train an ESPRO ensemble as is done in the paper run the following

```bash
python3 ensemble_trainer.py --data_path=<YOUR DATA PATH> \
                        --base_epochs=300 \
                        --simplex_epochs=10 \
                        --base_lr=0.05 \
                        --simplex_lr=0.01 \
                        --wd=5e-4 \
                        --LMBD=1e-6\
                        --n_component=<NUMBER OF ENSEMBLE COMPONENTS> \
                        --n_verts=<NUMBER OF VERTICES PER SIMPLEX> \ 
```

Each component model will be saved independently in the `saved-outputs` folder.

#### Base Models

To train the base models run
```bash
python3 base_trainer.py --data_path=<YOUR DATA PATH> \
                        --epochs=300 \
                        --lr_init=0.05 \
                        --wd=5e-4
```
each time you run this a new model will be trained and saved in the `saved-outputs` folder.


#### Simplexes 
To use pretrained base models to train simplexes around the SGD-found solutions run

```bash
python3 simplex_trainer.py --data_path=<YOUR DATA PATH> \
                           --epochs=10 \
                           --lr_init=0.01 \
                           --wd=5e-4 \
                           --base_idx=<INDEX OF PRETRAINED MODEL> \
                           --n_verts=<TOTAL NUMBER OF VERTICES IN SIMPLEX> \
                           --n_sample=5
```

#### Mode Connecting Complexes

To find simplicial complexes that connect modes in parameter space use
```bash
python3 simplex_trainer.py --data_path=<YOUR DATA PATH> \
                           --epochs=25 \
                           --lr_init=0.01 \
                           --wd=5e-4 \
                           --n_verts=<HOW MANY PRETRAINED MODELS TO CONNECT> \
                           --n_connector=<HOW MANY CONNECTING POINTS TO USE> \
                           --n_sample=5
```
Here `n_verts` is the number of independently trained models you want to connect through `n_connector` points. 
