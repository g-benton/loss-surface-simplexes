# Training Classifiers and Simplexes



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