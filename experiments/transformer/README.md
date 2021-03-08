## README

Here, we use a slightly different implementation of simplexes that is agnostic to model layers.
It is somewhat less tested, but is more memory efficient as it stores each parameter vector on the CPU.
The memory efficiency allows us to scale to reasonably large image transformer architectures.

First, we fine-tuned ImageNet transformers following the CIFAR100 script in this repo.
https://github.com/jeonsworld/ViT-pytorch.

Then, we ran SPRO on them.
```bash
# command we used to train the simplexes. swap base_idx for ensembles
python simplex_trainer.py --base_idx=0 --train_batch_size=128 --lr_init=0.01 --n_sample=1 --epochs=10
```

We struggled to keep the volume regularization to help out, as evidenced the Appendix of our paper.
However, we think that the approach is promising.

### Application to Other Architectures

The BasicSimplex implementation in the training script is easily adapted to new architectures via
```python
# model is your pre-defined model, that has been pre-trained.
# False says to train that vertex, True says not to train it.
# note that you should also be able to train "from scratch" by not adding a vertex
# for the first vertex
simplex_model = BasicSimplex(model, num_vertices=1, fixed_points=[True]).cuda()

for vv in range(1, args.n_verts+1):
    simplex_model.add_vert()
    simplex_model = simplex_model.cuda()

    for epoch in range(1, args.epochs + 1):
        simp_utils.train_transformer_epoch(
            train_loader,
            simplex_model,
            # your options here
        )
```