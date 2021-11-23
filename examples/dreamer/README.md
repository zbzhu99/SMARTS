## Citation

```
@article{hafner2020dreamerv2,
  title={Mastering Atari with Discrete World Models},
  author={Hafner, Danijar and Lillicrap, Timothy and Norouzi, Mohammad and Ba, Jimmy},
  journal={arXiv preprint arXiv:2010.02193},
  year={2020}
}
```

## Resources

- [Google AI Blog post](https://ai.googleblog.com/2021/02/mastering-atari-with-discrete-world.html)
- [Project website](https://danijar.com/dreamerv2/)
- [Research paper](https://arxiv.org/pdf/2010.02193.pdf)

## Manual Instructions

Monitor results:

```sh
tensorboard --logdir ~/logdir
```

Generate plots:

```sh
python3 common/plot.py --indir ~/logdir --outdir ~/plots \
  --xaxis step --yaxis eval_return --bins 1e6
```
