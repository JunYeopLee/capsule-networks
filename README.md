# A Tensorflow implementation of Capsule networks

## Requirements
  * tensorflow >= 1.4 (Probably 1.3 should work, too, though I didn't test it)
  * numpy
  * pillow
  * scipy

## Capsule network
I tried to implement the idea in [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)
<img src="figure/capsNet.png">

## File description
  * `config.py` includes all hyper parameters that are needed.
  * `utils.py` contains functions regarding loading and saving.
  * `model.py` has all building blocks for capsNet and whole model implementation.
  * `train.py` is for training.
  * `eval.py` is for evaluation.
  
## Usage
### Training
```
$ python train.py
```
### Evaluation
```
$ python eval.py
```

## Training
<img src="figure/margin_loss.png" width=250px> <img src="figure/reconstruction_loss.png" width=250px> <img src="figure/total_loss.png" width=250px>


## Result
### Classification
| Epoch     | 10     |
|-----------|--------|
| Test Acc. | 99.278 |

### Reconstruction
<img src="figure/test_000.png" width=200px> <img src="figure/test_005.png" width=200px> <img src="figure/test_010.png" width=200px> 
