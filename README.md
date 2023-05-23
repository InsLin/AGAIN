# AGAIN: Adversarial Training with Attribution Span Enlargement and Hybrid Feature Fusion
The Offical Code of CVPR2023: AGAIN: Adversarial Training with Attribution Span Enlargement and Hybrid Feature Fusion

## Requirements
The code has been implemented and tested with Python 3.7.15. To install the required packages:

```bash
$ pip install -r requirements.txt
```

## Usage
### Training Commands

```
$ python AGAIN_with_AWP/train_awp_again_cifar.py --arch <model_architecture> \
	--batch-size <train_batch_size> \
	--test-batch-size <test_batch_size> \
	--epochs 200 \
	--start_epoch 1 \
	--data <name_of_the_dataset>
	--data-path <path_to_dataset>
	--lr 0.1 \
	--epsilon 8 \
	--num-steps 10 \
	--step-size 2 \
	--model-dir <model_dir> \
	--save-freq <save_freq> \
```

### Evaluation Commands

```
$ python AGAIN_with_AWP/attack_test.py --data  <name_of_the_dataset> \
	--model_path <path_to_model>
```
