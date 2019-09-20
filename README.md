# Example Forgetting during Deep Neural Network Training


## Run Experiments:

```
python main.py \
    --dataset [cifar10/cifar100/svhn] 
    --output_dir results
    --data_augmentation
    --epochs 200
    --burn_in_epochs 50
    --noise_percent_labels 30
    --remove_percent 40
    --remove_strategy ["normal"/"unforgettable"/"random"/"low-acc"]
    --device "cuda:5"
    --seed 1
```
