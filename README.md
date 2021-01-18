# rfcx-kaggle
Kaggle Rainforest Connection Species Audio Detection

# Step By Step

### Step 1: Preprocessing

```bash
python preprocess.py
```

### Step 2: Training Supervise Contrastive Learning and Fine-tune Classification 

```bash

model_name = "densenet121"
version="v1"

for ((fold=0; fold<=4; fold++))
do
    echo "Start training fold ${fold}"
    python train.py train-model \
        --fold_idx ${fold} \
        --saved_path "./checkpoints/${model_name}_${version}" \
        --pretrained_with_contrastive 1 \

    python train_cls.py train-model \
        --fold_idx ${fold} \
        --saved_path "./checkpoints/${sub_folder}/${model_name}_${version}" \
        --pretrained_with_contrastive 0 \
        --pretrained_path "./checkpoints/${sub_folder}/${model_name}_${version}/pretrained_best_fold${fold}.h5" \
done
```
