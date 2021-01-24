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

    python train.py train-model \
        --fold_idx ${fold} \
        --saved_path "./checkpoints/${model_name}_${version}" \
        --pretrained_with_contrastive 0 \
        --pretrained_path "./checkpoints/${model_name}_${version}/pretrained_best_fold${fold}.h5" \
done
```

### Step 3: Multi-scale evaluation on each fold

```bash

model_name = "densenet121"
version="v1"

for ((fold=0; fold<=4; fold++))
do
    python evaluate.py run-multi-scale-eval \
        --fold ${fold} \
        --checkpoints_path "./checkpoints/${model_name}_${version}"
done

```

### Step 4: Multi-scale inference on test-set

```bash

model_name = "densenet121"
version="v1"

for ((fold=0; fold<=4; fold++))
do
    python prediction.py run-prediction \
        --fold ${fold} \
        --checkpoints_path "./checkpoints/${model_name}_${version}"
done

```

### Step 5: Ensemble

```bash

model_name = "densenet121"
version="v1"

python ensemble.py run_ensemble \
    --checkpoints_path "./checkpoints/${model_name}_${version}"

```
