## Training
## Annotation-only
```bash
CUDA_VISIBLE_DEVICES=3 python run.py \
    --do_train \
    --do_eval \
    --do_test \
    --do_lower_case \
    --model_type roberta \
    --model_name_or_path microsoft/codebert-base \
    --train_filename ../data/3-lines/annotation-only/train.annotation,../data/3-lines/annotation-only/train.all.seq \
    --dev_filename ../data/3-lines/annotation-only/valid.annotation,../data/3-lines/annotation-only/valid.all.seq \
    --test_filename ../data/3-lines/annotation-only/test.annotation,../data/3-lines/annotation-only/test.all.seq \
    --output_dir 20-Oct-ant \
    --max_source_length 256 \
    --max_target_length 256 \
    --beam_size 5 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --num_train_epochs 30 \
    2>&1 | tee 20-Oct-ant/train.log
```

```bash
CUDA_VISIBLE_DEVICES=1,2 python run.py \
        --do_train \
        --do_eval \
        --do_test \
        --do_lower_case \
        --load_model_path ./20-Oct-ant/checkpoint-last/pytorch_model.bin \
        --model_type roberta \
        --model_name_or_path microsoft/codebert-base \
        --train_filename ../data/3-lines/annotation-only/train.annotation,../data/3-lines/annotation-only/train.all.seq \
        --dev_filename ../data/3-lines/annotation-only/valid.annotation,../data/3-lines/annotation-only/valid.all.seq \
        --test_filename ../data/3-lines/annotation-only/test.annotation,../data/3-lines/annotation-only/test.all.seq \
        --output_dir ./22-Oct-code \
        --max_source_length 256 \
        --max_target_length 256 \
        --beam_size 5 \
        --train_batch_size 16 \
        --eval_batch_size 32 \
        --learning_rate 5e-5 \
        --num_train_epochs 18 \
        2>&1 | tee ./22-Oct-code/train.log
```

## Continue training
### Code-only
```bash
CUDA_VISIBLE_DEVICES=1 python run.py \
        --do_train \
        --do_eval \
        --do_test \
        --do_lower_case \
        --load_model_path ./3-lines-code/checkpoint-best-bleu/pytorch_model.bin \
        --model_type roberta \
        --model_name_or_path microsoft/codebert-base \
        --train_filename ../data/3-lines/code-only/train.code,../data/3-lines/code-only/train.seq \
        --dev_filename ../data/3-lines/code-only/valid.code,../data/3-lines/code-only/valid.seq \
        --test_filename ../data/3-lines/code-only/test.code,../data/3-lines/code-only/test.seq \
        --output_dir ./21-Oct-code \
        --max_source_length 256 \
        --max_target_length 256 \
        --beam_size 5 \
        --train_batch_size 32 \
        --eval_batch_size 32 \
        --learning_rate 5e-5 \
        --num_train_epochs 20 \
        2>&1 | tee ./21-Oct-code/train.log
```
## Testing
### Annotation-only
```bash
CUDA_VISIBLE_DEVICES=0 python run.py \
    --do_test \
    --model_type roberta \
    --model_name_or_path microsoft/codebert-base \
    --load_model_path 3-lines-annotation/checkpoint-best-bleu/pytorch_model.bin \
    --test_filename ../data/3-lines/annotation-only/test.annotation,../data/3-lines/annotation-only/test.all.seq \
    --output_dir 3-lines-annotation \
    --eval_batch_size 32 \
    2>&1 | tee 3-lines-annotation/test-22-Oct.log
```

### Code-only

```bash
CUDA_VISIBLE_DEVICES=3 python run.py \
    --do_test \
    --model_type roberta \
    --model_name_or_path microsoft/codebert-base \
    --load_model_path 3-lines-code/checkpoint-best-bleu/pytorch_model.bin \
    --test_filename ../data/3-lines/code-only/test.code,../data/3-lines/code-only/test.seq \
    --output_dir 3-lines-code \
    --eval_batch_size 32 \
    2>&1 | tee 3-lines-code/test-20-Oct.log
```

### Annotation + Code

```bash
CUDA_VISIBLE_DEVICES=3 python run.py \
    --do_test \
    --model_type roberta \
    --model_name_or_path microsoft/codebert-base \
    --load_model_path 3-lines-bimodal/checkpoint-best-bleu/pytorch_model.bin \
    --test_filename ../data/3-lines/bimodal/test.bimodal,../data/3-lines/bimodal/test.seq \
    --output_dir 3-lines-bimodal \
    --eval_batch_size 32 \
    2>&1 | tee 3-lines-bimodal/test-20-Oct.log
```