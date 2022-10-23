## Training
### single-annotation
```bash
CUDA_VISIBLE_DEVICES=7 python single-run.py --max_length 256 --batch_size 16 --output_dir '19-Oct-single-annotation'
```

### single-code
```bash
CUDA_VISIBLE_DEVICES=5 python single-code-run.py --max_length 256 --batch_size 16 --output_dir '19-Oct-single-code'
```

Continuing traning
```bash
CUDA_VISIBLE_DEVICES=1 python single-code-run.py --max_length 256 --batch_size 16 --epoch 6 --norm True --load_model_path 18-Oct-single-code/checkpoint-best-bleu/pytorch_model.bin --output_dir '21-Oct-single-code'
```

### dual-model
```bash
CUDA_VISIBLE_DEVICES=3 python dual-run.py --max_length 256 --batch_size 16 --epoch 30 --fuse True --norm True --output_dir '21-Oct-dual-ant-only'
```

continue training
```bash
CUDA_VISIBLE_DEVICES=1 python dual-run.py --max_length 256 --batch_size 16 --epoch 20 --output_dir '18-Oct-dual' --load_model_path '14-Oct-dual/checkpoint-best-bleu/pytorch_model.bin'
```


### tri-model
```bash
CUDA_VISIBLE_DEVICES=2 python tri-run.py --max_length 64 --batch_size 64 --output_dir '17-Oct-tri-biker'

CUDA_VISIBLE_DEVICES=0 python tri-run.py --max_length 64 --batch_size 64 --output_dir '17-Oct-tri-bert'
```

## Evaluation
### single-annotation
```bash
CUDA_VISIBLE_DEVICES=3 python evaluate.py --model_type 0 --max_length 256 --load_model_path ./14-Oct-single-256/checkpoint-best-bleu/pytorch_model.bin --test_filename ../data/test_3_lines_dedup.csv --output_dir 14-Oct-single-256/res
```

### single-code
```bash
CUDA_VISIBLE_DEVICES=1 python evaluate.py --model_type 1 --max_length 256 --load_model_path 13-Oct-single-code/checkpoint-best-bleu/pytorch_model.bin --test_filename ../data/test_3_lines_dedup.csv --output_dir 13-Oct-single-code/res
```

### dual-model
```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_type 2 --load_model_path 14-Oct-dual/checkpoint-best-bleu/pytorch_model.bin --test_filename ../data/test_3_lines_dedup.csv --output_dir 14-Oct-dual/res-1 --max_length 256
```

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model_type 2 --max_length 256 --load_model_path 18-Oct-dual-concat/checkpoint-best-bleu/pytorch_model.bin --test_filename ../data/test_3_lines_dedup.csv --output_dir 18-Oct-dual-concat/res
```


```bash
CUDA_VISIBLE_DEVICES=3 python evaluate.py --model_type 2 --load_model_path 14-Oct-dual/checkpoint-best-bleu/pytorch_model.bin --test_filename ../data/test_3_lines_dedup.csv --output_dir 14-Oct-dual/ant-only --max_length 256
```

```bash
CUDA_VISIBLE_DEVICES=3 python evaluate.py --model_type 2 --load_model_path 14-Oct-dual/checkpoint-best-bleu/pytorch_model.bin --test_filename ../data/test_3_lines_dedup.csv --output_dir 14-Oct-dual/code-only --max_length 256
```

### tri-model
```bash
CUDA_VISIBLE_DEVICES=2 python evaluate.py --model_type 3 --load_model_path 13-Oct-tri/checkpoint-best-bleu/pytorch_model.bin --test_filename ../data/test_3_lines_dedup.csv --output_dir 13-Oct-tri/res
```

## Calculate the results
### single-annotation
```bash
python calculate_bleu_score.py --reference end-to-end/13-Oct-single/res/test_ref.csv --candidate end-to-end/13-Oct-single/res/test_hyp.csv
```

### single-code
```bash
python calculate_bleu_score.py --reference end-to-end/13-Oct-single-code/res/test_ref.csv --candidate end-to-end/13-Oct-single-code/res/test_hyp.csv
```

### dual-model
```bash
python calculate_bleu_score.py --reference end-to-end/14-Oct-dual/res/test_ref.csv --candidate end-to-end/14-Oct-dual/res/test_hyp.csv
```

### tri-model
```bash
python calculate_bleu_score.py --reference end-to-end/13-Oct-tri/res/test_ref.csv --candidate end-to-end/13-Oct-tri/res/test_hyp.csv
```