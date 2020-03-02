# IEP-Ref

This is the code for the IEP-Ref model, a module network approach for referring expression problems. See our [paper](https://arxiv.org/abs/1901.00850):
```bash
@article{liu2019clevr,
  author    = {Runtao Liu and
               Chenxi Liu and
               Yutong Bai and
               Alan Yuille},
  title     = {CLEVR-Ref+: Diagnosing Visual Reasoning with Referring Expressions},
  journal   = {arXiv preprint arXiv:1901.00850},
  year      = {2019}
}
```

## Setup

All code was developed and tested on Ubuntu 16.04 with Python 3.5.

You can set up a virtual environment to run the code like this:

```bash
virtualenv -p python3 .env       # Create virtual environment
source .env/bin/activate         # Activate virtual environment
pip install -r requirements.txt  # Install dependencies
echo $PWD > .env/lib/python3.5/site-packages/iep.pth # Add this package to virtual environment
# Work for a while ...
deactivate # Exit virtual environment
```


## Preprocessing CLEVR-Ref+

Before you can train any models, you need to download the CLEVR-Ref+ dataset, extract features for the images, and preprocess the referring expressions and programs.

### Step 1: Download the data

First you need to download and unpack the [CLEVR-Ref+ dataset](https://cs.jhu.edu/~cxliu/data/clevr_ref+_1.0.zip).
For the purpose of this tutorial we assume that all data will be stored in a new directory called `data/`:

### Step 2: Extract Image Features

Extract ResNet-101 features for the CLEVR-Ref+ train and val images with the following commands:

```bash
python scripts/extract_features.py \
  --input_image_dir data/clevr_ref+_1.0/images/train/ \
  --output_h5_file data/train_features.h5 \
  --image_height 320 \
  --image_width 320 \
  --batch_size 32 

python scripts/extract_features.py \
  --input_image_dir data/clevr_ref+_1.0/images/val/ \
  --output_h5_file data/val_features.h5 \
  --image_height 320 \
  --image_width 320 \
  --batch_size 32 
```

### Step 3: Preprocess Referring Expressions

Preprocess the referring expressions and programs for the CLEVR-Ref+ train and val sets with the following commands:

```bash
python scripts/preprocess_refexps.py \
  --input_refexps_json data/clevr_ref+_1.0/refexps/clevr_ref+_train_refexps.json \
  --input_scenes_json data/clevr_ref+_1.0/scenes/clevr_ref+_train_scenes.json \
  --num_examples -1 \
  --output_h5_file data/train_refexps.h5 \
  --height 320 \
  --width 320 \
  --output_vocab_json data/vocab.json

python scripts/preprocess_refexps.py \
  --input_refexps_json data/clevr_ref+_1.0/refexps/clevr_ref+_val_refexps.json \
  --input_scenes_json data/clevr_ref+_1.0/scenes/clevr_ref+_val_scenes.json \
  --num_examples -1 \
  --output_h5_file data/val_refexps.h5 \
  --height 320 \
  --width 320 \
  --input_vocab_json data/vocab.json
```

When preprocessing referring expressions, we create a file `vocab.json` which stores the mapping between
tokens and indices for referring expressions and programs. We create this vocabulary when preprocessing
the training referring expressions, then reuse the same vocabulary file for the val referring expressions.

## Training on CLEVR-Ref+

Models are trained through a three-step procedure:

1. Train the program generator using a small number of ground-truth programs
2. Train the execution engine using predicted outputs from the trained program generator
3. Jointly fine-tune both the program generator and the execution engine without any ground-truth programs

All training code runs on GPU, and assumes that CUDA and cuDNN already been installed.

### Step 1: Train the Program Generator

In this step we use a small number of ground-truth programs to train the program generator:

```bash
mkdir data/run_PG_ref_18k

python scripts/train_model.py \
  --model_type PG \
  --num_iterations 32000 \
  --num_train_samples 18000 \
  --checkpoint_every 1000 \
  --checkpoint_path data/run_PG_ref_18k/program_generator.pt \
  --batch_size 64 \
  --train_refexp_h5 data/train_refexps.h5 \
  --train_features_h5 data/train_features.h5 \
  --val_refexp_h5 data/val_refexps.h5 \
  --val_features_h5 data/val_features.h5 \
  --vocab_json data/vocab.json \
```

### Step 2: Train the Execution Engine

In this step we train the execution engine, based on the trained program generator:

```bash
mkdir data/run_fixedPG+EE_ref

python scripts/train_model.py \
  --program_generator_start_from ./data/run_PG_ref_18k/program_generator.pt_32000 \
  --train_execution_engine  1 \
  --train_program_generator 0 \
  --model_type PG+EE \
  --num_iterations 450000 \
  --learning_rate 1e-4 \
  --checkpoint_path data/run_fixedPG+EE_ref/execution_engine.pt \
  --checkpoint_every 5000 \
  --train_refexp_h5 data/train_refexps.h5 \
  --train_features_h5 data/train_features.h5 \
  --val_refexp_h5 data/val_refexps.h5 \
  --val_features_h5 data/val_features.h5 \
  --vocab_json data/vocab.json \
  --batch_size 48 \
  --feature_dim 1024,20,20
```

Another option is that you can also train the execution engine using the ground-truth programs.

```bash
mkdir run_gt_EE

python scripts/train_model.py \
  --model_type EE \
  --num_iterations 450000 \
  --learning_rate 1e-4 \
  --checkpoint_path data/run_gt_EE/execution_engine.pt \
  --checkpoint_every 5000 \
  --train_refexp_h5 data/train_refexps.h5 \
  --train_features_h5 data/train_features.h5 \
  --val_refexp_h5 data/val_refexps.h5 \
  --val_features_h5 data/val_features.h5 \
  --vocab_json data/vocab.json \
  --batch_size 48 \
  --feature_dim 1024,20,20
```



### Step 3: Jointly train entire model

In this step we jointly train the program generator and execution engine using REINFORCE:

```bash
mkdir data/run_jointPG+EE_ref

python scripts/train_model.py \
  --program_generator_start_from data/run_PG_ref_18k/model/program_generator.pt_32000 \
  --execution_engine_start_from data/run_fixedPG+EE_ref/model/execution_engine.pt_450000 \
  --train_execution_engine  1 \
  --train_program_generator 1 \
  --model_type PG+EE \
  --num_iterations 200000 \
  --learning_rate 5e-5 \
  --checkpoint_path data/run_jointPG+EE_ref/joint_pg_ee.pt \
  --checkpoint_every 5000 \
  --train_refexp_h5 data/train_refexps.h5 \
  --train_features_h5 data/train_features.h5 \
  --val_refexp_h5 data/val_refexps.h5 \
  --val_features_h5 data/val_features.h5 \
  --vocab_json data/vocab.json \
  --batch_size 48 \
  --feature_dim 1024,20,20
```

### Step 4: Test the model

You can use the `run_model.py` script to test your model on the entire validation
sets. 

To test the model based on the programs predicted from program generator:
```bash
python scripts/run_model.py \
  --program_generator /path/to/pg.pt \
  --execution_engine /path/to/ee.pt \
  --input_refexp_h5 data/val_refexps.h5 \
  --input_features_h5 data/val_features.h5 \
  --batch_size 24 \
  --result_output_path ./data/result.json
```

To test the model using the ground-truth programs:
```bash
python scripts/run_model.py \
  --program_generator /path/to/pg.pt \
  --execution_engine /path/to/ee.pt \
  --input_refexp_h5 data/val_refexps.h5 \
  --input_features_h5 data/val_features.h5 \
  --batch_size 24 \
  --result_output_path ./data/result.json \
  --use_gt_programs 1
```
