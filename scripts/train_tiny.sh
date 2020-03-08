#!/bin/bash
 python scripts/train_model.py \
  --program_generator_start_from ./data/backup_models/baseline_18k_single_object_largeBatchSize/program_generator.pt_32000 \
  --train_execution_engine  1 \
  --train_program_generator 0 \
  --model_type PG+EE \
  --num_iterations 50 \
  --learning_rate 1e-4 \
  --checkpoint_path data/run_fixedPG+EE_ref_singleObject_tiny/execution_engine.pt \
  --checkpoint_every 10 \
  --train_refexp_h5 data/tiny_dataset/train_refexps.h5 \
  --train_features_h5 data/train_features.h5 \
  --val_refexp_h5 data/tiny_dataset/val_refexps.h5 \
  --val_features_h5 data/val_features.h5 \
  --vocab_json data/vocab.json \
  --batch_size 8 \
  --feature_dim 1024,20,20