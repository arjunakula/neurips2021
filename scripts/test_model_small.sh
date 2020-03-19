python scripts/run_model.py \
  --program_generator data/backup_models/baseline_18k_single_object_largeBatchSize/program_generator.pt_32000 \
  --execution_engine data/run_fixedPG+EE_ref_singleObject_small/execution_engine_with_lang_attention.pt_150000 \
  --input_refexp_h5 data/val_refexps_singleObject.h5 \
  --input_features_h5 data/val_features.h5 \
  --batch_size 8 \
  --result_output_path ./data/result.json