#!/bin/bash
python scripts/preprocess_refexps.py   --input_refexps_json data/clevr_ref+_1.0/refexps/clevr_ref+_val_refexps_singleObject_tiny.json   --input_scenes_json data/clevr_ref+_1.0/scenes/clevr_ref+_val_scenes.json   --num_examples -1   --output_h5_file data/tiny_dataset/val_refexps.h5   --height 320   --width 320   --output_vocab_json data/vocab.json
python scripts/preprocess_refexps.py   --input_refexps_json data/clevr_ref+_1.0/refexps/clevr_ref+_val_refexps_singleObject_smaller.json   --input_scenes_json data/clevr_ref+_1.0/scenes/clevr_ref+_val_scenes.json   --num_examples -1   --output_h5_file data/smaller_dataset/val_refexps.h5   --height 320   --width 320   --output_vocab_json data/vocab.json
python scripts/preprocess_refexps.py   --input_refexps_json data/clevr_ref+_1.0/refexps/clevr_ref+_val_refexps_singleObject_small.json   --input_scenes_json data/clevr_ref+_1.0/scenes/clevr_ref+_val_scenes.json   --num_examples -1   --output_h5_file data/small_dataset/val_refexps.h5   --height 320   --width 320   --output_vocab_json data/vocab.json
python scripts/preprocess_refexps.py   --input_refexps_json data/clevr_ref+_1.0/refexps/clevr_ref+_val_refexps_singleObject_medium.json   --input_scenes_json data/clevr_ref+_1.0/scenes/clevr_ref+_val_scenes.json   --num_examples -1   --output_h5_file data/medium_dataset/val_refexps.h5   --height 320   --width 320   --output_vocab_json data/vocab.json