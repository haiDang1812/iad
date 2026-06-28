# baseline v2 cùng pipeline (mốc để so)
# HF_HUB_OFFLINE=1 python eval_fewshot_bb.py --data_path ../data --model v2_base \
#   --tiles 2 --grid_tile 28 --shots 0 10 --enc_batch 128 --out_dir ./diag_fewshot_v2base

# v3 base
# HF_HUB_OFFLINE=1 python eval_fewshot_bb.py --data_path ../data --model v3_base \
#   --tiles 2 --grid_tile 28 --shots 0 10 --enc_batch 128 --out_dir ./diag_fewshot_v3base

# v3 large
# HF_HUB_OFFLINE=1 python eval_fewshot_bb.py --data_path ../data --model v3_large \
#   --tiles 2 --grid_tile 28 --shots 0 10 --enc_batch 64 --out_dir ./diag_fewshot_v3large

HF_HUB_OFFLINE=1 python eval_fewshot_bb.py --data_path ../data --model v3_large \
  --enc_batch 64 --tiles 2 --grid_tile 28 --shots 0 10 --global_norm --head_w 0.5 0.6 0.7 0.8 1.0 \
  --morph_close 0 --out_dir ./diag_v3l_match

HF_HUB_OFFLINE=1 python eval_fewshot_bb.py --data_path ../data --model v3_large \
  --enc_batch 64 --tiles 2 --grid_tile 28 --shots 0 10 --global_norm --head_w 0.5 0.7 \
  --morph_close 3 --out_dir ./diag_v3l_match_m3