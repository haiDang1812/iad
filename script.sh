# baseline v2 cùng pipeline (mốc để so)
# HF_HUB_OFFLINE=1 python eval_fewshot_bb.py --data_path ../data --model v2_base \
#   --tiles 2 --grid_tile 28 --shots 0 10 --enc_batch 128 --out_dir ./diag_fewshot_v2base

# v3 base
# HF_HUB_OFFLINE=1 python eval_fewshot_bb.py --data_path ../data --model v3_base \
#   --tiles 2 --grid_tile 28 --shots 0 10 --enc_batch 128 --out_dir ./diag_fewshot_v3base

# v3 large
# HF_HUB_OFFLINE=1 python eval_fewshot_bb.py --data_path ../data --model v3_large \
#   --tiles 2 --grid_tile 28 --shots 0 10 --enc_batch 64 --out_dir ./diag_fewshot_v3large

# HF_HUB_OFFLINE=1 python eval_fewshot_bb.py --data_path ../data --model v3_large \
#   --enc_batch 64 --tiles 2 --grid_tile 28 --shots 0 10 --global_norm --head_w 0.5 0.6 0.7 0.8 1.0 \
#   --morph_close 0 --out_dir ./diag_v3l_match

# HF_HUB_OFFLINE=1 python eval_fewshot_bb.py --data_path ../data --model v3_large \
#   --enc_batch 64 --tiles 2 --grid_tile 28 --shots 0 10 --global_norm --head_w 0.5 0.7 \
#   --morph_close 3 --out_dir ./diag_v3l_match_m3

# CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python diagnosis_scripts/diag26_head_shift_direction.py     --data_path ../data --out_dir ./diag26

# CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python eval_shiftsim.py \
#     --data_path ../data --out_dir ./shiftsim --bright 0.6 --jitter 0.2

# CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python eval_overlap.py \
#     --data_path ../data --out_dir ./ov_aupro \
#     --configs 3:40:0.5 4:40:0.5 --canvas 512 --agg mean \
#     --enc_batch 4 --max_train 40 --max_eval 15 \
#     --categories rice vial sheet_metal fabric fruit_jelly

# CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python infer_submit_mvtec_ad2.py     --data_path ../data --out_dir ./submit_res324     --tiles 3 --grid_tile 24 --enc_batch 16     --thr_mode test_ksig --thr_sigma 4.5

# CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# python eval_native.py --data_path ../data --out_dir ./native324 \
# --model v3_large --tiles 3 --grid_tile 24 --max_eval 100000 \
# --categories can fabric sheet_metal walnuts

# CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# python eval_nrs_head.py --data_path ../data --out_dir ./nrs24 \
# --model v3_large --tiles 3 --grid_tile 24 --categories can wallplugs fruit_jelly

# CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# python eval_nrs_head.py --data_path ../data --out_dir ./nrs48 \
# --model v3_large --tiles 3 --grid_tile 48 --categories can wallplugs fruit_jelly

# CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# python eval_nrs_head.py --data_path ../data --out_dir ./nrs48b \
# --model v3_large --tiles 3 --grid_tile 48 --categories fabric sheet_metal walnuts

# CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# python eval_nrs_head.py --data_path ../data --out_dir ./nrs24rv \
# --model v3_large --tiles 3 --grid_tile 24 --categories rice vial

# thrrules (ưu tiên trước — quyết định ngưỡng cho sub v2)
# CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# python eval_thr_rules.py --data_path ../data --out_dir ./thrrules

# # nrsab48
# CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# python eval_nrs_alpha.py --data_path ../data --out_dir ./nrsab48 \
# --model v3_large --tiles 3 --grid_tile 48 --categories fabric can sheet_metal

# CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python eval_nrs_head.py \
#     --data_path ../data --out_dir ./nrs72 \
#     --model v3_large --tiles 3 --grid_tile 72 --enc_batch 4 \
#     --categories can vial wallplugs

# CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python infer_submit_nrs2.py \
#     --data_path ../data --out_dir ./submit_nrs2

# for S in 10 25 50; do
#     CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python eval_nrs_head.py --data_path ../data \
#       --tiles 3 --grid_tile 48 --shots $S --eval_reserve 50 --max_eval 80 \
#       --categories can wallplugs --out_dir ./shotcurve/s${S}_48
#     CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python eval_nrs_head.py --data_path ../data \
#       --tiles 3 --grid_tile 24 --shots $S --eval_reserve 50 --max_eval 80 \
#       --categories vial --out_dir ./shotcurve/s${S}_24
# done

CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python infer_submit_nrs3.py --data_path ../data --out_dir ./submit_nrs3

rm submit_nrs3/log.txt

cd MVTecAD2_public_code_utils

python check_and_prepare_data_for_upload.py ../submit_nrs3