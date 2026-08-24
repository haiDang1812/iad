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

# CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python infer_submit_nrs3.py --data_path ../data --out_dir ./submit_nrs3

# rm submit_nrs3/log.txt

# cd MVTecAD2_public_code_utils

# python check_and_prepare_data_for_upload.py ../submit_nrs3

# CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python infer_submit_uniform.py \
#     --data_path ../data --out_dir ./submit_uniform 2>&1 | tee log_uniform.txt

# rm submit_uniform/log.txt

# cd MVTecAD2_public_code_utils

# python check_and_prepare_data_for_upload.py ../submit_uniform

# CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python eval_generalize.py \
#     --dataset visa --data_path /workspace/data/visa/1cls \
#     --out_dir ./gen_visa 2>&1 | tee log_gen_visa.txt

# CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python eval_segf1_morph.py \
#     --data_path ../data --out_dir ./morph 2>&1 | tee log_morph.txt


# A1 — multi-seed (chốt 0.069 ± σ), eff_grid 144, 8 cat × 5 seed
# CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python eval_nrs_ablation.py --data_path ../data --out_dir ./abl_seed \
#  --grids 48 --seeds 0 1 2 3 4

# # A2 — grid-sweep, eff_grid 48/72/96/144/192, seed 0
# CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python eval_nrs_ablation.py --data_path ../data --out_dir ./abl_grid \
#  --grids 16 24 32 48 64 --seeds 0

# A4 — MVTec AD generalize (cần data MVTec-AD ở ../data_mvtec)
# CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python eval_generalize.py --dataset mvtec --data_path ../data_mvtec --out_dir ./gen_mvtec

# CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python eval_nrs_ablation.py --data_path ../data
#   --out_dir ./abl_grid2 --grids 8 12 --seeds 0

# python eval_adapter.py --data_path ../data --out_dir ./adapter3 --max_eval 30 \
#     --neg_mode adv --lam_s 4 --beta_lo 0.5 --beta_hi 1.5 --margin 1.5 \
#     --categories vial wallplugs sheet_metal fabric rice walnuts can fruit_jelly

python eval_fairthr.py --data_path ../data --out_dir ./fairthr192 --max_eval 30 --tiles 4 \
      --categories vial wallplugs sheet_metal fabric rice walnuts can fruit_jelly