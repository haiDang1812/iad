python eval_dist_only.py --data_path ../data --encoder dinov2reg_vit_base_14 \
    --layers 2 3 4 5 6 7 8 9 --coreset_ratio 0.25 --region_mode mult \
    --out_dir ./diagnosis_novelty/diagnosis_region_mult
python eval_dist_only.py --data_path ../data --encoder dinov2reg_vit_base_14 \
    --layers 2 3 4 5 6 7 8 9 --coreset_ratio 0.25 --region_mode gmean \
    --out_dir ./diagnosis_novelty/diagnosis_region_gmean

python eval_dist_only.py --data_path ../data --encoder dinov2reg_vit_base_14 \
    --layers 2 3 4 5 6 7 8 9 --coreset_ratio 0.25 --region_mode median \
    --out_dir ./diagnosis_novelty/diagnosis_region_median
python eval_dist_only.py --data_path ../data --encoder dinov2reg_vit_base_14 \
    --layers 2 3 4 5 6 7 8 9 --coreset_ratio 0.25 --region_mode open \
    --out_dir ./diagnosis_novelty/diagnosis_region_open