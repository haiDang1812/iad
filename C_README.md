```bash
hf download lhd181204/reproduce_inp --local-dir ./reproduced_results
python diagnosis3_score_distribution.py --data_path ../data --ckpt_dir ./reproduced_results
python diagnosis4_inp_contamination.py --data_path /path/to/mvtecad2 --ckpt_dir ./reproduced_results
python diagnosis5_decoder_attention.py --data_path /path/to/mvtecad2 --ckpt_dir ./reproduced_results
```