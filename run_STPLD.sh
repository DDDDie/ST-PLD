#!/usr/bin bash
for seed in  0 
do
    for dataset in 'banking' 
    do  
        for known_cls_ratio in 0.25
        do
            for cluster_num_factor in 1.0
            do
                for bottom_k_num in 10
                do
                    python run.py \
                    --dataset $dataset \
                    --known_cls_ratio $known_cls_ratio \
                    --window_size '10' \
                    --bottom_k_num $bottom_k_num \
                    --seed $seed \
                    --train \
                    --tune \
                    --cluster_num_factor $cluster_num_factor \
                    --save_model \
                    --backbone 'bert_STPLD' \
                    --config_file_name 'config_STPLD' \
                    --gpu_id '0' \
                    --results_file_name 'results.csv' \
                    --save_results \
                    --output_dir '../outputs_un_pre/' 
                done
            done
        done
    done
done