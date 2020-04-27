# by default
zones=12
sequence=5

for method in {"svd_entropy_blocks","svd_entropy_blocks_norm","svd_entropy_blocks_permutation","entropy_blocks","entropy_blocks_norm","entropy_blocks_permutation"}; do
    for imnorm in {0,1}; do
        for size in {20,40}; do
            #for sequence in {4,5,6}; do
                    
                output_name=${method}_S${size}_seq${sequence}_imnorm${imnorm}
                output_name_zones=${output_name}_zones${zones}
                
                if [ ! -f data/saved_models/${output_name_zones}.joblib ]; then

                    if [ ! -f data/generated/${output_name} ]; then
                        python processing/prepare_data.py --method ${method} --params ${size},${size} --imnorm ${imnorm} --output ${output_name}
                    fi

                    python processing/prepare_dataset_zones.py --data data/generated/${output_name} --output ${output_name_zones} --sequence ${sequence} --n_zones 12

                    python train_lstm_weighted_v2.py --train data/datasets/${output_name_zones}/${output_name_zones}.train --test data/datasets/${output_name_zones}/${output_name_zones}.test --output ${output_name_zones}
                else
                    echo "${output_name_zones} already generated..."
                fi
            #done
        done
    done
done