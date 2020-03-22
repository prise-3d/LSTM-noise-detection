method="svd_entropy"
zones=12

for norm in {0,1}; do
    for imnorm in {0,1}; do
        for start in {0,50,100,150}; do
            for end in {50,100,150,200}; do
                for sequence in {2,3,4,5,6,7,8,9,10}; do
                    result=$((end - start))
                    if [ $result -gt 0 ]; then
                        
                        if [ $norm -eq 1 ]; then
                            method_used="${method}_norm"
                        else
                            method_used="${method}"
                        fi

                        output_name=${method_used}_${start}_${end}_imnorm${imnorm}_seq${sequence}
                        output_name_zones=${output_name}_zones${zones}
                        
                        python processing/prepare_data.py --method ${method_used} --params ${start},${end} --imnorm ${imnorm} --output ${output_name}

                        python processing/prepare_dataset_zones.py --data data/generated/${output_name} --output ${output_name_zones} --sequence ${sequence} --n_zones 12

                        python train_lstm.py --train data/datasets/${output_name_zones}/${output_name_zones}.train --test data/datasets/${output_name_zones}/${output_name_zones}.test --output ${output_name_zones}
                    fi
                done
            done
        done
    done
done