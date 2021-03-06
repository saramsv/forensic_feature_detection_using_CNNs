#!/usr/bin/env bash


declare -a learning_rate=(0.002)  #(0.2 0.02 0.002)
epochs=200
declare -a size=(512) # 256 512)
#backbone = ['resnet50', 'resnet101'] for now let's control this in the code. 

DIR="$(pwd)"
im_dir="$(dirname "$(pwd)")"

for lr in "${learning_rate[@]}"
do
	for s in "${size[@]}"
	do
	    #python3 training_unnormalized.py $im_dir'/all_images/all_imgs/' $DIR'/mummification_not_normalized_orig.csv' 5+ $epochs $lr $s
	    python3 training_unnormalized.py $im_dir'/MummImgsAugm/' $DIR'/Mumm.20181001.augmented.sorted.csv' 5+ $epochs $lr $s

	done
done

