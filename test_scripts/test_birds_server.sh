#!/bin/bash

echo "Description: $1"
embedding_net="/home/ubuntu/repo/embedding/cv/lm_sje_nc4_cub_hybrid_gru18_a1_c512_0.00070_1_10_trainvalids.txt_iter30000.t7"

query_str=$1 \
embedding_net=$embedding_net \
th /home/ubuntu/repo/test_scripts/save_bird_embeddings.lua 

python /home/ubuntu/repo/gan/test.py '/home/ubuntu/repo/test_scripts/bird_test_embedding.t7' --cfg=/home/ubuntu/repo/gan/cfg/eval_birds.yml

# python /home/bsk2133/code/GANTextToImage/gan/test.py '/home/bsk2133/code/GANTextToImage/test_scripts/bird_test_embedding.t7' --cfg=/home/bsk2133/code/GANTextToImage/gan/cfg/eval_birds.yml

rm "./bird_test_embedding.t7"
