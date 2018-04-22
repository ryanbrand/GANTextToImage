#!/bin/bash

read -p "Enter a test description of a bird: "  query
echo "Description: $query"

embedding_net="../embedding/cv/lm_sje_nc4_cub_hybrid_gru18_a1_c512_0.00070_1_10_trainvalids.txt_iter30000.t7"
echo "Using pretrained embedding net: $embedding_net"

query_str=$query \
embedding_net=$embedding_net \
th save_bird_embeddings.lua

python ../gan/test.py './bird_test_embedding.t7' --cfg=../gan/cfg/eval_birds.yml

#rm "./bird_test_embedding.t7"
