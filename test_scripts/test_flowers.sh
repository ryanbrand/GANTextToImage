#!/bin/bash

read -p "Enter a test description of a flower: "  query
echo "Description: $query"

embedding_net="../embedding/cv/lm_sje_flowers_c10_hybrid_0.00070_1_10_trainvalids.txt_iter16400.t7"
echo "Using pretrained embedding net: $embedding_net"

query_str=$query \
embedding_net=$embedding_net \
th save_flower_embeddings.lua

python ../gan/test.py './flower_test_embedding.t7' --cfg=../gan/cfg/eval_flowers.yml

# rm "./flower_test_embedding.t7"
