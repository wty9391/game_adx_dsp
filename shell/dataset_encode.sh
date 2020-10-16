#!/usr/bin/env bash
ipinyou="/home/wty/datasets/make-ipinyou-data/"
advertisers="1458 2261 2997 3386 3476 2259 2821 3358 3427 all"
#advertisers="2997"
for advertiser in $advertisers; do
    echo "run [python ../ipinyou_dataset_encode.py $ipinyou/$advertiser/train.log.txt $ipinyou/$advertiser/test.log.txt $ipinyou/$advertiser/featindex.txt ../result/$advertiser]"
    mkdir -p ../result/$advertiser/log/dataset_encode
    python ../ipinyou_dataset_encode.py $ipinyou/$advertiser/train.log.txt $ipinyou/$advertiser/test.log.txt $ipinyou/$advertiser/featindex.txt ../result/$advertiser\
        1>"../result/$advertiser/log/dataset_encode/1.log" 2>"../result/$advertiser/log/dataset_encode/2.log"&
done