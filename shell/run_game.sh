#!/usr/bin/env bash

advertisers="1458 2259 2261 2821 2997 3358 3386 3427 3476"
#advertisers="3386"

for advertiser in $advertisers; do
    mkdir -p ../result/$advertiser/log/run_game
    mkdir -p ../result/$advertiser/run_game
    echo "run [python ../run_game.py ../result/$advertiser]"
    python ../run_game.py ../result/$advertiser\
        1>"../result/$advertiser/log/run_game/1.log" 2>"../result/$advertiser/run_game/2.log"&
done