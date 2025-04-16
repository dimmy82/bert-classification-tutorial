#! /bin/bash

mkdir -p data

curl -O https://www.rondhuit.com/download/ldcc-20140209.tar.gz
mv ./ldcc-20140209.tar.gz ./data/
cd data
tar -zxvf ./ldcc-20140209.tar.gz
cd ..
