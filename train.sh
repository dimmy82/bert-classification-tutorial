#! /bin/bash

TOP_K=3 BEST_SCORE_KEY=recall EVALUATE_SUMMARY_TYPE=micro python src/train.py > train.log 2>&1