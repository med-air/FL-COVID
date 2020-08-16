#!/usr/bin/env bash
python3 ./fl_covid/bin/train_fed.py --batch-size=6 --steps=100 --gpu=0 --epochs=30 --snapshot-path=../fl_covid_model_and_log/