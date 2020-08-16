#!/usr/bin/env bash
# Writing the visual results
python3 ./fl_covid/bin/evaluate_overall_patient_wise.py --model=./model/model.h5  --gpu=0 --save-path=./output/ --save-result=1  --get_predicted_bbox --site=external1
# Print the statistical analysis
python3 ./fl_covid/bin/evaluate_overall_patient_wise.py --model=./model/model.h5  --gpu=0 --save-path=./output/ --save-result=0 --verbose --site=external1