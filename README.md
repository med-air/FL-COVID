# FL-COVID-19
"Federated deep learning for detecting COVID-19 lung abnormalities in CT: A privacy-preserving multinational validation study" by

Qi Dou, Tiffany So, Meirui Jiang, Quande Liu, VARUT Vardhanabhuti, Georgios Kaissis, Zeju Li, Weixin Si, Heather Lee, Kevin Yu, Zuxin Feng, Li Dong, Egon Burian, Friederike Jungmann, Rickmer Braren, Prof. Marcus Makowski, Bernhard Kainz, Daniel Rueckert, Ben Glocker*, Simon Yu*, Pheng Ann Heng*

For any inquiry, please contact Dr. Qi Dou (qdou@cse.cuhk.edu.hk). 

## Source code
The provided demo data and source code enables training and testing of our federated deep learning method for abnormalities detection in COVID-19 lung CT.

A model trained on a set of 60 annotated CT scans obtained from multiple clinical sites was also availble. This model has been validated on an internal set of 15 CT scans, and on three external set of 55 CT scans in total.


## Installation
1) We recommend using conda to setup the environment, See the `requirements.yaml` for environment configuration 

    If there is no conda installed on your PC, please find the installers from https://www.anaconda.com/products/individual

    If you have already installed conda, please change the directory to the same level as `environment.yaml` and use the following command to setup environment quickly.

    > conda env create -f environment.yaml

    After environment creation, please use the following command to activate your environment
    
    > conda activate flcovid

2) Run the script `setup-env.sh` to compile Cython code first.  
 
    > bash setup-env.sh
    
3) Download the data(https://drive.google.com/file/d/1ZdGtxOHUgTwBXgBQFf5W4_Jiu7_QGMTY/view?usp=sharing) and put it at the same level of folder **'fl_covid'**. Please refer to the File structure at the bottom.

## Usage with examples
### Train
To start traning, please run the `train.sh`.
```
cd FL-COVID
bash train.sh
```
You can modify and run your own command according to the template in `train.sh`
```
python3 ./fl_covid/bin/train_fed.py --batch-size=6 --steps=100 --gpu=0 --epochs=30 --snapshot-path=../fl_covid_model_and_log/
```
- --batch-size
        
    Size of the training batches
- --steps 

    Number of steps per epoch
- --epochs
    
    Number of epochs to train
- --gpu
    
    Id of the GPU to use (as reported by nvidia-smi)
- --snapshot-path  

    Path to save the snapshots of models and training log during training

### Test
To run inference with our model, please run `test.sh`. Inference will be ran on external site1 by default.
```
cd FL-COVID
bash test.sh
```

To choose which site to be validated, please change `--site` in shell script. Options please refer to the below explanation .

You may specify the `--save-path` to change the results saving directory , default folder is named 'output' at the same directory level as the code folder.

```
# Run inference and saving the visual and statistical results
python3 ./fl_covid/bin/evaluate_overall_patient_wise.py --model=./model/model.h5  --gpu=0 --save-path=./output/ --save-result=1  --get_predicted_bbox --site=external1

# Print the statistical analysis
python3 ./fl_covid/bin/evaluate_overall_patient_wise.py --model=./model/model.h5  --gpu=0 --save-path=./output/ --save-result=0 --verbose --site=external1
```
- --model  

    Path to the model for evaluation
- --save-path  

    Path to save the visualization results and statistical intermediate results
- --save-result 

    ‘1’ for both saving visual and statistical results, '0' for only statistics.
- --get_predicted_bbox 

    To be used for first time evaluation, saving intermediate results for following statistics.
- --site 
    
    Determine which site to be evaluated. Options are [ internal | external1 | external2 | external3 ]


### Input and Output files
The system take input from csv files that saving the data information, each line represents a scan image with label and unique path.

Outputs are statistical results saved in npy format and visual results saved in **nii.gz** format under the directory **FL-COVID/output**

For visualizing the **nii.gz** results, we recommand using **ITK-SNAP** (http://www.itksnap.org/pmwiki/pmwiki.php?n=Downloads.SNAP3) to visualize them.

## File structure
For code and demo data, structures listed below:

    |--FL-COVID  
        |--fl_covid (source code folder)  
        |--data (demo date folder)  
            |--internal  
                |--internal_1  
                    |--h5  
                    |--h5_normalize  
                |--internal_2  
                    ...  
                |--internal_3  
                    ...  
                |--internal_test  
                    |--h5  
                    |--h5_normalize  
                    |--lung_seg_h5  
            |--external1  
                |--h5  
                |--h5_normalize  
                |--lung_seg_h5  
            |--external2  
                ...  
            |--external3  
                ...  
        |--LICENSE  
        |--README.md  
        |--setup-env.sh  
        |--train.sh  
        |--test.sh  
