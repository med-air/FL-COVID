# Implementation of our manuscript
"Federated deep learning for detecting COVID-19 lung abnormalities in CT: A privacy-preserving multinational validation study" by

Qi Dou, Tiffany So, Meirui Jiang, Quande Liu, VARUT Vardhanabhuti, Georgios Kaissis, Zeju Li, Weixin Si, Heather Lee, Kevin Yu, Zuxin Feng, Li Dong, Egon Burian, Friederike Jungmann, Rickmer Braren, Prof. Marcus Makowski, Bernhard Kainz, Daniel Rueckert, Ben Glocker, Simon Yu, Pheng Ann Heng

License at: `LICENSE`

For any inquiry, please contact Dr. Qi Dou (qdou@cse.cuhk.edu.hk). 


## Installation
1) We recommend using conda to setup the environment, See the `requirements.yaml` for environment configuration 

    If there is no conda installed on your PC, please find the installers from https://www.anaconda.com/products/individual

    If you have already installed conda, please use the following command to setup environment quickly.

    > conda env create -f environment.yaml

    After environment creation, please use the following command to activate your environment
    
    > conda activate flcovid

2) Run the script `setup-env.sh` to compile Cython code first.  
 
    > bash setup-env.sh
    
3) Download the data and put it at the same level of folder **'fl_covid'**. Please refer to the File structure at the bottom.

## Usage   
### Train
Please run the `train.sh` to start traning.
You may specify `--gpu` to determine which one to use, default is '0'.


### Test
Please run the `test.sh` to start testing. 

To choose which site to be validated, please change the `--site`. Options for `--site` are [ internal | external1 | external2 | external3 ]

You may specify the `--save-path` to change the results saving directory , default folder is named 'output' at the same directory level as the code folder.

Here are the arguments used in the bash script.
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
- --model  

    Path to the model for evaluation
- --save-path  

    Path to save the visualization results and statistical intermediate results
- --save-result 

    ‘1’ for both saving visual and statistical results, '0' for only statistics.
- --get_predicted_bbox 

    To be used for first time evaluation, saving intermediate results for following statistics.
- --site 
    
    Determine which site to be evaluated.


### Visualize results
Visual results are saved in **'nii.gz'** format in directory **'output/visual_results'**
We recommand using **ITK-SNAP** (http://www.itksnap.org/pmwiki/pmwiki.php?n=Downloads.SNAP3) to visualize them.

## File structure
For code and demo data, structures listed below:

    |--project  
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