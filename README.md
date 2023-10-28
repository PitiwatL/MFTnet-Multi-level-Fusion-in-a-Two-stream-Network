# MFTnet
The explanation is during in preparation....
Optical Flow Videos <br><br>
<p align="center">  
  <img src = "./assets/rgb_makeup.gif" width = 250>
  <img src = "./assets/makeup_opt.gif" width = 250>
  <img src = "./assets/makeup_inverted_opt.gif" width = 250>
</p>
  
## Overall Architecture of the MFTnet
The late fusion models consist of 3 methods, which are averaged sum, ridge regression, and multinomial naive bayes.

<p align="center"> 
  <img src = "./assets/OverallNet.jpg" width = 600>
</p>

## Evaluation
Should prepare softmax scores in this formation before performing late fusion:
```Shell
├── SoftMaxScores
    ├── Spatial
        ├── train
        ├── test
    ├── OpticalFlow
        ├── train
        ├── test
    ├── IntermediateFusion
        ├── train
        ├── test
```
Separate Stream
<p align="center"> 
  
|                 Model                   |  UCF101 (%)  | NTU-RGB 60 (%)| 
|:---------------------------------------:|:------------:|:-------------:|
| Spatial Stream  (VGG16 + LSTM)          |    88.51     |    72.04      |
| Temporal Stream (DenseNet121 + LSTM)    |    87.27     |    83.30      |
</p>

Combined Stream S: (VGG16 + LSTM) + T: (DenseNet121 + LSTM)
| Late Fusion Methods | UCF101 (%) | NTU-RGB 60 (%)| 
|:-------------------:|:--------:|:--------:|
|     Averaged Sum        |   -  |   86.15  |
|    Ridge Regression     |      |   86.17  |
| Multinomial Naive Bayes |      |   86.10  |
| Majority Voting for these 3 Late fusion models |  |  |

Intermediate Fusion (Fusion inside Model) ()
| Intermediate Fusion Methods | Accuracy (%)  | NTU-RGB 60 (%)| 
|:---------------------------:|:-------------:|:-------------:|
| Sum Fusion                  |  -            |     75.74    | 
| Max Fusion                  |               |     73.93    |
| Concatenation Fusion        |               |     74.85    |
| Convolution Fusion          |    88.69      |     77.48    |

Our MFTnet 
| Intermediate Fusion Methods | Accuracy (%)  | NTU-RGB 60 (%)| 
|:---------------------------:|:-------------:|:-------------:|
| Sum Fusion                  |    -          |         | 
| Max Fusion                  |               |         |
| Concatenation Fusion        |               |         |
| Convolution Fusion          |    -          |         |
