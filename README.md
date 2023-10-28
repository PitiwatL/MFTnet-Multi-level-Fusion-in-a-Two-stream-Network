# MFTnet
The explanation is during in preparation....
Optical Flow Videos <br><br>
<p align="center">  
  <img src = "./assets/rgb_makeup.gif" width = 250>
  <img src = "./assets/makeup_opt.gif" width = 250>
  <img src = "./assets/makeup_inverted_opt.gif" width = 250>
</p>
  
## Overall Architecture of the MFTnet
<p align="center"> 
  <img src = "./assets/OverallNet.jpg" width = 600>
</p>

## Results
Separate Stream
<p align="center"> 
  
|   Model   | Accuracy (%) | 
|:---------:|:------------:|
| Spatial Stream  (VGG16 + LSTM)         | 88.51  | 
| Temporal Stream (DenseNet121 + LSTM)   | 87.27  |  
</p>

Combined Stream S: (VGG16 + LSTM) + T: (DenseNet121 + LSTM)
| Late Fusion Methods | Accuracy (%) | 
|:-----------:|:--------:|
| Averaged Sum | - | 
| Ridge Regression |  |
| Multinomial Naive Bayes |  |
| Majority Voting for these 3 Late fusion models |  |

Intermediate Fusion (Fusion inside Model) ()
| Intermediate Fusion Methods | Accuracy (%)  | 
|:---------------------------:|-------------|
| Sum Fusion                  |  -  |       | 
| Max Fusion                  |     |       |
| Concatenation Fusion        |     |       |
| Convolution Fusion          |   88.69     |

