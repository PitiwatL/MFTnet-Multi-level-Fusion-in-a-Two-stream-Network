# Test
The code is during in preparation....
Optical Flow Videos <br><br>
<div align="center">  
  <img src = "./assets/rgb_makeup.gif" width = 250>
  <img src = "./assets/makeup_opt.gif" width = 250>
  <img src = "./assets/makeup_inverted_opt.gif" width = 250>
</p>

Separate Stream
| Model | Result (%) | 
|:-------------------------------:|:--------:|
| Spatial Stream  (VGG16 + LSTM)         | 88.51 | 
| Temporal Stream (DenseNet121 + LSTM)   |  -  |  

Combined Stream S: (VGG16 + LSTM) + T: (DenseNet121 + LSTM)
| Late Fusion Methods | Result (%) | 
|:-----------:|:--------:|
| Averaged Sum | - | 
| Ridge Regression |  |
| Multinomial Naive Bayes |  |
| Majority Voting for these 3 Late fusion models | |
