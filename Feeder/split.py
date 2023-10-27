import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from sklearn.model_selection import train_test_split


def split(DATASETPATH_Spatial, DATASETPATH_OpticalFlow, mode = 'Spatial', ratio = 0.8) :
    
    Class =  ['BasketballDunk', 'HighJump',   'TennisSwing', 'FloorGymnastics', 'PlayingGuitar', 'TaiChi', 'Billiards', 'FrontCrawl', 'TrampolineJumping', 'Bowling', 
    'BodyWeightSquats', 'PlayingViolin', 'IceDancing', 'WallPushups', 'Punch', 'Surfing', 'TableTennisShot', 'BandMarching', 'StillRings', 'WritingOnBoard', 
    'Nunchucks',      'ThrowDiscus',   'Swing',     'ParallelBars', 'Lunges', 'BenchPress', 'SkyDiving', 'UnevenBars', 'CliffDiving', 'HammerThrow', 
    'HandstandPushups','Kayaking', 'BoxingSpeedBag', 'HorseRace', 'RopeClimbing', 'MoppingFloor', 'JavelinThrow', 'CricketBowling', 'SoccerPenalty', 'VolleyballSpiking', 
    'PlayingFlute',  'HeadMassage',   'PushUps', 'PoleVault', 'BrushingTeeth', 'HorseRiding', 'JumpRope', 'WalkingWithDog', 'LongJump', 'ApplyLipstick', 
    'CuttingInKitchen', 'Shotput', 'PlayingDhol', 'CricketShot', 'Hammering',  'Basketball',   'SoccerJuggling', 'BoxingPunchingBag', 'Mixing', 'Haircut', 
    'BlowingCandles', 'PizzaTossing', 'MilitaryParade', 'BreastStroke', 'BalanceBeam', 'ApplyEyeMakeup', 'Drumming', 'PlayingCello', 'FieldHockeyPenalty', 'JumpingJack', 
    'Biking', 'SalsaSpin', 'BabyCrawling', 'FrisbeeCatch', 'Diving', 'Fencing', 'Rowing', 'ShavingBeard',   'BaseballPitch', 'Skiing', 
    'Typing', 'Skijet', 'SumoWrestling', 'JugglingBalls', 'Rafting', 'SkateBoarding', 'YoYo', 'PullUps', 'HulaHoop', 'Archery', 
    
    'PommelHorse', 'PlayingSitar', 'PlayingPiano',   'HandstandWalking', 'CleanAndJerk', 'PlayingDaf', 'BlowDryHair', 'Knitting', 'PlayingTabla', 'RockClimbingIndoor', 'GolfSwing']
    
    DatasetSpatial = []
    DatasetOptical = []
    Labels  = []
    X_trainSpatial, y_train = [], []
    X_testSpatial, y_test  = [], []
    X_trainOptical, X_testOptical = [], [] 
    X_val, y_val   = [], []
    
    if mode == 'Spatial':
        Path = DATASETPATH_Spatial + '/'
        for clss in os.listdir(DATASETPATH_Spatial) :
            for vid in os.listdir(Path + clss):
                DatasetSpatial.append(Path + clss + "/" + vid)
                Labels.append(Class.index(clss))

        X_trainSpatial, X_testSpatial, y_train, y_test = train_test_split(DatasetSpatial, Labels,  
                                                         train_size = ratio, random_state=123)
        
        return X_trainSpatial, X_testSpatial, y_train, y_test
    
    if mode == 'OpticalFlow':
        Path = DATASETPATH_OpticalFlow + '/'
        for clss in os.listdir(DATASETPATH_OpticalFlow) :
            for vid in os.listdir(Path + clss):
                DatasetOptical.append(Path + clss + "/" + vid)
                Labels.append(Class.index(clss))

        X_trainOptical, X_testOptical, y_train, y_test = train_test_split(DatasetOptical, Labels,  
                                                           train_size = ratio, random_state=123)
        
        return X_trainOptical, X_testOptical, y_train, y_test
    
    if mode == 'IntermediateFusion':
        PathSpatial     = DATASETPATH_Spatial + '/'
        PathOpticalFlow = DATASETPATH_OpticalFlow + '/'
        
        for clss in os.listdir(DATASETPATH_Spatial) :
            for vid in os.listdir(PathSpatial + clss):
                DatasetSpatial.append(PathSpatial + clss + "/" + vid)
                Labels.append(Class.index(clss))

        X_trainSpatial, X_testSpatial, y_train1, y_test1 = train_test_split(DatasetSpatial, Labels,  
                                                             train_size = ratio, random_state=123)
        
        for clss in os.listdir(DATASETPATH_OpticalFlow) :
            for vid in os.listdir(PathOpticalFlow + clss):
                DatasetOptical.append(PathOpticalFlow + clss + "/" + vid)

        X_trainOptical, X_testOptical, y_train2, y_test2 = train_test_split(DatasetOptical, Labels,  
                                                             train_size = ratio, random_state=123)
        
        return (X_trainSpatial, X_trainOptical), (X_testSpatial, X_testOptical), y_train1, y_test1
        
    
    print('Num Dataset: ', len(DatasetSpatial))
    print('Num train: ', len(y_train))
    # print('Num val: ', len(y_val))
    print('Num test: ', len(y_test))