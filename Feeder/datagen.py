from .VideoDataset import VideoDataset
from torch.utils.data import Dataset, DataLoader
from .split import split
from Feeder.transform import train_transform

def datagen(DATASETPATH_Spatial, 
            DATASETPATH_OpticalFlow,
            MODE,
            num_sec_frame = 10,
            ratio = 0.8):

        if MODE == "Spatail" :
                X_trainSpatial, X_testSpatial, y_train, y_test = split(DATASETPATH_Spatial, DATASETPATH_OpticalFlow, 
                                                                 mode = MODE, ratio = ratio)

                train_dataset = VideoDataset(data_spatial = X_trainSpatial,
                                        data_optical = None,
                                        labels       = y_train,
                                        transform    = train_transform,
                                        num_sec_frame  = num_sec_frame,             # num_frames 5, 10, 15, 20
                                        mode = MODE)    

                test_dataset  = VideoDataset(data_spatial = X_testSpatial,
                                        data_optical = None,
                                        labels       = y_test,
                                        transform    = train_transform,
                                        num_sec_frame  = num_sec_frame,             # num_frames5, 10, 15, 20
                                        mode = MODE)     
        
        if MODE == "OpticalFlow" :
                X_trainOptical, X_testOptical, y_train, y_test = split(DATASETPATH_Spatial, DATASETPATH_OpticalFlow, 
                                                                        mode = MODE, ratio = ratio)

                train_dataset = VideoDataset(data_spatial = None,
                                        data_optical = X_trainOptical,
                                        labels       = y_train,
                                        transform    = train_transform,
                                        num_sec_frame  = num_sec_frame,             # num_frames 5, 10, 15, 20
                                        mode = MODE)    

                test_dataset  = VideoDataset(data_spatial = None,
                                        data_optical = X_testOptical,
                                        labels       = y_test,
                                        transform    = train_transform,
                                        num_sec_frame  = num_sec_frame,             # num_frames5, 10, 15, 20
                                        mode = MODE)    
        
        if MODE == "IntermediateFusion" :
                (X_trainSpatial, X_trainOptical), (X_testSpatial, X_testOptical), y_train, y_test = split(DATASETPATH_Spatial, DATASETPATH_OpticalFlow, 
                                                                                                mode = MODE, ratio = ratio)

                train_dataset = VideoDataset(data_spatial = X_trainSpatial,
                                        data_optical = X_trainOptical,
                                        labels       = y_train,
                                        transform    = train_transform,
                                        num_sec_frame  = num_sec_frame,             # num_frames 5, 10, 15, 20
                                        mode = MODE)    

                test_dataset  = VideoDataset(data_spatial = X_testSpatial,
                                        data_optical = X_testOptical,
                                        labels       = y_test,
                                        transform    = train_transform,
                                        num_sec_frame  = num_sec_frame,             # num_frames5, 10, 15, 20
                                        mode = MODE)     

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers = 32)
        test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False, num_workers = 32)
        
        return train_loader, test_loader
