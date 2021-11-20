import os
import glob
from posixpath import basename
from sklearn.model_selection import train_test_split
import shutil
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from util import create_generators, split_data, order_test_set

from Functions import streetsign_model

if __name__=="__main__":
    
  path_to_save_train = "B:\\MlProjects\\GTSRB\\archive\\training_data\\train"
  path_to_save_val = "B:\\MlProjects\\GTSRB\\archive\\training_data\\val"
  path_to_test = "B:\\MlProjects\\GTSRB\\archive\\Test"
  batch_size = 64
  epochs = 15


  train_generator, val_generator, test_generator =  create_generators(batch_size, path_to_save_train, path_to_save_val , path_to_test)
  nbr_classes = train_generator.num_classes

 
  path_to_save_model = './archive'
  ckpt_saver = ModelCheckpoint(
            path_to_save_model,
            monitor="val_accuracy",
            mode='max',
            save_best_only=True,
            save_freq='epochs',
            verbose=1
   )

  early_stop = EarlyStopping(monitor="val_accuracy", patience=10)

  model = streetsign_model(nbr_classes)

  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  model.fit(train_generator, 
                 epochs=epochs,
                 batch_size=batch_size,
                 validation_data=val_generator,      
                 callbacks = [ckpt_saver, early_stop]
                 )