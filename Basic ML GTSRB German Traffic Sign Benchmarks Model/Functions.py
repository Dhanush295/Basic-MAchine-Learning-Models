import tensorflow
from tensorflow.keras.layers import Conv2D, Input ,Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D, Flatten
from tensorflow.python.keras.layers.core import Flatten
from tensorflow.keras import Model


def streetsign_model(nbr_classes):
  
        my_inputs = Input(shape=(60,60,3))

        x = Conv2D(32, (3,3), activation='relu')(my_inputs)             #hyper parameters changes values to get better results
        x = MaxPool2D()(x)
        x = BatchNormalization()(x)

        x = Conv2D(64, (3,3), activation='relu')(x)
        x = MaxPool2D()(x)
        x = BatchNormalization()(x)

        x = Conv2D(128, (3,3), activation='relu')(x)
        x = MaxPool2D()(x)
        x = BatchNormalization()(x)

        x = Flatten()(x)
        # GlobalAvgPool2D()(x)
        x = Dense(68, activation='relu')(x)
        x= Dense(10, activation='softmax') (x)

        return Model(inputs=my_inputs, outputs=x)

if __name__=='__main__':

      model = streetsign_model(10)
      model.summary()