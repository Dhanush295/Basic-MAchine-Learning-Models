import tensorflow
from tensorflow.keras.layers import Conv2D, Input ,Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D



# functional approach - function return model
def functional_model():

       my_Input = Input(shape=(28,28,1))
       x = Conv2D(32, (3,3), activation='relu')(my_Input)             #hyper parameters changes values to get better results
       x = Conv2D(64, (3,3), activation='relu')(x)
       x = MaxPool2D()(x)
       x = BatchNormalization()(x)

       x = Conv2D(128, (3,3), activation='relu')(x)
       x = MaxPool2D()(x)
       x = BatchNormalization()(x)

       x = GlobalAvgPool2D()(x)
       x = Dense(68, activation='relu')(x)
       x= Dense(10, activation='softmax') (x)

       model = tensorflow.keras.Model(inputs=my_Input, outputs=x)

       return model  


# low.keras.Model - inherit from this class
class Myclass(tensorflow.keras.Model):

    def __init__(self):
        super().__init__()
        self.conv1 =  Conv2D(32, (3,3), activation='relu')              #hyper parameters changes values to get better results
        self.conv2 = Conv2D(64, (3,3), activation='relu')
        self.maxpool1 = MaxPool2D()
        self.batchnorm2 =  BatchNormalization()

        self.conv3 =  Conv2D(128, (3,3), activation='relu')
        self.maxpool1 = MaxPool2D()
        self.batchnorm2 =  BatchNormalization()

        self.globalavgpool1 = GlobalAvgPool2D()
        self.dense1 =  Dense(68, activation='relu')
        self.dense2 = Dense(10, activation='softmax') 

    def call(self, my_inputs):
      x = self.conv1(my_inputs)
      x = self.conv2(x)
      x = self.maxpool1(x)
      x = self.batchnorm1(x)
      x = self.conv3(x)
      x = self.maxpool1(x)
      x = self.batchnorm2(x)
      x = self.globalavgpool1(x)
      x = self.dense1(x)
      x = self.dense2(x)

      return x