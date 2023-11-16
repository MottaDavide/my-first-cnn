# checking for GPU
import torch
use_cuda = torch.cuda.is_available()

print('torch version', torch.__version__)

if use_cuda:
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__CUDA Device Name:',torch.cuda.get_device_name(0))
    print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)

else:
    print('No GPU available')

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# LET'S BEGIN
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PACKAGES
from tensorflow.keras.datasets import mnist
import pandas as pd
from dataclasses import dataclass


# ---------------------------------------------------------------------
@dataclass
class Dataset:
    to_pandas: bool = True #if True converts to a single pd.DataFrame object
    train_test: str = 'train' #'train' -> import only train; 'test' -> import only test; 'all' -> import both train and test


    def import_train(self):
        # importing mnist dataset
        (X_train, y_train), _ = mnist.load_data()

        if self.to_pandas:
            n_pixels = X_train.shape[1] * X_train.shape[2]
            n_rows = X_train.shape[0]

            # convert X_train to pandas.DataFrame
            X_train = X_train.reshape(n_rows,n_pixels)
            columns = [f'pixel{i + 1}' for i in range(n_pixels)]
            dictionary_x_train= {columns[i] : X_train.T[i] for i in range(n_pixels)}
            X_train = pd.DataFrame(dictionary_x_train)

            # convert y_train to pandas.Series
            y_train = pd.DataFrame({'label': y_train})

            # merging
            df_mnist = pd.merge(X_train, y_train, left_index=True, right_index=True)

            return df_mnist
        return (X_train, y_train)
    

    def import_test(self):
        # importing mnist dataset
        _, (X_test, y_test)= mnist.load_data()

        if self.to_pandas:
            n_pixels = X_test.shape[1] * X_test.shape[2]
            n_rows = X_test.shape[0]

            # convert X_test to pandas.DataFrame
            X_test = X_test.reshape(n_rows,n_pixels)
            columns = [f'pixel{i + 1}' for i in range(n_pixels)]
            dictionary_x_test= {columns[i] : X_test.T[i] for i in range(n_pixels)}
            X_test = pd.DataFrame(dictionary_x_test)

            # convert y_test to pandas.Series
            y_test = pd.DataFrame({'label': y_test})

            # merging
            df_mnist = pd.merge(X_test, y_test, left_index=True, right_index=True)

            return df_mnist
        return (X_test, y_test)
    
if __name__ == '__main__':
    data = Dataset()
    

