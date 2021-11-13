import tensorflow as tf
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


class DataLoader:

    @staticmethod
    def load_data(data_config):
        captions_df = pd.read_csv(data_config.captions_file, names=[
                                  'ID', 'Captions'], header=0)
        captions_df['Path'] = data_config.image_path+captions_df['ID']
        all_img_path = captions_df['Path'].values.tolist()
        all_captions = captions_df['Captions'].values.tolist()
        img_dataset = tf.data.Dataset.from_tensor_slices(all_img_path)
        return all_img_path, all_captions, img_dataset

    @staticmethod
    def _map_func(img_path, caption):
        img_tensor = np.load(img_path.decode('utf-8')+'.npy')
        return img_tensor, caption

    @staticmethod
    def _gen_dataset(img_path, caption):
        dataset = tf.data.Dataset.from_tensor_slices((img_path, caption))

        # Use map to load the numpy files in parallel
        dataset = dataset.map(lambda item1, item2: tf.numpy_function(
            DataLoader._map_func, [item1, item2], [tf.float32, tf.int32]),
            num_parallel_calls=tf.data.AUTOTUNE)

        return dataset

    @staticmethod
    def split_data(all_img_path, cap_vector, batch_size, buffer_size):
        path_train, path_test, cap_train, cap_test = train_test_split(
            all_img_path, cap_vector, train_size=0.8, random_state=42)
        train = DataLoader._gen_dataset(path_train, cap_train)
        test = DataLoader._gen_dataset(path_test, cap_test)

        # Shuffle and batch
        train_dataset = train.shuffle(buffer_size).batch(batch_size)
        train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        test_dataset = test.batch(batch_size)

        return train_dataset, test_dataset
