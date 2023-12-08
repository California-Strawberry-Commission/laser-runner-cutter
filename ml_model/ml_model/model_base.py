from abc import abstractmethod


class ModelBase:
    @abstractmethod
    def name():
        pass

    @abstractmethod
    def load_weights():
        pass

    @abstractmethod
    def model_train(self, img_dir, label_dir, weights_path, **kwargs):
        pass

    @abstractmethod
    def get_map_value():
        pass
