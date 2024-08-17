import os
import random
import shutil


class Dataset:
    def __init__(self, folder_path, build=False, test_size=0.2):
        """
        Dataset sınıfı başlatıcı metodu.

        Args:
            folder_path (str): Veri kümesinin bulunduğu ana klasörün yolu.
            build (bool, optional): Yeni bir veri kümesi oluşturulup oluşturulmayacağını belirten bayrak.
                                    Varsayılan olarak False.
            test_size (float, optional): Test veri kümesinin oranı. Varsayılan olarak 0.2 (yüzde 20).
        """
        self._classes = []
        self._train = []
        self._test = []
        self._validation = []
        self._folder_path = folder_path
        self._build = build
        if build:
            self._build_dataset(test_size)
        elif not build:
            self._get_dataset()

    @property
    def classes(self):
        """
        Veri kümesindeki sınıfları döndüren özellik (property).
        """
        return self._classes

    @classes.setter
    def classes(self, value):
        """
        Veri kümesindeki sınıfları ayarlayan özellik (property) setter metodu.
        """
        self._classes = value

    @property
    def train(self):
        """
        Eğitim veri kümesini döndüren özellik (property).
        """
        return self._train

    @train.setter
    def train(self, value):
        """
        Eğitim veri kümesini ayarlayan özellik (property) setter metodu.
        """
        self._train = value

    @property
    def test(self):
        """
        Test veri kümesini döndüren özellik (property).
        """
        return self._test

    @test.setter
    def test(self, value):
        """
        Test veri kümesini ayarlayan özellik (property) setter metodu.
        """
        self._test = value

    @property
    def validation(self):
        """
        Doğrulama veri kümesini döndüren özellik (property).
        """
        return self._validation

    @validation.setter
    def validation(self, value):
        """
        Doğrulama veri kümesini ayarlayan özellik (property) setter metodu.
        """
        self._validation = value

    @property
    def folder_path(self):
        """
        Veri klasörünün yolu özelliğini döndüren özellik (property).
        """
        return self._folder_path

    @folder_path.setter
    def folder_path(self, value):
        """
        Veri klasörünün yolu özelliğini ayarlayan özellik (property) setter metodu.
        """
        self._folder_path = value

    @property
    def build(self):
        """
        Yeni bir veri kümesi oluşturulup oluşturulmayacağını belirten özellik (property).
        """
        return self._build

    @build.setter
    def build(self, value):
        """
        Yeni bir veri kümesi oluşturulup oluşturulmayacağını ayarlayan özellik (property) setter metodu.
        """
        self._build = value

    def _get_dataset(self):
        """
        Mevcut veri kümesini yükleyen metod.
        """
        datasets = [dataset_name for dataset_name in os.scandir(self.folder_path) if dataset_name.is_dir()]
        for dataset in datasets:
            classes = [class_name for class_name in os.scandir(dataset.path) if class_name.is_dir()]
            for class_dir in classes:
                if class_dir.name not in self.classes:
                    self.classes.append(class_dir.name)
                images = [image_name for image_name in os.listdir(class_dir.path)]
                for image in images:
                    if dataset.name == "train":
                        self._train.append(os.path.join(f"{class_dir.name}", image))
                    elif dataset.name == "test":
                        self._test.append(os.path.join(f"{class_dir.name}", image))
                    elif dataset.name == "val":
                        self._validation.append(os.path.join(f"{class_dir.name}", image))
        self._classes = sorted(self.classes)

    def _build_dataset(self, test_size):
        """
        Yeni bir veri kümesi oluşturan metod.
        """
        classes = [class_name for class_name in os.scandir(self.folder_path) if class_name.is_dir()]

        for class_dir in classes:
            class_name = class_dir.name
            if class_dir.name not in self.classes:
                self.classes.append(class_dir.name)
            class_path = class_dir.path

            images = [image_name for image_name in os.listdir(class_path)]
            random.shuffle(images)
            test_images = images[-int(len(images) * test_size):]
            train_images = images[:int(len(images) * (1 - test_size))]

            for image in train_images:
                self.train.append(os.path.join(class_name, image))
            for image in test_images:
                self.test.append(os.path.join(class_name, image))

    def save_dataset(self, output_folder):
        """
        Veri kümesini belirtilen bir klasöre kaydeden metod.
        """
        for class_name in self._classes:
            os.makedirs(os.path.join(output_folder, "train", class_name), exist_ok=True)
            os.makedirs(os.path.join(output_folder, "test", class_name), exist_ok=True)
            os.makedirs(os.path.join(output_folder, "val", class_name), exist_ok=True)
        if self.build:
            for image in self.train:
                shutil.copyfile(os.path.join(self.folder_path, image), os.path.join(output_folder, "train", image))
            for image in self.test:
                shutil.copyfile(os.path.join(self.folder_path, image), os.path.join(output_folder, "test", image))
            for image in self.validation:
                shutil.copyfile(os.path.join(self.folder_path, image), os.path.join(output_folder, "val", image))
        elif not self.build:
            for image in self.train:
                shutil.copyfile(os.path.join(self.folder_path, "train", image),
                                os.path.join(output_folder, "train", image))
            for image in self.test:
                shutil.copyfile(os.path.join(self.folder_path, "test", image),
                                os.path.join(output_folder, "test", image))
            for image in self.validation:
                shutil.copyfile(os.path.join(self.folder_path, "val", image),
                                os.path.join(output_folder, "val", image))
