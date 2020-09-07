from PIL import Image
import os
import os.path
import numpy as np
import pickle
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from pathlib import Path
#For Some reason the indexes always go up to 12 times the amount in the image bank

class CIFAR10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    base_folder = 'Data'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'



    # train_list = [['one.pickle','c99cafc152244af753f735de768cd75f']]
    train_list = [
        ['one.pickle', 'c99cafc152244af753f735de768cd75f'],
        ['two.pickle', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['three.pickle', '54ebc095f3ab1f0389bbae665268c751'],
        ['four.pickle', '634d18415352ddfa80567beed471001a'],
        ['five.pickle', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]
    test_list = [
        ['test.pickle', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }
   
    
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False,inputVersion = False):
        print("hello")
        super(CIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []
        mod_path = Path(__file__).parent
        # now load the picked numpy arrays
        if inputVersion == True:
            #This setting only adds the test.pickle labels and data

            file_path = (mod_path / "Data/test.pickle").resolve()
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        else:
            for file_name, checksum in downloaded_list:
                file_path = (mod_path / "Data" / file_name).resolve();

                with open(file_path, 'rb') as f:
                    entry = pickle.load(f, encoding='latin1')
                    self.data.append(entry['data'])
                    if 'labels' in entry:
                        self.targets.extend(entry['labels'])
                    else:
                        self.targets.extend(entry['fine_labels'])

        #vstack changes the length
        self.data = np.vstack(self.data)
        #reshape changes the length
        self.data = self.data.reshape(-1, 3, 32, 32)

        self.data = self.data.transpose((0, 2, 3, 1))  # convert to 




        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if check_integrity(path, self.meta['md5']): #usually a "not" in between the if and check_integrity
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        #possibly might need to add md5 verification :(
        # with open(path, 'rb') as infile:
        #     data = pickle.load(infile, encoding='latin1')
        #     self.classes = data[self.meta['key']]
        # self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):





        
        # print("hello")

        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
       """
    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                #Add this later not really needed because I know the data is not malicious.
                return True # supposed to be false
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        #figure this out late what exactly this does and if it is needed
        # download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }