from generater_animegan import AnimeGenerator
from paddle.io import Dataset
from PIL import Image
import paddle
import numpy as np
import pickle, six, os

from paddle.vision.transforms import Compose, Resize, Transpose, Normalize

device = 'gpu'  # or cpu


class Compose(object):
    """
    Composes several transforms together use for composing list of transforms
    together for a dataset transform.

    Args:
        functions (list[callable]): List of functions to compose.

    Returns:
        A compose object which is callable, __call__ for this Compose
        object will call each given :attr:`transforms` sequencely.

    """

    def __init__(self, functions):
        self.functions = functions

    def __call__(self, datas):

        for func in self.functions:
            try:
                datas = func(datas)
            except Exception as e:

                print("fail to perform fuction [{}] with error: "
                      "{} and stack:\n{}".format(func, e, str("XSSD")))
                raise RuntimeError
        return datas


class MyDataset(Dataset):
    def __init__(self, dataroot):
        super().__init__()
        self.file_list = os.listdir(dataroot)
        self.dataroot = dataroot
        self.transform = Compose([
            Resize((256, 256), interpolation="bicubic"),
            Transpose(),
            Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], keys=['image', 'image'])
        ])

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        filepath = os.path.join(self.dataroot, filename)
        image = paddle.vision.transforms.functional.pil_loader(filepath)
        image = self.transform({'image': image})
        return image

    def __len__(self):
        return len(self.file_list)


def tensor2img(input_image, min_max=(-1., 1.), image_num=1, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor): the input image tensor array
        image_num (int): the convert iamge numbers
        imtype (type): the desired type of the converted numpy array
    """

    def processing(img, transpose=True):
        """"processing one numpy image.

        Parameters:
            im (tensor): the input image numpy array
        """
        # grayscale to RGB
        if img.shape[0] == 1:
            img = np.tile(img, (3, 1, 1))
        img = img.clip(min_max[0], min_max[1])
        img = (img - min_max[0]) / (min_max[1] - min_max[0])
        if imtype == np.uint8:
            # scaling
            img = img * 255.0
        # tranpose
        img = np.transpose(img, (1, 2, 0)) if transpose else img
        return img

    if not isinstance(input_image, np.ndarray):
        # convert it into a numpy array
        image_numpy = input_image.numpy()
        ndim = image_numpy.ndim
        if ndim == 4:
            image_numpy = image_numpy[0:image_num]
        elif ndim == 3:
            # NOTE for eval mode, need add dim
            image_numpy = np.expand_dims(image_numpy, 0)
            image_num = 1
        else:
            raise ValueError(
                "Image numpy ndim is {} not 3 or 4, Please check data".format(
                    ndim))

        if image_num == 1:
            # for one image, log HWC image
            image_numpy = processing(image_numpy[0])
        else:
            # for more image, log NCHW image
            image_numpy = np.stack(
                [processing(im, transpose=False) for im in image_numpy])

    else:
        # if it is a numpy array, do nothing
        image_numpy = input_image
    image_numpy = image_numpy.round()
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def load(file_name):
    with open(file_name, 'rb') as f:
        state_dicts = pickle.load(f) if six.PY2 else pickle.load(
            f, encoding='latin1')
    return state_dicts


def Load_Para(model, weight_path):
    '''
    copy from https://github.com/PaddlePaddle/PaddleGAN/blob/develop/ppgan/engine/trainer.py
    Args:
        model: AnimeGenerator
        weight_path: Trained weight
    Returns:None
    '''
    state_dicts = load(weight_path)

    # Only the weights of the generator are loaded
    net_name = "netG"
    if net_name in state_dicts:
        model.set_state_dict(state_dicts[net_name])
        print(f"Loaded pretrained weight for net {net_name}")
    else:
        print("Loaded pretrained weight ERROR")


if __name__ == '__main__':

    model = AnimeGenerator().to(device=device)
    Load_Para(model, weight_path='E:\Python_Project\paddle_Lab\PaddleGAN\output_dir\epoch_42_weight.pdparams')
    model.eval()

    # dataset preprocess
    transform = Compose([
        Transpose(),
        Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], keys=['image', 'image'])
    ])

    # using DatasetFolder for loading
    dataroot = r'./test_img'
    test_dataset = paddle.vision.datasets.DatasetFolder(
        dataroot,
        transform=transform, extensions=('jpg', 'jpeg', 'png', 'gif'))

    test_dataloader = paddle.io.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    for i, data in enumerate(test_dataloader):
        print("processing img {}".format(i + 1))
        with paddle.no_grad():
            out1 = model.forward(data[0])
        image_numpy = tensor2img(out1, min_max=(-1.0, 1.0), image_num=1)
        save_image(image_numpy, "./output/{}.jpg".format(i))
