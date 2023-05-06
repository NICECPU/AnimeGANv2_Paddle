from generater_animegan import AnimeGenerator
from PIL import Image
import paddle
import numpy as np
import pickle, six, os, cv2
from paddle.vision.transforms import Compose,Transpose, Normalize

device = 'gpu'  # or cpu

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


class MyDataLoader(object):
    def __init__(self, dataroot, batch_size=1,shuffle=False):
        self.dataroot = dataroot
        self.file_list = os.listdir(dataroot)
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.transform = Compose([
            Transpose(),
            Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], keys=['image', 'image'])
        ])

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        filepath = os.path.join(self.dataroot, filename)

        image = cv2.imread(filepath)
        image = self.transform(image)
        # image = paddle.vision.transforms.functional.pil_loader(filepath)
        # image = self.transform(image)

        return filename, paddle.to_tensor(image)

    def __len__(self):
        return len(self.file_list)


if __name__ == '__main__':

    model = AnimeGenerator().to(device=device)
    Load_Para(model, weight_path='epoch_42_weight.pdparams')
    model.eval()

    # dataset preprocess
    transform = Compose([
        Transpose(),
        Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], keys=['image', 'image'])
    ])
    test_dataloader = MyDataLoader("./test_img/img", batch_size=1, shuffle=False)
    for i, filename_data in enumerate(test_dataloader):
        filename, data = filename_data
        print("{} :processing --->{}".format(i, filename))
        with paddle.no_grad():
            data = data.unsqueeze(0)
            out1 = model.forward(data)
        image_numpy = tensor2img(out1, min_max=(-1.0, 1.0), image_num=1)
        save_image(image_numpy, "./output/{}-output.jpg".format(filename))
