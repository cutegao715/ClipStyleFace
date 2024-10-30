import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from piq import fid, kid, inception_score, FID
from torchvision.models.inception import inception_v3
import torchvision.models as models
# from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.linalg import sqrtm

class ImageDataset(Dataset):
    def __init__(self, test_path):
        self.path = test_path
        self.images = test_path
        self.transform = transforms.Compose([
            transforms.Resize(299),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = self.transform(image)
        return {'images': image}

    def __len__(self):
        return len(self.images)


class ReferenceDataset(Dataset):
    def __init__(self, test_path):
        self.path = test_path
        self.images = os.listdir(self.path)
        self.transform = transforms.Compose([
            transforms.Resize(299),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.path, self.images[index]))
        image = self.transform(image)
        return {'images': image}

    def __len__(self):
        return len(self.images)

if __name__ == '__main__':
    dtype = torch.cuda.FloatTensor
    # styles = ['Anime Painting', 'Impressionism Painting', 'Mona Lisa Painting', 'Pixar', 'Edvard Munch', 'Cubism Painting', 'Sketch', 'Dali Painting',
    #           'Fernando Botero Painting', 'Ukiyo-e']
    styles = ['Anime Painting', 'Mona Lisa Painting', 'Pixar', 'Edvard Munch', 'Sketch', 'Dali Painting',
              'Fernando Botero Painting', 'Ukiyo-e']
    # styles = ['Anime Painting']

    ref_path = 'F:/datasets3/style_images/{}/'
    # ref_path = r'F:\datasets3\train_set'
    test_path = 'D:/experiments3/HyperDomainNet-main/results_cin_clip2/{}/'
    # test_path = 'F:/results/clipface/stylized_results/{}2/'
    # test_path = 'D:/experiments3/ClipFace-main/stylized_results3/{}/'
    # test_path = "D:/experiments3/HyperDomainNet-main/results_test/{}/"
    #
    fid_metric = FID()
    kid_metric = kid.KID()
    # resnet = InceptionResnetV1(pretrained='vggface2').eval()
    inception_model = inception_v3(pretrained=True, transform_input=True).type(dtype)
    # inception_model = models.inception_v3(pretrained=True)
    inception_model.eval()
    #
    all_IS_value = []
    all_FID_value = []
    all_KID_value = []
    # ref_path_style = 'F:/datasets3/ref_images'
    # refData = ReferenceDataset(test_path=ref_path_style)
    # refloader = DataLoader(refData, batch_size=4, shuffle=False, drop_last=False)
    for style in styles:
        ref_path_style = ref_path.format(style)
        # ref_path_style = ref_path
        refData = ReferenceDataset(test_path=ref_path_style)
        refloader = DataLoader(refData, batch_size=4, shuffle=False, drop_last=False)

        ref_features = []
        for j, batch in enumerate(refloader):
            data = batch['images']
            data = data.type(dtype)
            with torch.no_grad():
                output = inception_model(data)
                ref_features.append(inception_model(data))
        ref_features = torch.cat(ref_features, dim=0)

        test_path_style = test_path.format(style)
        object_names = os.listdir(test_path_style)
        object_IS_value = []
        object_FID_value = []
        object_KID_value = []
        object_list = []
        for object_name in object_names:
            object_path = os.path.join(test_path_style, object_name, '8')
            object_name = [os.path.join(object_path, o) for o in os.listdir(object_path)]
            object_name = [o for o in object_name if o.endswith('.jpg') and 'texture' not in o]
            object_list += object_name

        testData = ImageDataset(test_path=object_list)
        dataloader = DataLoader(testData, batch_size=4, shuffle=False, drop_last=False)
        features = []

        # load model
        for i, batch in enumerate(dataloader):
            data = batch['images']
            data = data.type(dtype)
            with torch.no_grad():
                features.append(inception_model(data))
        features = torch.cat(features, dim=0)

        # calculate metrics
        # IS metric
        IS_value = inception_score(features, num_splits=10)
        object_IS_value.append(IS_value[0])
        fid_score = fid_metric(features, ref_features)

        kid_score = kid_metric.compute_metric(features[:500], ref_features) + kid_metric.compute_metric(features[500:1000], ref_features)
        object_FID_value.append(fid_score.detach().cpu().numpy())
        object_KID_value.append(kid_score.detach().cpu().numpy())
        print("is_score of {}: {}".format(style, np.mean(object_IS_value)))
        print("fid_score of {}: {}".format(style, np.mean(object_FID_value)))
        print("kid_score of {}: {}".format(style, np.mean(object_KID_value)))
        all_IS_value.append(np.mean(object_IS_value))
        all_FID_value.append(np.mean(object_FID_value))
        all_KID_value.append(np.mean(object_KID_value))
    print("mean_is_score: {}".format(np.mean(all_IS_value)))
    print("mean_fid_score: {}".format(np.mean(all_FID_value)))
    print("mean_kid_score: {}".format(np.mean(all_KID_value)))

    ########### test #############
    # refData = ReferenceDataset(test_path='F:/datasets3/style_images/test2')
    # refloader = DataLoader(refData, batch_size=4, shuffle=False, drop_last=False)
    #
    # ref_features = []
    # for j, batch in enumerate(refloader):
    #     data = batch['images']
    #     data = data
    #     with torch.no_grad():
    #         ref_features.append(resnet(data))
    # ref_features = torch.cat(ref_features, dim=0)
    #
    # testData = ReferenceDataset(test_path='F:/datasets3/style_images/test1')
    # dataloader = DataLoader(testData, batch_size=4, shuffle=False, drop_last=False)
    #
    # features = []
    # for i, batch in enumerate(dataloader):
    #     data = batch['images']
    #     data = data
    #     with torch.no_grad():
    #         features.append(resnet(data))
    # features = torch.cat(features, dim=0)
    #
    # fid_score = fid_metric(features, ref_features)
    # print(features.shape)
    # print(fid_score)
    #
    # x_features = torch.rand(500, 512)
    # y_features = torch.rand(500, 512)
    #
    # if torch.cuda.is_available():
    #     # Move to GPU to make computaions faster
    #     x_features = x_features.cuda()
    #     y_features = y_features.cuda()
    #
    # # Use FID class to compute FID score from image features, pre-extracted from some feature extractor network
    # fid: torch.Tensor = FID()(x_features, y_features)
    # print(fid)

