'''
mutli object
multi prompt
multi interation
'''

from tqdm import tqdm
import numpy as np
from PIL import Image
import os
import torch
import clip
import torchvision.transforms as transforms

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


class SimilarityCalculator:
    def __init__(self):
        self.device = self._get_device()
        self.model, self.preprocess = self._initialize_model("ViT-B/32", self.device)
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=0)
        self.image_1 = None
        self.image_2 = None
        self.raw_similarity = None

    def _get_device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _initialize_model(self, model_name="ViT-B/32", device="cpu"):
        model, preprocess = clip.load(model_name, device=device)
        return model, preprocess

    @torch.no_grad()
    def _embed_image(self, image_path):
        preprocessed_image = (
            self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        )
        image_embeddings = self.model.encode_image(preprocessed_image)
        image_embeddings /= image_embeddings.clone().norm(dim=-1, keepdim=True)
        return image_embeddings

    @torch.no_grad()
    def _embed_text(self, text):
        tokens = clip.tokenize(text).to(self.device)
        text_features = self.model.encode_text(tokens).detach()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def calculate_similarity(self, image_path_1, text):
        self.image = self._embed_image(image_path_1)
        self.text = self._embed_text(text)
        self.raw_similarity = self.cosine_similarity(
            self.image[0], self.text[0]
        ).item()
        return self.raw_similarity

if __name__ == '__main__':
    # styles = ['Anime Painting', 'Impressionism Painting', 'Mona Lisa Painting', 'Pixar', 'Edvard Munch', 'Cubism Painting', 'Sketch', 'Dali Painting',
    #           'Fernando Botero Painting', 'Ukiyo-e']
    styles = ['Anime Painting', 'Mona Lisa Painting', 'Pixar', 'Edvard Munch', 'Sketch', 'Dali Painting',
              'Fernando Botero Painting', 'Ukiyo-e']
    # styles = ['Anime Painting']
    prompt = "A Photo of a Face in the Style of {}"
    save_path = "D:/experiments3/HyperDomainNet-main/results_cin_clip2/{}/"
    # save_path = 'D:/experiments3/ClipFace-main/stylized_results3/{}/'
    # save_path = "F:/results/clipface/stylized_results/{}2/"
    all_style_scores = []
    sim_calculator = SimilarityCalculator()
    for style in styles:
        _save_path = save_path.format(style)
        _prompt = prompt.format(style)
        object_names = os.listdir(_save_path)
        threshold = 0.22

        # clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s34b_b88k', device=device)
        # tokenizer = open_clip.get_tokenizer('ViT-g-14')


        # clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        #     'hf-hub:laion/CLIP-ViT-g-14-laion2B-s34B-b88K')
        # tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-g-14-laion2B-s34B-b88K')

        # prompt_token = tokenizer([_prompt]).to(device)
        # with torch.no_grad():
        #     encoded_prompt = clip_model.encode_text(prompt_token)
        ##### example ############
        all_score = []
        for object_name in object_names:
            object_path = os.path.join(_save_path, object_name, '6')
            view_num = os.listdir(object_path)
            all_views = []
            scores = []
            for view in view_num:
                if view.endswith("jpg") and 'texture' not in view:
                    view_path = os.path.join(object_path, view)
                    encoded_views = sim_calculator._embed_image(view_path)
                    encoded_prompt = sim_calculator._embed_text(_prompt)
                    score = 100 * encoded_views @ encoded_prompt.T
                    score = score.mean()
                    scores.append(score.item())
            all_score.append(np.mean(scores))
        # np.save("all_score_{}.npy".format(style), all_score)
        print("final_MSE_{}, {}".format(style,np.mean(all_score)))
        all_style_scores.append(np.mean(all_score))
    print('total_MSE', np.mean(all_style_scores))