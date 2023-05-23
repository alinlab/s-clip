import torch
import torch.nn as nn


def create_custom_model(args, model):
    return CustomCLIP(model)


class CustomCLIP(nn.Module):
    override = ["clip", "forward", "lock_text_tower"]

    def __init__(self, clip):
        super().__init__()
        self.clip = clip

    def __getattr__(self, name):
        if name in self.override:
            return super().__getattr__(name)
        else:
            return getattr(self.clip, name)

    def lock_text_tower(self, unlocked_layers, freeze_layer_norm):
        # ignore options and just lock the entire text tower
        for param in self.clip.transformer.parameters():
            param.requires_grad = False

    def forward(self, image, text, query=None, keyword=None):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        out = {
            "image_features": image_features,
            "text_features": text_features,
            "logit_scale": self.logit_scale.exp()
        }

        if query is not None:  # unlabeled image
            query_features = self.encode_image(query, normalize=True)
            out.update({
                "query_features": query_features,
            })

        if keyword is not None:  # keyword tokens
            keyword_features = self.encode_text(keyword, normalize=True)
            out.update({
                "keyword_features": keyword_features,
            })

        if self.output_dict:
            return out

        return image_features, text_features, self.logit_scale.exp()
