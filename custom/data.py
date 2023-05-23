import os
import json
import h5py
import random
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from datasets import load_dataset

from training.data import DataInfo
from torchrs.datasets import RSICD, UCMCaptions, SydneyCaptions
from torchrs.datasets import UCM, WHURS19, RSSCN7, AID, RESISC45


class RSICD_CLS(RSICD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_class_info(os.path.join(self.root, "txtclasses_rsicd"))

    def load_class_info(self, class_dir):
        classes = []
        path2class = {}
        for idx, fn in enumerate(sorted(os.listdir(class_dir))):
            classes.append(fn.split(".txt")[0])
            with open(os.path.join(class_dir, fn)) as f:
                for line in f.readlines():
                    path2class[line.strip()] = idx

        self.classes = classes
        self.path2class = path2class

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        filename = self.captions[idx]["filename"]
        path = os.path.join(self.root, self.image_root, filename)
        x = self.transform(Image.open(path).convert("RGB"))
        y = self.path2class[filename]
        return x, y


class Fashion200k(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.transform = transform

        self.data = self._load_annotation_db(split)

    def _load_annotation_db(self, split):
        split = {'train': 'train', 'val': 'test'}[split]

        txt_path = [
            f'dress_{split}_detect_all.txt',
            f'jacket_{split}_detect_all.txt',
            f'pants_{split}_detect_all.txt',
            f'skirt_{split}_detect_all.txt',
            f'top_{split}_detect_all.txt',
        ]

        data = {}
        for txt in txt_path:
            with open(os.path.join(self.root, txt), 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    image_path, _, sentences = line.split('\t')
                    item_id = image_path.split('/')[3]

                    if not os.path.exists(os.path.join(self.root, image_path)):
                        continue

                    if item_id in data:
                        data[item_id]['image_path'].append(image_path)
                    else:
                        data[item_id] = dict(image_path=[image_path], sentences=sentences)
        data = [dict({'id': item_id}, **data[item_id]) for item_id in data]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        path = os.path.join(self.root, random.choice(item["image_path"]))
        x = Image.open(path).convert("RGB")
        x = self.transform(x)

        sentences = item['sentences']
        return dict(x=x, captions=sentences)


class FashionGen(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.transform = transform

        self.data, self.images = self._load_annotation_db(split)

    def _load_annotation_db(self, split):
        split = {'train': 'train', 'val': 'validation'}[split]
        h5_path = os.path.join(self.root, f"fashiongen_256_256_{split}.hdf5")
        h5_file = h5py.File(h5_path)

        data = {}
        for idx in range(len(h5_file['index'])):
            item_id = int(h5_file['input_productID'][idx])
            input_name = h5_file['input_name'][idx][0]
            input_desc = h5_file['input_description'][idx][0]

            if item_id in data:
                data[item_id]['image_idx'].append(idx)
            else:
                data[item_id] = dict(image_idx=[idx], input_name=input_name, input_desc=input_desc)
        data = [dict({'id': item_id}, **data[item_id]) for item_id in data]

        images = h5_file['input_image']

        return data, images

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        x = self.images[random.choice(item['image_idx'])]
        x = Image.fromarray(x)
        x = self.transform(x)

        sentences = item['input_name'].decode('latin-1') + ". "
        sentences += item['input_desc'].decode('latin-1')

        return dict(x=x, captions=sentences)


class Polyvore(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.transform = transform

        self.data = self._load_annotation_db(split)

    def _load_annotation_db(self, split):
        json_path = os.path.join(self.root, f"{split}_info.json")
        with open(json_path, 'r') as f:
            anno_json = json.load(f)

        data = []
        for item in anno_json:
            data.append(
                {
                    "image_path": item["images"],
                    "id": item["id"],
                    "sentences": item["title"] + "." + item["description"],
                    "attributes_id": item["attributes_id"],
                }
            )

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        path = os.path.join(self.root, 'images', item["image_path"])
        x = Image.open(path).convert("RGB")
        x = self.transform(x)

        sentences = item['sentences']
        return dict(x=x, captions=sentences)


class Fashion200k_CLS(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.transform = transform

        self.data = self._load_annotation_db(split)

        # Remove some broken links
        self.data = [item for item in self.data if os.path.exists(os.path.join(self.root, 'women', item["image_path"]))]

        self.classes = set()
        for item in self.data:
            cls = item['class_name']
            self.classes.add(cls)
        self.classes = list(sorted(list(self.classes)))

    def _load_annotation_db(self, split):
        split = {'train': 'train', 'val': 'test'}[split]
        json_path = os.path.join(self.root, f"{split}_info.json")
        with open(json_path, 'r') as f:
            anno_json = json.load(f)

        data = []
        for item in anno_json:
            for image_path in item['images']:
                class_name = image_path.split("/")[0].replace("_", " ")
                data.append(
                    {
                        "image_path": image_path,
                        "class_name": class_name
                    }
                )

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        path = os.path.join(self.root, 'women', item["image_path"])
        x = Image.open(path).convert("RGB")
        x = self.transform(x)

        cls_name = item['class_name']
        y = self.classes.index(cls_name)
        return x, y


class Fashion200k_SUBCLS(Fashion200k_CLS):
    def _load_annotation_db(self, split):
        split = {'train': 'train', 'val': 'test'}[split]
        json_path = os.path.join(self.root, f"{split}_info.json")
        with open(json_path, 'r') as f:
            anno_json = json.load(f)

        data = []
        for item in anno_json:
            for image_path in item['images']:
                class_name = image_path.split("/")[1].replace("_", " ")
                data.append(
                    {
                        "image_path": image_path,
                        "class_name": class_name
                    }
                )

        return data


class FashionGen_CLS(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.transform = transform

        self.data = self._load_annotation_db(split)

        self.classes = set()
        for cls in self.data['input_category']:
            cls = cls[0].decode('UTF-8').lower()
            self.classes.add(cls)
        self.classes = list(sorted(list(self.classes)))

    def _load_annotation_db(self, split):
        split = {'train': 'train', 'val': 'validation'}[split]
        h5_path = os.path.join(self.root, f"fashiongen_256_256_{split}.hdf5")
        data = h5py.File(h5_path)

        return data

    def __len__(self):
        return len(self.data['index'])

    def __getitem__(self, idx):
        x = self.data['input_image'][idx]
        x = Image.fromarray(x)
        x = self.transform(x)

        cls_name = self.data['input_category'][idx][0].decode('UTF-8').lower()
        y = self.classes.index(cls_name)
        return x, y


class FashionGen_SUBCLS(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.transform = transform

        self.data = self._load_annotation_db(split)

        self.classes = set()
        for cls in self.data['input_subcategory']:
            cls = cls[0].decode('UTF-8').lower()
            self.classes.add(cls)
        self.classes = list(sorted(list(self.classes)))

    def _load_annotation_db(self, split):
        split = {'train': 'train', 'val': 'validation'}[split]
        h5_path = os.path.join(self.root, f"fashiongen_256_256_{split}.hdf5")
        data = h5py.File(h5_path)

        return data

    def __len__(self):
        return len(self.data['index'])

    def __getitem__(self, idx):
        x = self.data['input_image'][idx]
        x = Image.fromarray(x)
        x = self.transform(x)

        cls_name = self.data['input_subcategory'][idx][0].decode('UTF-8').lower()
        y = self.classes.index(cls_name)
        return x, y


class Polyvore_CLS(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.transform = transform

        self.data = self._load_annotation_db(split)

        self.classes = set()
        for item in self.data:
            cls = item['class_name']
            self.classes.add(cls)
        self.classes = list(sorted(list(self.classes)))

    def _load_annotation_db(self, split):
        json_path = os.path.join(self.root, f"{split}_info.json")
        metadata_path = os.path.join(self.root, "polyvore_item_metadata.json")
        with open(json_path, 'r') as f:
            anno_json = json.load(f)
        with open(metadata_path, 'r') as f:
            meta_json = json.load(f)

        data = []
        for item in anno_json:
            data.append(
                {
                    "image_path": item["images"],
                    "id": item["id"],
                    "class_name": meta_json[str(item['id'])]['semantic_category']
                }
            )

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        path = os.path.join(self.root, 'images', item["image_path"])
        x = Image.open(path).convert("RGB")
        x = self.transform(x)

        cls_name = item['class_name']
        y = self.classes.index(cls_name)
        return x, y


class SciCap(Dataset):
    MAXLEN = 77  # maximum length for caption
    def __init__(self, root, split, transform=None):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform

        self.samples = self._init_data()

    def _init_data(self):
        image_root = os.path.join(self.root, "SciCap-No-Subfig-Img", self.split)
        json_root = os.path.join(self.root, "SciCap-Caption-All", self.split)

        samples = []
        for filename in os.listdir(json_root):
            with open(os.path.join(json_root, filename)) as f:
                json_object = json.load(f)
            if json_object["contains-subfigure"]:
                continue

            path = os.path.join(image_root, str(filename).replace("json", "png"))
            caption = json_object['0-originally-extracted']
            caption = caption[:self.MAXLEN]  # cut long captions
            samples.append([path, caption])

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, caption = self.samples[idx]
        image = self.transform(Image.open(path).convert("RGB"))
        return image, caption


class TokenizedDataset(Dataset):
    def __init__(self, dataset, image_key=None, text_key=None,
                 tokenizer=None, keywords=None):
        self.dataset = dataset
        self.image_key = image_key
        self.text_key = text_key
        self.tokenize = tokenizer or (lambda x: x)

        self.keywords = keywords
        self.keyword_tokens = self._init_keyword_tokens()

    def _init_keyword_tokens(self):
        if self.keywords is not None:
            BOS, EOS = 49406, 49407
            keyword_tokens = []
            for k in self.keywords:
                k = self.tokenize(k).flatten().tolist()
                k = k[k.index(BOS) + 1: k.index(EOS)]
                keyword_tokens.append(k)
            return keyword_tokens
        else:
            return None

    def _find_keyword(self, tokens, key):
        for i in range(len(tokens)):
            idx = i  # candidate
            for j in range(len(key)):
                if tokens[i+j] != key[j]:
                    idx = None
                    break

            if idx is not None:
                return idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[int(idx)]

        # read data, which is dict or list
        if isinstance(data, (list, tuple)):
            images, texts = data
        else:
            assert isinstance(data, dict)
            assert self.image_key and self.text_key
            images = data[self.image_key]
            texts = data[self.text_key]

        # tokenize captions
        if isinstance(texts, list):
            texts = str(random.choice(texts))
        tokens = self.tokenize([str(texts)])[0]

        # done if not using keywords
        if self.keywords is None:
            return images, tokens

        # logics for parsing keyword labels
        keyword_labels = torch.zeros(len(self.keywords), 3)
        spaced = lambda word: " {} ".format(word)
        for i, k in enumerate(self.keywords):
            if spaced(k) in spaced(texts):
                # find index of the keyword
                key = self.keyword_tokens[i]
                idx = self._find_keyword(tokens.tolist(), key)
                assert all(tokens[idx+i] == key[i] for i in range(len(key)))

                keyword_labels[i][0] = 1
                keyword_labels[i][1] = idx
                keyword_labels[i][2] = len(key)

        return images, tokens, keyword_labels


def split_data(d, split_ratio, seed=42, hf_data=False):
    # set random seed
    gen = torch.Generator()
    gen.manual_seed(seed)

    # split labeled and unlabeled data
    indices = torch.randperm(len(d), generator=gen)
    size = int(len(d) * split_ratio)

    if hf_data is False:
        d1 = Subset(d, indices[:size])
        d2 = Subset(d, indices[size:])
    else:
        d1 = [d[int(i)] for i in indices[:size]]
        d2 = [d[int(i)] for i in indices[size:]]

    return d1, d2


def read_keywords(path):
    keywords = []
    with open(path, "r") as f:
        for line in f.readlines():
            keywords.append(line.strip())
    return keywords


def create_datainfo(args, dataset, batch_size, is_train):
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None
    workers = args.workers if not args.train_data else 0

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=False,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_custom_data(args, data, preprocess_fn, is_train, **data_kwargs):
    split = "train" if is_train else "val"

    REMOTE_SENSING_CAPTIONS = ["RSICD", "UCM", "Sydney", "RS-ALL"]
    REMOTE_SENSING_ZEROSHOT = ["RSICD-CLS", "UCM-CLS", "WHU-RS19", "RSSCN7", "AID", "RESISC45"]

    FASHION_CAPTIONS = ["Fashion200k", "FashionGen", "Polyvore"]
    FASHION_ZEROSHOT = ["Fashion200k-CLS", "Fashion200k-SUBCLS", "FashionGen-CLS", "FashionGen-SUBCLS", "Polyvore-CLS"]

    if data in REMOTE_SENSING_CAPTIONS:
        if data == "RSICD":
            d = RSICD("/data/aerial/RSICD", split=split, transform=preprocess_fn)
        elif data == "UCM":
            d = UCMCaptions("/data/aerial/UCM_captions", split=split, transform=preprocess_fn)
        elif data == "Sydney":
            d = SydneyCaptions("/data/aerial/Sydney_captions", split=split, transform=preprocess_fn)
        elif data == "RS-ALL":
            d = ConcatDataset([
                RSICD("/data/aerial/RSICD", split=split, transform=preprocess_fn),
                UCMCaptions("/data/aerial/UCM_captions", split=split, transform=preprocess_fn),
                SydneyCaptions("/data/aerial/Sydney_captions", split=split, transform=preprocess_fn),
            ])

        d = TokenizedDataset(d, image_key="x", text_key="captions", **data_kwargs)

        return d

    elif data in REMOTE_SENSING_ZEROSHOT:
        if data == "RSICD-CLS":
            d = RSICD_CLS("/data/aerial/RSICD", split=split, transform=preprocess_fn)
        elif data == "UCM-CLS":
            d = UCM("/data/aerial/UCMerced_LandUse", transform=preprocess_fn)
        elif data == "WHU-RS19":
            d = WHURS19("/data/aerial/WHU-RS19", transform=preprocess_fn)
        elif data == "RSSCN7":
            d = RSSCN7("/data/aerial/RSSCN7", transform=preprocess_fn)
            d.classes = [c[1:] for c in d.classes]  # "aGrass" -> "Grass"
        elif data == "AID":
            d = AID("/data/aerial/AID", transform=preprocess_fn)
        elif data == "RESISC45":
            d = RESISC45("/data/aerial/NWPU-RESISC45", transform=preprocess_fn)

        template = [lambda c: f"an aerial photograph of {c}."]

        return d, d.classes, template

    elif data in FASHION_CAPTIONS:
        if data == "Fashion200k":
            d = Fashion200k("/data/fashion200k", split=split, transform=preprocess_fn)
        elif data == "FashionGen":
            d = FashionGen("/data/FashionGen", split=split, transform=preprocess_fn)
        elif data == "Polyvore":
            d = Polyvore("/data/PolyvoreOutfits", split=split, transform=preprocess_fn)
        elif data == "Fashion-ALL":
            d = ConcatDataset([
                Fashion200k("/data/fashion200k", split=split, transform=preprocess_fn),
                FashionGen("/data/FashionGen", split=split, transform=preprocess_fn),
                Polyvore("/data/PolyvoreOutfits", split=split, transform=preprocess_fn),
            ])

        d = TokenizedDataset(d, image_key="x", text_key="captions", **data_kwargs)
        return d

    elif data in FASHION_ZEROSHOT:
        if data == 'Fashion200k-CLS':
            d = Fashion200k_CLS("/data/fashion200k", split=split, transform=preprocess_fn)
        elif data == 'Fashion200k-SUBCLS':
            d = Fashion200k_SUBCLS("/data/fashion200k", split=split, transform=preprocess_fn)
        elif data == "FashionGen-CLS":
            d = FashionGen_CLS("/data/FashionGen", split=split, transform=preprocess_fn)
        elif data == 'FashionGen-SUBCLS':
            d = FashionGen_SUBCLS("/data/FashionGen", split=split, transform=preprocess_fn)
        if data == "Polyvore-CLS":
            d = Polyvore_CLS("/data/PolyvoreOutfits", split=split, transform=preprocess_fn)

        template = [lambda c: f"a photo of a {c}."]

        return d, d.classes, template

    else:
        if data == "SciCap":
            d = SciCap("/data/science/scicap_data", split=split, transform=preprocess_fn)
            d = TokenizedDataset(d, **data_kwargs)

        elif data in ["Simpsons", "Simpsons-Captions"]:
            d = load_dataset("Norod78/simpsons-blip-captions", keep_in_memory=True)
            image_key, text_key = "image", "text"

            def transform(batch, MAXLEN=77):
                batch[image_key] = [preprocess_fn(image) for image in batch[image_key]]
                batch[text_key] = [text[:MAXLEN] for text in batch[text_key]]
                return batch
            d.set_transform(transform)

            train_ratio = 0.9  # use 90% for training data
            d_train, d_val = split_data(d["train"], train_ratio, seed=42, hf_data=True)
            d = d_train if is_train else d_val

            d = TokenizedDataset(d, image_key=image_key, text_key=text_key, **data_kwargs)

        elif data == "Simpsons-Images":
            d = ImageFolder("/data/simpsons_dataset", transform=preprocess_fn)

        else:
            raise ValueError(f"Unknown dataset: {data}")

        return d


def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data:
        train_kwargs = {"is_train": True, "preprocess_fn": preprocess_train, "tokenizer": tokenizer}

        if args.keyword_path is not None:
            keywords = read_keywords(args.keyword_path)
            data["keyword"] = torch.cat([tokenizer(k) for k in keywords])
            train_kwargs.update({"keywords": keywords})

        if args.train_data == "RS-SHIFT":
            d_train = get_custom_data(args, "RS", **train_kwargs)
            d_train, _ = split_data(d_train, args.label_ratio, seed=args.seed)
            d_query, _, _ = get_custom_data(args, "RESISC45", **train_kwargs)
        elif args.train_data == "Simpsons":
            d_train = get_custom_data(args, "Simpsons-Captions", **train_kwargs)
            d_query = get_custom_data(args, "Simpsons-Images", **train_kwargs)
        else:
            d_train = get_custom_data(args, args.train_data, **train_kwargs)
            d_train, d_query = split_data(d_train, args.label_ratio, seed=args.seed)

        if args.method == "base":
            data["train"] = create_datainfo(args, d_train, args.batch_size, is_train=True)
        else:
            # assume L:U = 1:1
            data["train"] = create_datainfo(args, d_train, args.batch_size // 2, is_train=True)
            data["query"] = create_datainfo(args, d_query, args.batch_size // 2, is_train=True)

    if args.val_data:
        d_val = get_custom_data(args, args.val_data, preprocess_val, is_train=False, tokenizer=tokenizer)
        data["val"] = create_datainfo(args, d_val, args.batch_size, is_train=False)

    if args.imagenet_val is not None:
        d_zeroshot, classnames, template = get_custom_data(args, args.imagenet_val, preprocess_val, is_train=False)
        data["zeroshot-val"] = create_datainfo(args, d_zeroshot, args.batch_size, is_train=False)
        data["classnames"] = classnames
        data["template"] = template

    return data

