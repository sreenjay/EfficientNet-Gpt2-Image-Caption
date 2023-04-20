from collections import OrderedDict
from torch.utils.data import Dataset
import os
import json
import PIL
import torch


class ImageCoco(Dataset):
    
    def __init__(self, root_directory:str, annFile:str, tokenizer, transform=None, frequency_threshold=1):
        super().__init__()
        self.root_dir = root_directory
        self.annotations = self._get_annotations(annFile)
        self.transform = transform
        self.captions = [caption for _, caption in self.annotations]
        self.tokenizer = tokenizer
        self.vocab = self.tokenizer.get_vocab()

        
    def __len__(self):
        return len(self.captions)
    
    
    def __getitem__(self, index):
        img_id, caption = self.annotations[index]
        file_dest = os.path.join(self.root_dir, img_id)
        img = PIL.Image.open(file_dest).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        encoded_caption = [self.tokenizer.bos_token_id]
        encoded_caption += self.tokenizer.encode_plus(caption)['input_ids'] 
        encoded_caption.append(self.tokenizer.eos_token_id)
        return img, encoded_caption
        
        
    def _get_annotations(self, annotation_file):
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        
        images = annotations['images']
        annotations_list = annotations['annotations']
        image_filename_dict = {image['id'] : image['file_name'] for image in images}
        id_to_caption = []
        for annotation in annotations_list:
            image_id = annotation['image_id']
            category_id = annotation['id']
            caption = annotation['caption']
            if image_id in image_filename_dict.keys():
                filename = image_filename_dict[image_id]
                id_to_caption.append((f"{filename}", caption))
        
        return id_to_caption
    
    
    
class MyCollate:
    
    def __init__(self, pad_idx, tokenizer, max_len=1024):
        self.pad_idx = pad_idx
        self.max_len = max_len
        self.tokenizer = tokenizer
        
    def __call__(self, X_batch):
        images, captions = zip(*X_batch)
        captions = tuple([tokens for tokens in captions])
        tokenized_texts = tuple([sent[:self.max_len] if len(sent) > self.max_len else sent for sent in captions])
        max_length = max(len(seq) for seq in tokenized_texts)
        padded_tokenized_texts = [seq + [self.pad_idx]*(max_length - len(seq)) for seq in tokenized_texts]
        # Create attention masks
        attention_masks = torch.Tensor([[int(token_id == self.pad_idx) for token_id in seq] for seq in padded_tokenized_texts])
        mask = attention_masks == 0
        return torch.stack(images), torch.Tensor(padded_tokenized_texts).long(), mask