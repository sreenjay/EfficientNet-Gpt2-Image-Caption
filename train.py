import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from dataset import ImageCoco
import config
from model import ImageCaptioningModel
from test import caption_image
from engine import fit, validate
# from utils import SaveBestModel, EarlyStopping, LrScheduler, MyCollate


def run():
    transform = transforms.Compose(
        [   
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


    train_dataset = ImageCoco(root_directory='../../datasets/coco/train2014/', 
                        annFile='../../datasets/coco/annotations/captions_train2014.json', 
                        tokenizer=config.BERT_TOKENIZER,
                        transform=transform)

    val_dataset = ImageCoco(root_directory='../../datasets/coco/val2014/', 
                        annFile='../../datasets/coco/annotations/captions_val2014.json', 
                        tokenizer=config.BERT_TOKENIZER,
                        transform=transform)



    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, 
                                  batch_size=config.TRAIN_BATCH_SIZE, 
                                  collate_fn=MyCollate(config.BERT_TOKENIZER.pad_token_id, config.BERT_TOKENIZER))
    test_dataloader = DataLoader(dataset=val_dataset, shuffle=True, 
                                 batch_size=config.VALID_BATCH_SIZE, 
                                 collate_fn=MyCollate(config.BERT_TOKENIZER.pad_token_id, config.BERT_TOKENIZER))



    target_vocab_len = config.BERT_TOKENIZER.vocab_size
    model = ImageCaptioningModel(target_vocab_len, num_layers=config.NUM_LAYERS, heads=config.NUM_HEADS).to("cuda")
    print(f"Model paramters : {sum(p.numel() for p in model.parameters())}")
    save_best_model = SaveBestModel("bestmodel")
    pad_idx = config.BERT_TOKENIZER.all_special_ids[2]

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


    model.load_state_dict(torch.load("bestmodel.pth")["model_state_dict"])
    optimizer.load_state_dict(torch.load("bestmodel.pth")["optimizer_state_dict"])


    model, train_loss, val_loss = model_train(model,
                                              train_dataloader,
                                              test_dataloader, 
                                              config.EPOCHS,
                                              config.LEARNING_RATE, 
                                              loss_fn, 
                                              optimizer=optimizer,
                                              early_stop=True)
    
    
    run()