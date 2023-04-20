import torch
import PIL
import numpy
import requests
from io import BytesIO
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import config
import numpy as np



def caption_image(model, image_link, max_len=100):
    if image_link.startswith('http'):
        res = requests.get(image_link)
        img = PIL.Image.open(BytesIO(res.content))
    else:
        img = PIL.Image.open(image_link)
        
    img = TF.resize(img, [224, 224])
    plt.imshow(img)
    
    numpyarray = np.asarray(img)
    image_tensor = TF.to_tensor(numpyarray)
    encoder_input = TF.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(config.DEVICE)
    encoder_input = encoder_input.unsqueeze(0)
    encoder_output = model.image_encoder(encoder_input).unsqueeze(1)
    tgt_tokens = torch.ones(1, 1 ).fill_(config.BERT_TOKENIZER.cls_token_id).type(torch.long).to(config.DEVICE)
    
    for i in range(max_len - 1):
        tgt_mask = model.generate_square_subsequent_mask(tgt_tokens.size(1))
        decoder_emb = model.embeddings(tgt_tokens)
        out = model.decoder(decoder_emb, encoder_output)
        prob = model.decoder_fc_out(out[:, -1])
        next_word = torch.argmax(prob, dim=1)
        next_word = next_word.item()
        if next_word == config.BERT_TOKENIZER.sep_token_id:
            break
        decoder_input_masked = torch.ones(1,1).type_as(tgt_tokens.data).fill_(next_word).to(config.DEVICE)
        tgt_tokens = torch.cat((tgt_tokens, decoder_input_masked), dim=1)
        
    decoded_sentence =  config.BERT_TOKENIZER.decode(tgt_tokens[0])
    return decoded_sentence.replace(config.BERT_TOKENIZER.cls_token, '')[1:].capitalize()



if __name__ == "__main__" :
    file = 'https://hips.hearstapps.com/hmg-prod/images/golden-retriever-royalty-free-image-506756303-1560962726.jpg?crop=1.00xw:0.756xh;0,0.0756xh&resize=1200:*'
    caption_image(model, file)
    
