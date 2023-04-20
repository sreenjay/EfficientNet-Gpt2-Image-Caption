import torch
import torchvision
import torchvision.models as models
import config



class ImageCaptioningModel(torch.nn.Module):
    
    
    def __init__(self, target_vocab_len, embedding_size=768, num_layers=12, heads=12, dropout=0.2, feedforward_dim=3072, max_len=1024):
        super().__init__()
        self.max_len = max_len
        
        self.image_encoder = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        self.image_encoder.classifier = torch.nn.Sequential(torch.nn.Dropout(dropout),
                                                            torch.nn.Linear(self.image_encoder.classifier[1].in_features,
                                                                            embedding_size, bias=True))
        
        self.decoder = config.PRETRAINED_GPT2MODEL
        self.decoder.resize_token_embeddings(target_vocab_len)
        # output last few layers of the model to be done later on self.decoder
        # self.output_layer = torch.nn.Linear(embedding_size, target_vocab_len)
        
        for p in self.parameters():
            p.requires_grad = True
            
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1).to(config.DEVICE)
        
        
    def forward(self, image_batch, target_batch, attention_mask):
        N, seq_len = target_batch.size()
        image_encoder = self.image_encoder(image_batch)                       # `Shape : (BATCH_SIZE, EMBEDDING_DIM)`
        image_encoder = image_encoder.unsqueeze(1)                            # `Shape : (BATCH_SIZE, 1, EMBEDDING_DIM)`

        decoder_output = self.decoder(input_ids=target_batch, 
                                      encoder_hidden_states=image_encoder, 
                                      attention_mask=attention_mask)['logits']

        # output = self.output_layer(decoder_output)
        return decoder_output
    
    
    
    
    
    
    
class Encoder(torch.nn.Module):
    
    def __init__(self, embedding_size, dropout=0.1, bias=True):
        super().__init__()
        self.encoder = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        self.encoder.classifier = torch.nn.Sequential(torch.nn.Dropout(dropout),
                                                    torch.nn.Linear(self.encoder.classifier[1].in_features,
                                                                    embedding_size, 
                                                                    bias=bias)
                                                     )
        
        
    def forward(self, X):
        return self.encoder(X)
    
    

class Decoder(torch.nn.Module):
    
    def __init__(self):
        self.decoder = config.PRETRAINED_GPT2MODEL
        self.decoder.resize_token_embeddings(target_vocab_len)
        
    def forward(self, input_ids, encoder_output, attention_mask):
        decoder_output = self.decoder(input_ids=target_batch, 
                                      encoder_hidden_states=encoder_output, 
                                      attention_mask=attention_mask)['logits']
        
        
        
# class ImageCaptioningModel