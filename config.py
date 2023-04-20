import transformers
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from transformers import logging
logging.set_verbosity_warning()


DEVICE = "cuda"
MAX_LEN = 512
TRAIN_BATCH_SIZE = 18 # Convert to 16
VALID_BATCH_SIZE = 64
DECODER_DROPOUT = 0.2
EPOCHS = 10
LEARNING_RATE = 3e-5
NUM_LAYERS = 12
NUM_HEADS = 12
DECODER_PATH = "gpt2"
MODEL_PATH = "/root/docker_data/model.bin"
PRE_TRAINED_MODELS_PATH = "./pre_trained_models/"
TOKENIZER = GPT2Tokenizer.from_pretrained(DECODER_PATH,
                                       bos_token = '<|endoftext|>',
                                       eos_token = '<|endoftext|>',
                                       pad_token = '<|pad|>',
                                       do_lower_case=True)


PRETRAINED_GPT2MODEL = GPT2LMHeadModel.from_pretrained(DECODER_PATH, 
                                    # output_hidden_states=True, 
                                    embd_pdrop = 0.2, 
                                    attn_pdrop = 0.2,
                                    add_cross_attention=True, 
                                    activation_function='gelu_new').to(DEVICE)

# PRETRAINED_GPT2MODEL = transformers.GPT2Model.from_pretrained(config).to(DEVICE)
