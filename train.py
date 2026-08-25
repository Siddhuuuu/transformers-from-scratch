import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

from dataset import BilingualDataset,causal_mask

from model import build_transformer

from datasets import load_dataset
from tokenizers import Tokenizer
# from tokenizers.models import WordLevel
# from tokenizers.trainers import WordLevelTrainer

from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from config import get_weights_file_path, get_config

from tqdm import tqdm

import warnings


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    
    # precompute the encoder output and reuse it for every token we get from the decoder
    encoder_ouput = model.encode(source, source_mask)
    
    # how inference happens? we give the decoder SOS token so that the decoder will output the 1st token of the sentence/translation
    # then at every iteration we add the previous token to the decoder input so that the decoder could output the next token
    
    # initialize the decoder input with the sos token
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break
        # build mask for the target (decoder input)
        
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        
        # calculate the output of the decoder
        
        out = model.decode(encoder_ouput, source_mask, decoder_input, decoder_mask)
        
        # get the next token
        prob = model.project(out[:,-1])
        # select the token with the max probability ( because it is a greedy search )
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim = 1)
        
        if next_word == eos_idx:
            break
        
    return decoder_input.squeeze(0)
   

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, writer, num_examples=2):
    model.eval()    #this tells pytorch that we are going to evaluate our model
    # then we will do inference of 2 sentences and see the output of the model
    
    count = 0
    ##source_texts = []
    ##expected = []
    ##predicted = []
    # size of the control window ( just use a default value )
    console_width = 80
    
    with torch.no_grad():  # we are disabling the gradient calculation for this part
        for batch in validation_ds:
            count+=1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"
            
            model_out = greedy_decode( model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
            
            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
            
            ##source_texts.append(source_text)
            ##expected.append(target_text)
            ##predicted.append(model_out_text)
            # print to the console
            # why print msg why no direct print reason is tdqm
            
            print_msg('-'*console_width)
            print_msg(f'SOURCE:{source_text}')
            print_msg(f'TARGET:{target_text}')
            print_msg(f'PREDICTED:{model_out_text}')
            
            if count == num_examples:
                break
            
    ## if writer:
        ## TorchMetrics CharrErrorRate, BLEU, WordErrorRate            

def get_all_sentences(ds, lang):
    for item in ds:
        yield item[lang]
    

def get_or_build_tokenizer(config, ds, lang):
    # config['tokenizer_file'] = '../tokenizers/tokinzer_{0}.json' 
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2,
            vocab_size=16000
        )
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer
    
def get_ds(config):
    ds_train_raw = load_dataset(
        'rvv-karma/English-Hinglish-TOP',
        split='train'
    )

    ds_val_raw = load_dataset(
        'rvv-karma/English-Hinglish-TOP',
        split='validation'
    )
    
    print(f"Validation examples: {len(ds_val_raw)}")

    # build tokenizers
    tokenizer_src = get_or_build_tokenizer(
        config,
        ds_train_raw,
        config['lang_src']
    )

    tokenizer_tgt = get_or_build_tokenizer(
        config,
        ds_train_raw,
        config['lang_tgt']
    )

    # this dataset uses separate fields for English and Hinglish
    # def fits_seq_len(item):
    #     src_len = len(
    #         tokenizer_src.encode(item['en']).ids
    #     ) + 2

    #     tgt_len = len(
    #         tokenizer_tgt.encode(item['hi_en']).ids
    #     ) + 1

    #     return max(src_len, tgt_len) <= config['seq_len']

    # print(f"Original training examples: {len(ds_train_raw)}")

    # ds_train = ds_train_raw.filter(fits_seq_len)
    # ds_val = ds_val_raw.filter(fits_seq_len)
    
    def tokenize_and_filter(batch):
        src_tokens = tokenizer_src.encode_batch(batch['en'])
        tgt_tokens = tokenizer_tgt.encode_batch(batch['hi_en'])
        
        keep_indices = []
        src_ids = []
        tgt_ids = []
        
        for i in range(len(batch['en'])):
            src_len = len(src_tokens[i].ids) + 2
            tgt_len = len(tgt_tokens[i].ids) + 1
            if max(src_len, tgt_len) <= config['seq_len']:
                keep_indices.append(i)
                src_ids.append(src_tokens[i].ids)
                tgt_ids.append(tgt_tokens[i].ids)
                
        return {
            'src_ids': src_ids,
            'tgt_ids': tgt_ids,
            'en': [batch['en'][i] for i in keep_indices],
            'hi_en': [batch['hi_en'][i] for i in keep_indices]
        }

    # Parallel tokenization using C++ backend (100x faster than Python loops)
    ds_train = ds_train_raw.map(tokenize_and_filter, batched=True, batch_size=1000, remove_columns=ds_train_raw.column_names)
    ds_val = ds_val_raw.map(tokenize_and_filter, batched=True, batch_size=1000, remove_columns=ds_val_raw.column_names)

    print(f"Filtered validation examples: {len(ds_val)}")
    print(f"Removed validation examples: {len(ds_val_raw) - len(ds_val)}")

    print(f"Filtered training examples: {len(ds_train)}")
    print(f"Removed training examples: {len(ds_train_raw) - len(ds_train)}")

    train_ds = BilingualDataset( ds_train, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    val_ds = BilingualDataset( ds_val, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'] )

    train_dataloader = DataLoader(
        train_ds,
        batch_size=config['batch_size'],
        shuffle=True,
        pin_memory=True,
        num_workers=2,
        persistent_workers=True
    )

    val_dataloader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        persistent_workers=True
    )

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'] )
    return model


def train_model(config):
    # define the device i.e using gpu or cpu 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device{device}')
    
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    if torch.__version__ >= '2.0':
        print("Compiling model for faster training...")
        model = torch.compile(model)
    
    # TensorBoard -> allows to visualise the loss / graphics / charts
    
    writer = SummaryWriter(config['experiment_name'])
    
    optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config['lr'],
    weight_decay=config['weight_decay'],
    eps=1e-9
)
    scaler = GradScaler("cuda")
    
    intial_epoch = 0
    global_step = 0
    
    if config['preload']:
        model_filename = get_weights_file_path( config, config['preload'])
        print(f'Preloading model{model_filename}')
        
        state = torch.load(model_filename, map_location=device)

        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])

        intial_epoch = state['epoch'] + 1
        global_step = state['global_step']
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id('[PAD]'), label_smoothing=0.1).to(device) 
    # label smoothing helps reduce overfit and increase accuracy by reducing very high confidence of model making it a just bit less sure of its choices
    
    # TRAINING LOOP 
    
    for epoch in range( intial_epoch,config['nums_epochs']):
        
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        
        for batch_index, batch in enumerate(batch_iterator):
            model.train()
            
            encoder_input = batch['encoder_input'].to(device, non_blocking=True) # ( B,seq_len)
            decoder_input = batch['decoder_input'].to(device, non_blocking=True) # ( B,seq_len)
            encoder_mask = batch['encoder_mask'].to(device, non_blocking=True)  # ( B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device, non_blocking=True)  # ( B,1, seq_len, seq_len)
            
            # Run the tensors throught the transformers
            # encoder_output = model.encode(encoder_input, encoder_mask)  # ( B, seq_len, d_model)
            # decoder_output = model.decode(encoder_output, encoder_mask, decoder_input,decoder_mask) # ( B, seq_len, d_model)
            # proj_output = model.project(decoder_output) # ( B, seq_len, tgt_vocab_size)
            
            with autocast("cuda"):
                encoder_output = model.encode(encoder_input, encoder_mask)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                proj_output = model.project(decoder_output)

                label = batch['label'].to(device, non_blocking=True)

                loss = loss_fn(
                    proj_output.view(-1, tokenizer_tgt.get_vocab_size()),
                    label.view(-1)
                )
            
            #label = batch['label'].to(device) # ( B, seq_len)
            
            #  ( B, seq_len, tgt_vocab_size) -->> ( B * seq_len, tgt_vocab_size)
            

            # Log the LOSS
            
            writer.add_scalar('train loss', loss.item(), global_step)
            
            
            # Backpropagate the loss
            # loss.backward()
            
            # update the weights
            # optimizer.step()
            # optimizer.zero_grad()
            
            loss = loss / config["gradient_accumulation_steps"]
            scaler.scale(loss).backward()
            
            if ((batch_index + 1) % config["gradient_accumulation_steps"] == 0
                    or (batch_index + 1) == len(train_dataloader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            
            
            global_step += 1
            
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)
        
        writer.flush()
            
            # save the model at the end of the every epoch
            
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        # Save the ORIGINAL (uncompiled) model's state dict for compatibility
        model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
        torch.save({
            'epoch' : epoch,
            'model_state_dict' : model_to_save.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'global_step' : global_step  
        }, model_filename)
            
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)         
            
            
            
            
            
            
            
                                       
            
            
        
        
        
    
    
    
     
    

    