import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    
    def __init__(self, ds, tokenizer_src,tokenizer_tgt, src_lang, tgt_lang, seq_len ) -> None:
        super().__init__()
        
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        
        # self.tokenized_src = []
        # self.tokenized_tgt = []

        # for item in self.ds:
        #     self.tokenized_src.append(
        #         self.tokenizer_src.encode(
        #             item['translation'][self.src_lang]
        #         ).ids
        #     )

        #     self.tokenized_tgt.append(
        #         self.tokenizer_tgt.encode(
        #             item['translation'][self.tgt_lang]
        #         ).ids
        #     )     
        
          
        
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id('[PAD]')], dtype=torch.int64)
                
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index : any) -> any:
        src_target_pair = self.ds[index]

        src_text = src_target_pair['en']
        tgt_text = src_target_pair['hi_en']
        
            # Read pre-tokenized IDs from the dataset item
        enc_input_tokens = src_target_pair['src_ids']
        dec_input_tokens = src_target_pair['tgt_ids']
        
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # minus 2 because of SOS AND EOS tokens included
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # minus 1 because in training we only add SOS token in decoder side and during label we only add EOS 
        
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')
        
        
        pad_id = self.pad_token.item() # Extract the raw integer
        
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.full((enc_num_padding_tokens,), pad_id, dtype=torch.int64)
            ]
        )
        
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.full((dec_num_padding_tokens,), pad_id, dtype=torch.int64)
            ]
        )
        
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.full((dec_num_padding_tokens,), pad_id, dtype=torch.int64)
            ]
        )
        
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len
        
        return{
            "encoder_input" : encoder_input ,       
            "decoder_input" : decoder_input,        
            "encoder_mask" : (encoder_input != pad_id).unsqueeze(0).unsqueeze(0).int(),     
            "decoder_mask" : (decoder_input != pad_id).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), 
             "label" : label,    # ( seq_len)
            "src_text" : src_text,
            "tgt_text" : tgt_text
                        
            }
        
        # add SOS and EOS to the source text
        
        # encoder_input = torch.cat(
        #     [
        #         self.sos_token,
        #         torch.tensor(enc_input_tokens, dtype=torch.int64),
        #         self.eos_token,
        #         torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
        #     ]
            
        # )
        
        # # add SOS to the decoder input
        # decoder_input = torch.cat(
        #     [
        #         self.sos_token,
        #         torch.tensor(dec_input_tokens, dtype=torch.int64),
        #         torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
                
        #     ]
        # )
        
        # # add EOS to the label (what we expect as output from the decoder )
        
        # label = torch.cat(
        #     [
        #         torch.tensor(dec_input_tokens, dtype=torch.int64),
        #         self.eos_token,
        #         torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
                
        #     ]
        # )
        
        # assert encoder_input.size(0) == self.seq_len
        # assert decoder_input.size(0) == self.seq_len
        # assert label.size(0) == self.seq_len
        
        # return{
        #     "encoder_input" : encoder_input ,       # seq_len
        #     "decoder_input" : decoder_input,        # seq_len
        #     "encoder_mask" : (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),     # ( 1, 1, seq_len)
        #     "decoder_mask" : (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # ( 1, seq_len) & ( 1, seq_len, seq_len)                                   
        #     # the mask should be causal not see future tokens / words or each it means that each word can only look at previous word and each word can only look at non padding words
        #     "label" : label,    # ( seq_len)
        #     "src_text" : src_text,
        #     "tgt_text" : tgt_text
            
        # }


def causal_mask(size):  # if martix of N X N  where n is no of words we want all the words about the diagonal of the matrix masked / hidden
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0            
        
        
            
        
        
        