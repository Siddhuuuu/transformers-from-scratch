import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size
        self.embedding=nn.Embedding(vocab_size,d_model)
        
    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model) # as per paper multiply by sqrt of d_model
        # this self.embedding is just a dictionary kind of a layer that just maps numbers to the same vector every time and this vector is learned by the model 
    
class PositionalEncoding(nn.Module):   # we want to convey to the model the info about the position of each word inside the sentence and
    # this is done by adding another vector of same size as the embedding so of size 512 that includes some special values given by a formula
    # that tells the model that this particular word occupies this postion in the sentence 
    #so we will create these vectors called positional embeddings and we will add them to the embeddings
    #these are only computed once and reused for every sentence during training and inference 
     
    
    def __init__(self,d_model:int,seq_len:int,dropout:float) -> None: #dropout is to make model less overfit
       # d_model -> size of the vector the positional encoding should be & seq_len is the maximum length of the sentence
       # and because we need to create one vector for each position
        super().__init__()
        self.d_model=d_model
        self.seq_len=seq_len
        self.dropout=nn.Dropout(dropout)
        
        #positional encoding is a matrix of size - seq_len into d_model
        # create a matrix of shape (seq_len,d_model)
        
        pe=torch.zeros(seq_len,d_model)      
        position=torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)  # tensor of shape (seq_len,1) 
        
        div_term=torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0)/d_model))
        # we calculated it in log space for numerical stability the actual calculated value from the given formula will be slightly different
        # but the result will be the same the model will learn this positiona encoding 
        # functions that convey the positional information to the model
        
        # sine is used for even position and cosine is used for odd positions 
        
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        # and then we need to add the batch dimension to this tensor so that we can apply it to the whole sentences
        
        pe = pe.unsqueeze(0) #(1,seq_len,d_model)
        
        # finally we can register this tensor in the buffer of this module 
        # basically when we have a tensor that u want to keep inside module not as a learned parameter 
        #  but u want it to be saved when u save the file of the model s we should register it as a buffer
        # this way the tensor will be saved in the file along with the state of the model
        
        self.register_buffer('pe',pe)
        
    def forward(self,x):
        x = x + (self.pe[:,:x.shape[1],:]).requires_grad_(False) # positional encoding fixed dont want the model to learn so requires_grad is False
        return self.dropout(x)    
      
      
      
# class LayerNormalization(nn.Module):
    
#     def __init__(self, eps:float=10**-6)-> None:
#         super().__init__()
#         self.eps=eps       # eps is used for numerical stability / to avoid division by zero 
#         self.alpha = nn.Parameter(torch.ones(1))  # nn parameter makes it learnable parameter 
#         self.bias = nn.Parameter(torch.zeros(1))    # alpha is multiplicative and bias is additive parameter
        
#     # def forward(self,x):
#     #    mean = x.mean(dim = -1,keepdim=True)     
#     #    std = x.std(dim = -1, keepdim = True)
#     #    return self.alpha * (x - mean) / (std + self.eps ) + self.bias 
    
#     def forward(self,x):
#        import torch.nn.functional as F
#        # Use PyTorch's highly optimized CUDA kernel for LayerNorm
#        return F.layer_norm(x, (x.shape[-1],), self.alpha, self.bias, self.eps)

class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps:float=10**-6)-> None:
        super().__init__()
        self.eps=eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        
    def forward(self,x):
       import torch.nn.functional as F
       return F.layer_norm(x, (x.shape[-1],), self.alpha, self.bias, self.eps)
   
 
class FeedForwardBlock(nn.Module):
    
    def __init__(self, d_model : int , d_ff : int, dropout : float) -> None:
        super().__init__()
        
        self.linear_1 = nn.Linear(d_model,d_ff) # matrix W1 and Bias B1 also bias default is true
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff,d_model) #matrix W2 and Bias B2 
    
    def forward(self,x):
        # input sentence which is ( Batch, seq_len, d_model) -->  1st linear ( Batch, seq_len, d_ff) --> then 2nd linear ( Batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
    
class MultiHeadAttentionBlock(nn.Module):
    
    def __init__(self, d_model:int, h:int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0,"d_model is not divisiable by h"
        
        # d_model must be divisiable by h ( no of heads )
        # d_model divide by h is d_k
        
        self.d_k = d_model // h
        
        self.w_q = nn.Linear(d_model,d_model) # Wq 
        self.w_k = nn.Linear(d_model,d_model) # Wk
        self.w_v = nn.Linear(d_model,d_model) #Wv
        
        self.w_o = nn.Linear(d_model , d_model) #Wo
        self.dropout = nn.Dropout(dropout)
        
    # @staticmethod # this means that u can call this function without having any instance of this class
    # def attention(query, key, value, mask, dropout: nn.Dropout):
    #     d_k = query.shape[-1]
        
    #     # (Batch, h, seq_len, d_k) --> ( Batch, h, seq_len,seq_len)
    #     attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        
    #     if mask is not None:
    #         attention_scores.masked_fill_(
    #             mask == 0,
    #             torch.finfo(attention_scores.dtype).min
    #         )
        
    #     attention_scores = attention_scores.softmax(dim = -1) # (Batch, h, seq_len, seq_len)
        
    #     if dropout is not None:
    #         attention_scores = dropout(attention_scores)
            
    #     return (attention_scores @ value), attention_scores
    #           # ^ this one is for next layer     ^  this one is for visulization
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        import torch.nn.functional as F
        
        # PyTorch 2.0+ Scaled Dot-Product Attention (FlashAttention)
        if mask is not None:
            mask = mask.bool() # SDPA expects boolean: True = attend, False = mask out
            
        out = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=mask,
            dropout_p=dropout.p if dropout is not None and dropout.training else 0.0
        )
        
        # Return dummy attention scores (None) since we don't need the heavy NxN matrix for training
        return out, None 
        
    def forward(self,q,k,v,mask): # query , key , value , mask -> if we want some words to not interact with other words we mask them
        query = self.w_q(q)   # batch , seq_len , d_model ---> batch , seq_len , d_model  ( same for all 3 )
        key = self.w_k(k)
        value = self.w_v(v)
          
        # ( Batch, seq_len, d_model) --> (Batch, seq_len, h, d_k) -> then transpose to ( Batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)   # h into d_k = d_model
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        
        x,self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # ( Batch, h, seq_len, d_k) -> ( Batch, seq_len, h, d_k ) --> ( Batch, seq_len, d_model )
        
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k ) 
        # for pytorch to transform the shape of a tensor need to put the memory to be contiguous to do it in place 
        # ( Batch, seq_len, d_model) --> ( Batch, seq_len, d_model )
        return self.w_o(x)
    
    
class ResidualConnection(nn.Module):   # skip connection between the add & norm and previous layer 
    
    def __init__(self, features: int, dropout:float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)
        
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
 

class EncoderBlock(nn.Module):
    
    def __init__(self, features: int, self_attention_block : MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout:float) -> None :
        super().__init__()       
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])
        
    def forward(self, x, src_mask) : # src _ mask is the mask we wanna apply to the input of the encoder because we want to hide the interaction of the padding word with another word
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
# we can have upto N number of encoder blocks so 

class Encoder(nn.Module):
    
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x) 

class DecoderBlock(nn.Module):
    
    def __init__(self, features: int, self_attention_block : MultiHeadAttentionBlock, cross_attention_block : MultiHeadAttentionBlock, feed_forward_block : FeedForwardBlock, dropout: float ) -> None :
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x : self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x : self.cross_attention_block(x, encoder_output,encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
        

class Decoder(nn.Module):
    
    def __init__(self, features: int, layers : nn.ModuleList ) -> None :
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module): # this layer converts the embeddings into to a position of the vocabulary 
    
    def __init__(self, d_model : int, vocab_size : int  ) -> None :
        super().__init__()
        self.proj = nn.Linear(d_model,vocab_size)
    
    def forward(self, x):
       # ( Batch, seq_len, d_model) converted into ( Batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1) # log softmax for numerical stability


class Transformer(nn.Module):
    
    def __init__(self, encoder:Encoder, decoder:Decoder, src_embed : InputEmbeddings, tgt_embed : InputEmbeddings, src_pos : PositionalEncoding, tgt_pos : PositionalEncoding, projection_layer : ProjectionLayer ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
    
    def encode(self, src, src_mask): # why different 3 methods instead of one forward method is during inference we can reuse the output of the encoder we dont need to recompute it everytime
        # also we prefer to keep this output separate for visualisation of the attention 
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src,src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)


def build_transformer(src_vocab_size : int, tgt_vocab_size : int , src_seq_len : int, tgt_seq_len : int, d_model : int = 512, N : int = 6, h : int = 8, dropout : float = 0.1, d_ff : int = 2048) -> Transformer:
    # create embedddings layer
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    
    # create the positional encoding layers
    
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # create the encoder blocks
    encoder_blocks = []
    
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
        
    
    # create decoder blocks
    
    decoder_blocks = []
    
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
           
    # create the encoder and the decoder
    
    # encoder = Encoder(nn.ModuleList(encoder_blocks))
    # decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # create the projection layer 
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # create the Transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer )
    
    # initialize the parameters ( using Xavier ) to make the Training faster so they dont just start with Random values
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer  

# tokeniser (word level) - its purpose is to split the words by space and then make a vocabulary of all the words where each word will mapped to one number 
# special tokens - padding tokens to complete the whole input size , SOS - start of sentence and EOS - end of sentence which are few imp special tokens needed to train the transformer    
                
    
    
    
        
        
                  
        
          
          
            
        
        
        
        
       
 
        
        
        
        
        
        
        
         
        
        
        
        
        
        
        
        
        
         
    
    