import torch
import torch.nn as nn
import os
import csv
import sentencepiece as spm
import time
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split



class Head(nn.Module):


    def __init__(self, n_embd, head_size, block_size, dropout):

        '''initialize three linear layers to transform token embeddings
        into unique vectors: query, key, and value... '''

        super().__init__()

        #Q K and V matrices...
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)

        #triangular masking for decoder blocks...
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        #dropout probability to combat overfitting...
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, mask = True):

        '''use query and key vectors to get attention weights, then
        apply to value vectors to get output. mask and droput as 
        necessary...'''

        #get batch keys and queries from linear layers...
        q = self.query(x)
        k = self.key(x)

        #compute attention scores ("affinities") & normalize wei with softmax......
        B,T,C = x.shape
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        if mask:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim = -1)

        #apply dropout...
        wei = self.dropout(wei)

        #perform weight aggregation of values...
        v = self.value(x)
        out = wei @ v
        return out



class MultiHeadAttention(nn.Module):

    
    def __init__(self, n_embd, num_heads, head_size, block_size, dropout):

        '''initialize multiple heads and a projection layer...'''

        super().__init__()

        #use the Head class to create a number of heads...
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for i in range(num_heads)])

        #projection layer to combine attention vectors...
        self.proj = nn.Linear(n_embd, n_embd)

        #dropout probability to combat overfitting...
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, mask = True):

        '''run multiple heads in parallel and combine outputs...'''

        #now out is back to the same size as the token embedding...
        out = torch.cat([h(x, mask) for h in self.heads], dim = -1)
        out = self.dropout(self.proj(out))
        return out



class FeedFoward(nn.Module):


    def __init__(self, n_embd, dropout):

        '''initialize an oveparameterized multilayer perceptron
        for rich transformations following mhsa layers...'''

        super().__init__()

        #simple MLP...
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )


    def forward(self, x):

        '''allow tokens to "think" on information gathered 
        during self-attention layer...'''

        return self.net(x)



class Block(nn.Module):


    def __init__(self, n_embd, n_head, block_size, dropout):

        '''combine the self attention layers with the feed 
        forward layers to create a repeatable block of layers...'''

        super().__init__()

        #embeddings are divided equally between the mhsa heads...
        self.head_size = n_embd // n_head

        #communication - computation...
        self.sa = MultiHeadAttention(n_embd, n_head, self.head_size, block_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)

        #normalization layers...
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)


    def forward(self, x, mask = True):

        '''run normalized multi-head attention layers 
        followed by normalized feed forward layers...'''

        #residual connections 'x + _' give processed data back to token embeddings...
        x = x + self.sa(self.ln1(x), mask)
        x = x + self.ffwd(self.ln2(x))
        return x



class Transformer(nn.Module):


    def __init__(self, memory, load_checkpoint = '', n_embd = 256, n_head = 8, n_layer = 6, dropout = 0.1, set_random_seed = True):
        
        '''use decoder/encoder blocks to create a model capable 
        of predicting the next word of a sequence given past context...'''

        super().__init__()
        
        #model size parameters...
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer

        #dropout probability...
        self.dropout = dropout

        #for reproducibility...
        if set_random_seed:
            torch.manual_seed(1)

        #fixed max input and output parameters so model can be quickly updated with new words...
        self.block_size = 8
        self.max_vocab = 160

        #unique embeddings (256 dimensional by default) for every word found in memory...
        self.token_embedding_table = nn.Embedding(self.max_vocab, self.n_embd)

        #unique embeddings (256 dimensional by default) for every position within the block size...
        self.position_embedding_table = nn.Embedding(self.block_size, self.n_embd)
        
        #communication - computaion blocks (3 by default)...
        self.blocks = nn.Sequential( * [Block(self.n_embd, 
                                              self.n_head, 
                                              self.block_size, 
                                              self.dropout) for i in range(self.n_layer)])

        #final normalization and output layer...
        self.ln_f = nn.LayerNorm(self.n_embd)
        self.final = nn.Linear(self.n_embd, self.max_vocab)

        #option to initialize transformer with saved weights...
        if load_checkpoint:
            self.load_state_dict(torch.load(os.path.join(load_checkpoint, "transformer_weights.pth")))
        
        #process memory...   
        spm.SentencePieceTrainer.Train(
            input = memory,
            model_prefix = 'chatbot',
            vocab_size = self.max_vocab,            
            model_type = 'bpe',            
            character_coverage = 1.0,      
            user_defined_symbols = ['<U>', '<B>', '<E>', '<S>'])
        self.memory_path = memory
        self.process_memory(self.memory_path)


    def forward(self, x, y = None):

        '''take in a sequence of words in tensor form 
        (sequence of integers) and output logits of next word...'''

        #get context dimensions....
        B, T = x.shape

        #assign character embeddings...
        self.tok_emb = self.token_embedding_table(x)
        self.pos_emb = self.position_embedding_table(torch.arange(T, device = 'cpu'))
        x = self.tok_emb + self.pos_emb

        #communication - computaion blocks...
        x = self.blocks(x)

        #final normalization layer...
        x = self.ln_f(x)

        #final linear layer...
        logits = self.final(x)

        #calculate and return loss...
        if y is None:
            loss = None
        else:
            logits = logits[:, -1, :]
            loss = F.cross_entropy(logits, y)
        return logits, loss
    

    def process_memory(self, memory, sp_model = 'chatbot.model'):

        '''process csv data "string-input, string-response"
        to produce training & validation datasets...'''

        #load sentencepiece model...
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model)

        #add special tokens to the tokenizer...
        special_tokens = ["<U>", "<B>", "<E>", "<S>"]
        for tok in special_tokens:
            if not self.sp.piece_to_id(tok):
                self.sp.set_piece_size(len(self.sp))
        
        #read csv...
        with open(memory, 'r', encoding = 'utf-8') as file:
            data = list(csv.reader(file))

        #add special tokens to chat-response pairs...
        seqs = []
        for tup in data:
            u = "<U> " + tup[0]
            r = " <B> " + tup[1] + " <E>"
            seqs.append(u + r)

        #encode sequences using sentencepiece...
        all_ids = [self.sp.encode(seq, out_type = int) for seq in seqs]

        #create X (context) and Y (next token) datasets...
        X, Y = [], []
        for seq_ids in all_ids:
            #start context with <S> token
            context = [self.sp.piece_to_id("<S>")] * self.block_size
            for idx in seq_ids:
                X.append(context)
                Y.append(idx)
                context = context[1:] + [idx]
        self.X = torch.tensor(X, dtype=torch.long)
        self.Y = torch.tensor(Y, dtype=torch.long)


    def train_model(self, checkpoint_path = '', epochs = 300, batch_size = 8, patience = 15,
                    val_split = 0.2, learning_rate = 0.0001, progress_interval = 5, 
                    subset_size = 150000, show_graph = False):
        
        '''includes per parameter adaptation, global learning 
        rate refinement, and an auto stop function...'''

        #memory is processed again so new words can be added without re-intialization...
        if self.process_memory(self.memory_path) == 'ERROR':
            return

        #split dataset...
        full_dataset = TensorDataset(self.X, self.Y)
        if subset_size < len(full_dataset):
            indices = torch.randperm(len(full_dataset))[:subset_size]
            dataset = torch.utils.data.Subset(full_dataset, indices)
        else:
            dataset = full_dataset
        tot = len(dataset)
        va_size = int(val_split * tot)
        tr_size = tot - va_size
        tr_dataset, va_dataset = random_split(dataset, (tr_size, va_size))

        #create dataloaders...
        tr_loader = DataLoader(tr_dataset, batch_size = batch_size, shuffle = True)
        va_loader = DataLoader(va_dataset, batch_size = batch_size, shuffle = False)       

        #per parameter adaptation and global learning rate refinement...
        optimizer = torch.optim.AdamW(self.parameters(), lr = learning_rate, weight_decay = 2e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode = 'min', factor = 0.2,
                                                               patience = 5, min_lr = 2e-7)
        print(f"\nUpdating {(sum(p.numel() for p in self.parameters())) / 1e6:.2f} M Parameters...")

        #training loop...
        tr_l, va_l = [], []
        best_l = float('inf')
        no_impr = 0

        #for every epoch...
        for epoch in range(epochs):

            #ensure normalization layers are using batch statistics...
            self.train()

            #for training updates...
            tr_loss, tr_steps = 0.0, 0
            va_loss, va_steps = 0.0, 0
            last_print_time = time.time()

            #train model on the training data...
            for Xb, Yb in tr_loader:

                #evaluate loss...
                logits, loss = self.forward(Xb, Yb)
                tr_loss += loss.item()
                tr_steps += 1
                
                #update parameters...
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #show progress...
                if time.time() - last_print_time > progress_interval:
                    print(f'Epoch {epoch + 1:>3} | Step {tr_steps:>6}/{len(tr_loader)} | TR Loss: {tr_loss / tr_steps:.5f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')
                    last_print_time = time.time()
            
            #record epoch training loss...
            tr_loss /= tr_steps
            tr_l.append(tr_loss)

            #get validation loss...
            self.eval()
            with torch.no_grad():
                for Xb, Yb in va_loader:

                    #evaluate loss...
                    logits, loss = self.forward(Xb, Yb)
                    va_loss += loss.item()
                    va_steps += 1

                    #show progress...
                    if time.time() - last_print_time > progress_interval:
                        print(f'Epoch {epoch + 1:>3} | Step {va_steps:>6}/{len(va_loader)} | VAL Loss: {va_loss / va_steps:.5f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')
                        last_print_time = time.time()
            
            #record epoch val loss...
            va_loss /= va_steps
            va_l.append(va_loss)
            scheduler.step(va_loss)

            #auto stop training...
            if va_loss < best_l:
                best_l = va_loss
                no_impr = 0
                if checkpoint_path:
                    torch.save(self.state_dict(), 
                               os.path.join(checkpoint_path, "transformer_weights.pth"))
            else:
                no_impr += 1
                if no_impr >= patience:
                    print("Training Complete.")
                    break

        #plot training loss and validation loss to visualize overfitting...
        if show_graph:
            plt.plot(tr_l, color = 'b', label = 'Training Loss')
            plt.plot(va_l, color = 'k', label = 'Validation Loss')
            plt.xlabel(f'Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()


    def respond(self, chat, max_len = 50, temperature = 1.0):
        
        '''produces a response from a string context...'''
        
        #turn initial chat into a valid input tensor...
        process = f'Input String: "{chat}"'
        chat_s = '<U> ' + chat
        chat_ids = self.sp.encode(chat_s, out_type = int)

        #pad list with start tokens if necessary...
        if len(chat_ids) < self.block_size:
            pad = [self.sp.piece_to_id('<S>')] * (self.block_size - len(chat_ids))
            chat_ids = pad + chat_ids
        else:
            chat_ids = chat_ids[-self.block_size:]
        chat_tensor = torch.tensor(chat_ids, dtype = torch.long).reshape(1, -1)
        
        #generate tokens...
        self.eval()
        pred_ids = []
        with torch.no_grad():
            for step in range(max_len):
                logits, _ = self.forward(chat_tensor)
                logits = logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim = -1)
                next_id = torch.multinomial(probs, num_samples = 1).item()
                next_piece = self.sp.id_to_piece(next_id)
                pred_ids.append(next_id)

                #update input window...
                chat_ids = chat_ids[1:] + [next_id]
                chat_tensor = torch.tensor(chat_ids, dtype = torch.long).reshape(1, -1)

                #update process log: show accumulated response so far...
                response_so_far = self.sp.decode(pred_ids)
                process += f'\nStep {step+1}: {response_so_far}'

                #stop at end token...
                if next_piece == '<E>':
                    break
            pred_ids.append(self.sp.piece_to_id('<E>'))

        #extract final response between <B> and <E>...
        try:
            start = pred_ids.index(self.sp.piece_to_id('<B>')) + 1
            end = pred_ids.index(self.sp.piece_to_id('<E>'))
            response_ids = pred_ids[start:end]
            response = self.sp.decode(response_ids)
        except ValueError:
            response = "Sorry, I couldn't understand that. Please train me more."
        return response, process
