from itertools import product
from pathlib import Path
from typing import Any, Dict, Generic, OrderedDict, Sequence, TypeVar
from music_data_analysis import Pianoroll
import numpy as np
from torch import Tensor, nn
import torch
from tqdm import tqdm

from pianogen.data.vocab import Vocabulary
from pianogen.dataset.pianorolldataset import Sample
from pianogen.pe import binary_positional_encoding
from pianogen.tokenizer import PianoRollTokenizer, nucleus_sampling_batch

class FeatureLoader:
    '''
    Loads a feature from disk.
    '''
    def load(self, sample: Sample, pad_to: int) -> Any:
        raise NotImplementedError

T = TypeVar('T')
class Feature(nn.Module, Generic[T]):
    '''
    A feature has two representations: a natural representation and a embedded representation.
    The natural representation is some data structure that is native to humans.
    The embedded representation is a tensor that is native to neural networks. Their purpose is to condition other neural networks.
    '''

    def __init__(self, embed_size: int, condition_size: Dict[str, int]):
        '''Initialize the feature with the size of the embedding.

        Args:
            embed_size (int): The size of the embedding representing the feature.
            condition_size (Dict[str, int]): A map from condition names to the size of the condition embeddings.
        '''
        super().__init__()
        self._embed_size = embed_size
        self._condition_size = condition_size

    @property
    def embed_size(self) -> int:
        return self._embed_size

    def get_loader(self) -> FeatureLoader:
        '''Returen a loader that loads one sample of the feature from disk. Called in the dataset's __getitem__ function.
        The loader should not refer to the Feature object itself because it will be copied to multiple processes
        by PyTorch DataLoader.

        Returns:
            FeatureLoader: The loader that loads the feature from disk.
        '''
        raise NotImplementedError

    def calculate_loss(self, conditions: Dict[str, Tensor], gt: T) -> Tensor:
        '''Try to generate the ground truth data from the conditions and calculate the loss.
        Optimizing the loss should improve the generate() function.

        Args:
            conditions (Dict[str, Tensor]): A map from condition names to condition embeddings.
            gt (T): The ground truth data in natural representation.

        Raises:

        Returns:
            Tensor: The calculated loss.
        '''
        raise NotImplementedError
        
    def generate(self, conditions: Dict[str, Tensor]) -> T:
        '''Generate data in natural representation from the conditions. Sample if it's a probabilistic model.

        Args:
            conditions (Dict[str, Tensor]): A map from condition names to condition embeddings.

        Raises:

        Returns:
            T: The generated data in natural representation.
        '''
        raise NotImplementedError
    
    def embed(self, data: T) -> Tensor:
        '''Convert data in natural representation to a tensor that can natively condition other neural networks.
        The embedding should be big enough to contain all the information that is useful.

        Args:
            data (T): Data in natural representation of the feature.

        Raises:

        Returns:
            Tensor: The embedded data.
        '''
        raise NotImplementedError

class ScalarLoader(FeatureLoader):
    def __init__(self, file_name: str, time_granularity: int, vmin: float, vmax: float, num_classes: int):
        self.file_name = file_name
        self.time_granularity = time_granularity
        self.vmin = vmin
        self.vmax = vmax
        self.num_classes = num_classes

    def load(self, sample: Sample, pad_to: int) -> Tensor:
        '''
        Returns:
            Tensor: [L]
        '''
        scalars = torch.tensor(sample.get_feature_slice(self.file_name, self.time_granularity, pad_to=pad_to // self.time_granularity))
        
        indices = ((scalars - self.vmin) / (self.vmax - self.vmin) * self.num_classes).long()
        indices = torch.clamp(indices, 0, self.num_classes-1)

        return indices

class ScalarFeature(Feature[Tensor]):
    '''
    data_type: Tensor
    '''
    def __init__(self, condition_size: Dict[str,int], hidden_dim: int, file_name: str, time_granularity: int,
                    vmin: float, vmax: float, num_classes: int):
        super().__init__(1, condition_size)

        self.file_name = file_name
        self.granularity = time_granularity

        self.vmin = vmin
        self.vmax = vmax
        self.num_classes = num_classes


        cond_size = sum(condition_size.values()) # use all conditions
        
        self.classifier = nn.Sequential(
            nn.Linear(cond_size, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def get_loader(self):
        return ScalarLoader(self.file_name, self.granularity, self.vmin, self.vmax, self.num_classes)

    def get_logits(self, conditions: Dict[str, Any]) -> Tensor:
        condition = torch.cat([conditions[key] for key in conditions], dim=1) # [B, D]
        return self.classifier(condition) # [B, num_classes]

    def calculate_loss(self, conditions: Dict[str, Any], gt: Tensor) -> Tensor:
        logits = self.get_logits(conditions)
        return nn.CrossEntropyLoss()(logits, gt)
        
    def generate(self, conditions: Dict[str, Any]) -> Tensor:
        logits = self.get_logits(conditions) # [B, num_classes]
        return torch.argmax(logits, dim=1) # [B]
    
    def embed(self, data: Tensor) -> Tensor:
        return data.float().unsqueeze(-1) / (self.num_classes - 1) # normalize to [0, 1], [B, L, 1]

class ChordLoader(FeatureLoader):

    def __init__(self, vocab: Vocabulary):
        self.vocab = vocab

    def load(self, sample: Sample, pad_to: int) -> Tensor:
        '''
        Returns:
            Tensor: [L(bar), 4]
        '''
        chords = sample.get_feature_slice('chords', 8, pad_to=pad_to//8, pad_value='None')
        assert isinstance(chords, dict)
        chords = chords['name']
        indices = self.vocab.tokens_to_indices(chords) # [L*4]
        indices = indices.view(-1, 4) # [L, 4]
        return indices

    
class ChordFeature(Feature):
    
    def __init__(self, embed_size: int, condition_size: Dict[str, int], hidden_dim: int):
        super().__init__(embed_size, condition_size)
        qualities = ['M', 'm', 'o', '+', '7', 'M7', 'm7', 'o7', '/o7', 'sus2', 'sus4']
        roots = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        self.num_classes = len(qualities) * len(roots)
        
        chord_names = [f'{root}_{quality}' for root, quality in product(roots, qualities)]
        self.vocab = Vocabulary(['None']+chord_names) # None is when no note is played

        cond_size = sum(condition_size.values()) # use all conditions

        self.recurrent_embed = nn.Embedding(len(self.vocab), 32)

        self.classifier = nn.Sequential(
            nn.Linear(cond_size+32*4, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, len(self.vocab))
        )

        self.embedder = nn.Embedding(len(self.vocab), embed_size//4)
    
    def get_loader(self) -> FeatureLoader:
        return ChordLoader(self.vocab)
    
    def calculate_loss(self, conditions: Dict[str, Any], gt: Tensor) -> Tensor:
        '''
        conditions: Dict[str, Tensor] [B, D]
        gt: Tensor [B, L=4]
        '''
        B = gt.shape[0]
        embeds = self.recurrent_embed(gt[:,:3]) # B, 3, 32
        condition = torch.cat([conditions[key] for key in conditions], dim=1) # B, D
        
        inputs = []
        for i in range(4):
            # the input is the condition and the previous chords
            inputs.append(torch.cat([
                condition.unsqueeze(1), # B, 1, D
                embeds[:,:i].view(B, 1, -1), # B, 1, 32*i
                torch.zeros(B, 1, 32*(4-i)).to(condition.device) # B, 1, 32*(4-i)
            ], dim=2)) # B, 1, D+32*4
        inputs = torch.stack(inputs, dim=1) # B, L=4, D

        logits = self.classifier(inputs.view(B*4, -1)).view(B, 4, -1) # B, L=4, out

        return nn.CrossEntropyLoss()(logits.view(B*4, -1), gt.view(-1))
    
    def generate(self, conditions: Dict[str, Any]) -> Tensor:
        '''
        conditions: Dict[str, Tensor] [B, D]
        Returns:
            Tensor: [B, 4]
        '''
        B = next(iter(conditions.values())).shape[0]
        condition = torch.cat([conditions[key] for key in conditions], dim=1)
        
        results: list[Tensor] = [] # each is [B, 1]
        embeds: list[Tensor] = [] # each is [B, 32]
        
        for i in range(4):
            inp = torch.cat([
                condition, # B, D
                *embeds, # B, 32*i
                torch.zeros(B, 32*(4-i)).to(condition.device) # B, 32*(4-i)
            ], dim=1)

            logits = self.classifier(inp).view(B, -1) # B, out
            new_chord = nucleus_sampling_batch(logits, 0.9) # B
            results.append(new_chord)
            embeds.append(self.recurrent_embed(new_chord)) # B, 32

        return torch.stack(results, dim=1) # B, 4
    
    def embed(self, gt: Tensor) -> Tensor:
        '''

        Args:
            gt (Tensor): [B, L=4]

        Returns:
            Tensor: [B, D]
        '''
        B = gt.shape[0]
        return self.embedder(gt).view(B, -1) # [B, D]
        
class BinaryPositionalEncoding(nn.Module):
    '''
    Input: B, L (long)
    Output: B, L, D
    '''
    def __init__(self, dim:int, max_len:int):
        super().__init__()
        self.register_buffer('pos_encoding', binary_positional_encoding(max_len, dim).unsqueeze(0))

    def forward(self, pos: torch.Tensor):
        return torch.gather(self.pos_encoding.expand(pos.shape[0], -1, -1), 1, pos.unsqueeze(-1).expand(-1, -1, self.pos_encoding.shape[-1]))
    

class PianoRollLoader(FeatureLoader):
    def __init__(self, tokenizer: PianoRollTokenizer, token_per_bar:int):
        self.tokenizer = tokenizer
        self.token_per_bar = token_per_bar

    def load(self, sample: Sample, pad_to: int) -> Dict[str, Tensor]:
        '''
        Returns:
            Dict[str, Any]: A dictionary containing the following keys:
                indices: Tensor [L,self.token_per_bar]
                pos_frame: Tensor [L,self.token_per_bar]
                pos_pitch: Tensor [L,self.token_per_bar]
                output_mask: Tensor [L,self.token_per_bar2, len(tokenizer.vocab)]
        '''
        pr = sample.song.read_pianoroll('pianoroll').slice(sample.start, sample.end)
        all_tokens = self.tokenizer.tokenize(pr,
                                            token_per_bar=self.token_per_bar,
                                            token_seq_len=self.token_per_bar*(pad_to//32))

        indices_list = []
        pos_frame_list = []
        pos_pitch_list = []
        output_mask_list = []
        for i in range(0, len(all_tokens), self.token_per_bar): # each bar
            tokens = all_tokens[i:i+self.token_per_bar]
            indices = self.tokenizer.token_to_idx_seq(tokens)
            pos = self.tokenizer.get_frame_indices(tokens)
            pos_pitch = self.tokenizer.get_pitch_sequence(tokens)
            output_mask = self.tokenizer.get_output_mask([None] + tokens[:-1])
            
            indices_list.append(torch.tensor(indices, dtype=torch.long))
            pos_frame_list.append(pos)
            pos_pitch_list.append(pos_pitch)
            output_mask_list.append(output_mask)

        duration_in_bars = np.ceil((sample.end - sample.start) / 32)
        not_playing_bars = pad_to//32 - duration_in_bars
        playing_mask_list = [1] * int(duration_in_bars) + [0] * int(not_playing_bars)

        return {
            "indices": torch.stack(indices_list, dim=0),
            "pos_frame": torch.stack(pos_frame_list, dim=0),
            "pos_pitch": torch.stack(pos_pitch_list, dim=0),
            "output_mask": torch.stack(output_mask_list, dim=0),
            "playing_mask": torch.tensor(playing_mask_list, dtype=torch.int)
        }

class PianoRollFeature(Feature):
    def __init__(self, 
        embed_size: int,
        condition_size: Dict[str, int],
        token_per_bar:int=64,
        hidden_dim: int=256,
        num_layers: int=4,
        n_pitch: int=88,
        n_velocity: int=32,

        embed_num_layers: int=3,
        embed_hidden_dim: int=256,
    ):
        super().__init__(embed_size, condition_size)
        self.token_per_bar = token_per_bar
        self.n_pitch = n_pitch
        self.n_velocity = n_velocity
        self.tokenizer = PianoRollTokenizer(n_pitch=n_pitch, n_velocity=n_velocity)

        '''
        The embed size of each token is hidden_dim:
        1 for mark the token as start (the condition information is put in the start token)
        5 for the positional encoding of this token (zero for the start token)
        5 for the positional encoding of next token 
        1 for the pitch of this token (zero for the start token)
        hidden_dim - 12 for the token embedding or the condition content
        '''
        
        self.pe = BinaryPositionalEncoding(5, token_per_bar)

        # for prediction
        cond_size = sum(condition_size.values()) # use all conditions
        self.cond_adapter = nn.Linear(cond_size, hidden_dim - 12)
        self.input_embedding = nn.Embedding(len(self.tokenizer.vocab), hidden_dim - 12)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=512, batch_first=True),
            num_layers=num_layers
        )
        self.output_layer = nn.Linear(hidden_dim, len(self.tokenizer.vocab))

        self.crit = nn.CrossEntropyLoss(ignore_index=0)


        # for embedding
        self.embed_embedding = nn.Embedding(len(self.tokenizer.vocab), hidden_dim - 7)
        self.embed_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_hidden_dim, nhead=8, dim_feedforward=256, batch_first=True),
            num_layers=embed_num_layers
        )
        self.embed_output_layer = nn.Linear(hidden_dim, embed_size)
    
    def get_loader(self) -> FeatureLoader:
        return PianoRollLoader(self.tokenizer, self.token_per_bar)
    
    def get_logits(self, conditions: Dict[str, Any], indices: Tensor, pos_frame: Tensor, pos_pitch: Tensor) -> Tensor:
        '''
        indices: B, L-1
        pos_frame: B, L (0-31)
        pos_pitch: B, L-1 (0-self.n_pitch)
        '''
        
        condition = torch.cat([conditions[key] for key in conditions], dim=1) # B, D_cond
        start = self.cond_adapter(condition).unsqueeze(1) # B, 1, D-12

        # get the token embeddings. The last token is not needed for the input
        x = self.input_embedding(indices) # B, L-1, D-12

        # prepend the start token
        x = torch.cat([start, x], dim=1) # B, L, D-12

        # add the positional encoding
        pos_enc_this = self.pe(pos_frame[:,:-1]) # B, L-1, 5
        # pos_enc of start token is zero
        pos_enc_this = torch.nn.functional.pad(pos_enc_this, (0,0,1,0,0,0), value=0) # B, L, 5
        pos_enc_next = self.pe(pos_frame) # B, L, 5

        pos_enc_pitch = pos_pitch.unsqueeze(-1).float() / (self.n_pitch - 1) # B, L-1 , 1
        pos_enc_pitch = torch.nn.functional.pad(pos_enc_pitch, (0,0,1,0,0,0), value=0) # B, L, 1

        # start token mark
        start_token_mark = torch.zeros_like(x[:,:,0:1]) # B, L, 1
        start_token_mark[:,0] = 1 # the first token is the start token
        
        x = torch.cat([start_token_mark, pos_enc_this, pos_enc_next, pos_enc_pitch, x], dim=-1) # B, L, D

        x = self.transformer(x, mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1]), is_causal=True)

        x = self.output_layer(x) # B, L, out

        return x
    
    def calculate_loss(self, conditions: Dict[str, Any], gt: Dict[str, Any]) -> Tensor:
        logits = self.get_logits(conditions, gt['indices'][:,:-1], gt['pos_frame'], gt['pos_pitch'][:,:-1]) # B, L, out
        return self.crit((logits+gt['output_mask']).transpose(1,2), gt['indices'])
    
    def generate(self, conditions: Dict[str, Any]):
        device = next(self.parameters()).device
        B = next(iter(conditions.values())).shape[0]
        indices = torch.zeros(B,0,dtype=torch.long, device=device) # B, L=0
        pos_frame = torch.zeros(B,1,dtype=torch.long, device=device) # B, L=0
        pos_pitch = torch.zeros(B,0,dtype=torch.long, device=device) # B, L=0
    
        last_token = [None] * B

        max_length = 64 # maximum length
        for _ in range(max_length):
    
            logits = self.get_logits(conditions, indices, pos_frame, pos_pitch)[:,-1].detach().cpu()
            new_token = [self.tokenizer.sample_from_logits(logits[i], last_token[i], method='nucleus', p=0.9) for i in range(B)]
            last_token = new_token
    
            # update indices and pos
    
            new_pos_frame_batch = []
            for i in range(B):
                if pos_frame[i,-1] == 32:
                    new_token[i] = 'pad' # ignore the token if the frame is full
                if new_token[i] == 'next_frame':
                    new_pos_frame = pos_frame[i,-1] + 1
                else:
                    new_pos_frame = pos_frame[i,-1]
                new_pos_frame_batch.append(new_pos_frame)
            pos_frame = torch.cat([pos_frame, torch.tensor(new_pos_frame_batch).unsqueeze(1).to(device)], dim=-1)

            new_pos_pitch_batch = []
            for i in range(B):
                
                if isinstance(new_token[i], dict) and new_token[i]['type'] == 'pitch':
                    new_pos_pitch = new_token[i]['value']
                elif new_token[i] == 'next_frame':
                    new_pos_pitch = 0
                else:
                    if pos_pitch.shape[1] == 0:
                        new_pos_pitch = 0
                    else:
                        new_pos_pitch = pos_pitch[i,-1]
                new_pos_pitch_batch.append(new_pos_pitch)
            pos_pitch = torch.cat([pos_pitch, torch.tensor(new_pos_pitch_batch).unsqueeze(1).to(device)], dim=-1)

            new_token_idx = [self.tokenizer.token_to_idx(token) for token in new_token]
            indices = torch.cat([indices, torch.tensor(new_token_idx).unsqueeze(1).to(device)], dim=-1)

            # break if all new_pos_frame >= 32
            if all(x >= 32 for x in new_pos_frame_batch):
                break

        # pad to self.token_per_bar
        if indices.shape[1] < self.token_per_bar:
            pad_length = self.token_per_bar - indices.shape[1]
            indices = torch.cat([indices, 0+torch.zeros(B, pad_length, dtype=torch.long, device=device)], dim=-1)
            pos_frame = torch.cat([pos_frame, 32+torch.zeros(B, pad_length, dtype=torch.long, device=device)], dim=-1)
            pos_pitch = torch.cat([pos_pitch, 0+torch.zeros(B, pad_length, dtype=torch.long, device=device)], dim=-1)

        return {
            "indices": indices, # B, L
            "pos_frame": pos_frame[:,1:], # B, L
            "pos_pitch": pos_pitch
        }
    
    def embed(self, gt) -> Tensor:
        '''
        indices: B, L
        pos_frame: B, L (0-31)
        pos_token: B, L (0-self.n_pitch)
        '''

        indices = gt['indices']
        pos_frame = gt['pos_frame']
        pos_pitch = gt['pos_pitch']

        x = self.embed_embedding(indices) # B, L, D-7

        # add the positional encoding
        pos_enc_frame = self.pe(pos_frame) # B, L, 5
        pos_enc_pitch = pos_pitch.unsqueeze(-1).float() / (self.n_pitch - 1) # B, L-1 , 1
        x = torch.cat([pos_enc_frame, pos_enc_pitch, x], dim=-1) # B, L, D-1

        out_token = torch.zeros_like(x[:,0:1]) # B, 1, D-1
        x = torch.cat([out_token, x], dim=1) # B, L, D-1

        # add is_out mark on each token
        is_out = torch.zeros_like(x[:,:,0:1]) # B, L, 1
        is_out[:,0:1] = 1
        x = torch.cat([is_out, x], dim=-1) # B, L, D

        x = self.embed_transformer(x)

        x = self.embed_output_layer(x) # B, L, out

        # retrieve the first token (the out token)
        return x[:,0] # B, D


def sinusoidal_positional_encoding(length: int, dim: int):
    res = []
    for d in range(dim // 2):
        res.append(torch.sin(torch.arange(length) / 10000 ** (2 * d / dim)))
    for d in range(dim // 2):
        res.append(torch.cos(torch.arange(length) / 10000 ** (2 * d / dim)))
    if dim % 2:
        res.append(torch.zeros(length))
    return torch.stack(res, dim=1)

class Stem(nn.Module):
    '''A causal stem that propagates information from past inputs to future outputs.

    Args:
        
    '''
    def __init__(self, dim_input:int, dim_output:int, max_len:int, dim_model: int, num_layers: int, dim_feedforward: int, num_heads: int=8, dropout: float=0.1):
        super().__init__()
        self.input_layer = nn.Linear(dim_input, dim_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim_model, num_heads, dim_feedforward, dropout, batch_first=True),
            num_layers
        )
        self.output_layer = nn.Linear(dim_model, dim_output)
        # self.pe = sinusoidal_positional_encoding(max_len, dim_model)
        self.register_buffer('pe', sinusoidal_positional_encoding(max_len, dim_model))

    def forward(self, x: Tensor) -> Tensor:
        '''_summary_

        Args:
            x (Tensor): [B, L, D]

        Returns:
            Tensor: [B, L, D]
        '''
        x = self.input_layer(x) + self.pe[:x.shape[1]].unsqueeze(0)
        x = self.transformer(x, mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1]), is_causal=True)
        x = self.output_layer(x)
        return x

def all_same(items):
    return all(x == items[0] for x in items)

def concat_tensors_in_nested_dict(d: Sequence[dict]) -> dict[str, Tensor]:
    '''
    Concatenate the tensors in a list of nested dictionaries.
    '''
    res = {}
    for key in d[0]:
        if isinstance(d[0][key], dict):
            res[key] = concat_tensors_in_nested_dict([x[key] for x in d])
        else:
            res[key] = torch.cat([x[key] for x in d], dim=-1)

    return res
        


class Cake(nn.Module):
    '''Propagate information using the stem, then predict the features' values locally one by one.
    '''

    def __init__(self, a0_size, max_len:int, dim_model: int, num_layers: int, dim_feedforward: int, num_heads: int=8, dropout: float=0.1):
        super().__init__()

        self.stem_input_dim = 256+3+256
        self.a0_size = a0_size
        self.stem = Stem(self.stem_input_dim, a0_size, max_len, dim_model, num_layers, dim_feedforward, num_heads, dropout)

        condition_size = {'a0': a0_size}
        self.chord = ChordFeature(256, condition_size.copy(), 128)
        condition_size['chord'] = 256
        self.velocity = ScalarFeature(condition_size.copy(), dim_model, 'velocity', 32,
            vmin=0, vmax=128, num_classes=32)
        condition_size['velocity'] = 1
        self.polyphony = ScalarFeature(condition_size.copy(),  dim_model , 'polyphony', 32,
            vmin=0, vmax=8, num_classes=8)
        condition_size['polyphony'] = 1
        self.density = ScalarFeature(condition_size.copy(), dim_model , 'note_density', 32,
            vmin=0, vmax=32, num_classes=16)
        condition_size['density'] = 1
        self.piano_roll = PianoRollFeature(256, condition_size.copy(), 150)
        condition_size['piano_roll'] = 256

        self.features: OrderedDict[str, Feature] = OrderedDict(
            chord = self.chord,
            velocity = self.velocity,
            polyphony = self.polyphony,
            density = self.density,
            piano_roll = self.piano_roll
        )

    def calculate_loss(self, gt_feature: Dict[str, Tensor|Dict[str,Tensor]]) -> Dict[str, Tensor]:
        '''
        Args:
            gt_feature: Dict[str, List[Any]]: All the ground truth features. Each value is a list of ground truth data.
            The list contains data of each bar.
        '''
        # convert all features to embeddings
        
        gt_feature_merged = OrderedDict() # gt_feature, but the batch and bar dimensions are merged. [B*L, ...] or Dict[str, Tensor]: [B*L, ...]
        all_feature_embeds = OrderedDict() # the embeddings of all features. [B, L, D]

        for feature_name, feature_data in gt_feature.items():
            # merge dimension batch(B) and bar(L) for parallel run of Feature.embed()

            if isinstance(feature_data, dict):
                feature_data_merged = {}
                for key, value in feature_data.items():
                    B, L = value.shape[:2]
                    feature_data_merged[key] = value.view(B*L, *value.shape[2:])
            else:
                B, L = feature_data.shape[:2]
                feature_data_merged = feature_data.view(B*L, *feature_data.shape[2:])

            # cache for later use (loss calculation)
            gt_feature_merged[feature_name] = feature_data_merged

            # feature_data_merged: [B*L, ...] or Dict[str, Tensor]: [B*L, ...]

            all_feature_embeds[feature_name] = self.features[feature_name].embed(feature_data_merged).view(B, L, -1) # [B, L, D]

        all_feature_embeds_cat = torch.cat([all_feature_embeds[key] for key in all_feature_embeds], dim=2) # [B, L, D]
        a0 = self.stem(all_feature_embeds_cat) # [B, L, a0_size]

        # a0 is shifted right by one bar to avoid information leakage from the embedding of the current bar.
        # the a0 of the first bar is defined to be zero.

        a0 = torch.cat([torch.zeros_like(a0[:,:1]), a0[:,:-1]], dim=1) # [B, L, a0_size]

        # after the stem, all the features are predicted locally (bar-wise). Therefore, from now on,
        # the batch dimension and the bar dimension are merged.
        B, L = a0.shape[:2]

        a0 = a0.view(B*L, -1)

        # accumulate the loss of predicting each feature

        conditions = OrderedDict()
        losses = {}
        loss_mask = gt_feature_merged['piano_roll']['playing_mask']==1 # only calculate the loss of bars that are still playing
        conditions['a0'] = a0[loss_mask]
        for feature_name, feature_model in self.features.items():
            # calculate the loss of the feature based on the current conditions

            # conditions: Dict[str, Tensor]: [B*L, D]
            # gt_feature_merged[feature_name]: [B*L, ...] or Dict[str, Tensor]: [B*L, ...]
            if isinstance(gt_feature_merged[feature_name], dict):
                gt = {key: value[loss_mask] for key, value in gt_feature_merged[feature_name].items()}
            else:
                gt = gt_feature_merged[feature_name][loss_mask]
            losses[feature_name] = feature_model.calculate_loss(conditions, gt)

            # use the ground truth feature as condition for the following features
            conditions[feature_name] = all_feature_embeds[feature_name].view(B*L, -1)[loss_mask]

        losses['total'] = sum(losses.values())
        return losses

    
    def generate(self, n_bars: int, batch_size: int=1) -> Dict[str, Any]:
        device = next(self.parameters()).device
        history: Tensor = torch.zeros(batch_size, 0, self.stem_input_dim,device=device) # [B, L=0, D]

        generated_all = []

        for _ in tqdm(range(n_bars)):
            if history.shape[1] == 0:
                # a0 of the first bar is zero
                a0 = torch.zeros(batch_size, 1, self.a0_size, device=device) # [B, 1, a0_size]
            else:
                a0 = self.stem(history) # [B, L, a0_size]
    
            a0 = a0[:,-1] # [B, a0_size]

            # generate the features
            conditions = OrderedDict(a0=a0)
            generated = {}
            for feature_name, feature_model in self.features.items():
                generated[feature_name] = feature_model.generate(conditions)
                conditions[feature_name] = self.features[feature_name].embed(generated[feature_name])

            generated_all.append(generated)
            embeds = conditions.copy()
            embeds.pop('a0')

            embeds_cat = torch.cat([embeds[key] for key in embeds], dim=1) # [B, D]
            embeds_cat = embeds_cat.unsqueeze(1) # [B, 1, D]
            history = torch.cat([history, embeds_cat], dim=1) # [B, L+1, D]

        return concat_tensors_in_nested_dict(generated_all)
    
    def sample_and_save(self, save_path:str|Path, n_bars:int, batch_size:int=1):
        '''
        Sample a song and save it to the save_path.
        '''
        save_path = Path(save_path)
        generated = self.generate(n_bars, batch_size)
        batch = generated['piano_roll']['indices']
        for i, indices in enumerate(batch):
            tokens = self.piano_roll.tokenizer.idx_to_token_seq(indices.cpu().tolist())
            if batch_size == 1:
                song_save_path = save_path
            else:
                song_save_path = save_path.with_name(save_path.stem + f'_{i}.mid')

            pianoroll = Pianoroll([])
            for bar in range(n_bars):
                pr = self.piano_roll.tokenizer.detokenize(tokens[bar*self.piano_roll.token_per_bar:(bar+1)*self.piano_roll.token_per_bar])
                pianoroll += (pr >> pianoroll.duration)
            pianoroll.to_midi(song_save_path)

        print('Generated: ', tokens[:10])
        print('Chords: ', self.chord.vocab.indices_to_tokens(generated['chord'][0,:32].cpu()))
        return tokens
        