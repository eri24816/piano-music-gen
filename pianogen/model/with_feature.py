from itertools import product
from pathlib import Path
import tempfile
from typing import Any, Callable, Dict, Generic, List, Mapping, OrderedDict, TypeVar
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
    
    def to_human_representation(self, data: Tensor|dict[Tensor]) -> Any:
        raise NotImplementedError
    
    def to_tensor_representation(self, data: Any) -> Tensor|dict[Tensor]:
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
    
    def to_human_representation(self, indices: Tensor) -> List|float:
        '''
        Args:
            indices (Tensor): [L]

        Returns:
            Tensor: [L]
        '''
        return (indices.float() / self.num_classes * (self.vmax - self.vmin) + self.vmin).tolist()
    
    def to_tensor_representation(self, scalars: Tensor|List[float]|float|int) -> Tensor:
        '''
        Args:
            scalars (Tensor): [L]

        Returns:
            Tensor: [L]
        '''
        if not isinstance(scalars, Tensor):
            scalars = torch.tensor(scalars, dtype=torch.float)
        if len(scalars.shape) == 0:
            scalars = scalars.unsqueeze(0)
        indices = ((scalars - self.vmin) / (self.vmax - self.vmin) * self.num_classes).long()
        indices = torch.clamp(indices, 0, self.num_classes-1)
        return indices

class ScalarFeature(Feature[Tensor]):
    '''
    data_type: Tensor
    range: [vmin, vmax). Outside the range will be clamped.
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
        return data.float().unsqueeze(-1) / self.num_classes # normalize to [0, 1], [B, L, 1]

class ChordLoader(FeatureLoader):

    note_alias = {
        'Cb': 'B',
        'Db': 'C#',
        'Eb': 'D#',
        'Fb': 'E',
        'Gb': 'F#',
        'Ab': 'G#',
        'Bb': 'A#'
    }

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
    
    def to_human_representation(self, indices: Tensor) -> list[str]:
        '''
        Args:
            indices (Tensor): [L, 4]

        Returns:
            list[str]: [L]
        '''

        def humanize_chord_name(name:'str') -> 'str':
            if name == 'None':
                return 'None'
            root, quality = name.split('_')
            if quality == 'M':
                quality = ''
            return root + quality

        result = [self.vocab.indices_to_tokens(bar) for bar in indices]
        for i, result_bar in enumerate(result):
            if result_bar[0] == result_bar[1] == result_bar[2] == result_bar[3]:
                result[i] = [result_bar[0]]
            elif result_bar[0] == result_bar[1] and result_bar[2] == result_bar[3]:
                result[i] = [result_bar[0], result_bar[2]]

            result[i] = ' '.join([humanize_chord_name(name) for name in result[i]])
                                                               
        return result
    
    def to_tensor_representation(self, chords: list[str]) -> Tensor:
        '''
        Args:
            chords (list[str]): [L]

        Returns:
            Tensor: [L, 4]
        '''

        def dehumanize_chord_name(name:'str') -> 'str':
            if name == 'None':
                return 'None'
            root_len = 2 if len(name) > 1 and name[1] in ['b', '#'] else 1
            root = name[:root_len]
            quality = name[root_len:]
            if quality == '':
                quality = 'M'

            root = root[0].upper() + root[1:]
            if len(root) == 2 and root[1] == 'b':
                root = self.note_alias[root]
            return root + '_' + quality

        indices = []
        for bar in chords:
            bar = bar.split()
            bar = [dehumanize_chord_name(chord) for chord in bar]
            if len(bar) == 1:
                bar = [bar[0], bar[0], bar[0], bar[0]]
            elif len(bar) == 2:
                bar = [bar[0], bar[0], bar[1], bar[1]]
            elif len(bar) == 3:
                bar = [bar[0], bar[1], bar[2], bar[2]]
            indices.append(self.vocab.tokens_to_indices(bar))
        return torch.stack(indices, dim=0)

    
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
        self.cache_path = Path(tempfile.mkdtemp())

    def cache(self, sample: Sample, pad_to: int):
        
        cache_file = self.cache_path / f'{sample.song.song_name}_{sample.start}_{sample.end}.pt'

        if cache_file.exists():
            return torch.load(cache_file)

        pr = sample.song.read_pianoroll('pianoroll').slice(sample.start, sample.end)
        all_tokens = self.tokenizer.tokenize(pr,
                                            token_per_bar=self.token_per_bar,
                                            token_seq_len=self.token_per_bar*(pad_to//32))
        
        indices_list = []
        pos_frame_list = []
        pos_pitch_list = []
        for i in range(0, len(all_tokens), self.token_per_bar): # each bar
            tokens = all_tokens[i:i+self.token_per_bar]
            indices = self.tokenizer.token_to_idx_seq(tokens)
            pos = self.tokenizer.get_frame_indices(tokens)
            pos_pitch = self.tokenizer.get_pitch_sequence(tokens)
            
            indices_list.append(torch.tensor(indices, dtype=torch.long))
            pos_frame_list.append(pos)
            pos_pitch_list.append(pos_pitch)
        
        duration_in_bars = np.ceil((sample.end - sample.start) / 32)
        not_playing_bars = pad_to//32 - duration_in_bars
        playing_mask_list = [1] * int(duration_in_bars) + [0] * int(not_playing_bars)
        
        cache_data = {
            "indices": torch.stack(indices_list, dim=0),
            "pos_frame": torch.stack(pos_frame_list, dim=0),
            "pos_pitch": torch.stack(pos_pitch_list, dim=0),
            "playing_mask": torch.tensor(playing_mask_list, dtype=torch.int),
            "tokens": all_tokens
        }

        self.cache_path.mkdir(exist_ok=True)
        torch.save(cache_data, self.cache_path / f'{sample.song.song_name}_{sample.start}_{sample.end}.pt')

        return cache_data

    def load(self, sample: Sample, pad_to: int) -> Dict[str, Tensor]:
        '''
        Returns:
            Dict[str, Any]: A dictionary containing the following keys:
                indices: Tensor [L,self.token_per_bar]
                pos_frame: Tensor [L,self.token_per_bar]
                pos_pitch: Tensor [L,self.token_per_bar]
                output_mask: Tensor [L,self.token_per_bar2, len(tokenizer.vocab)]
        '''
        data = self.cache(sample, pad_to)

        output_mask_list = []
        for i in range(0, len(data['tokens']), self.token_per_bar): # each bar
            tokens = data['tokens'][i:i+self.token_per_bar]
            output_mask = self.tokenizer.get_output_mask([None] + tokens[:-1])
            output_mask_list.append(output_mask)

        data['output_mask'] = torch.stack(output_mask_list, dim=0)
        data.pop('tokens')
        return data
    
    def to_human_representation(self, data: Dict[str, Tensor]) -> Pianoroll:
        '''
        Args:
            data (Dict[str, Tensor]): A dictionary containing the following keys:
                indices: Tensor [L,self.token_per_bar]
                ...

        Returns:
            Pianoroll: A pianoroll object
        '''
        tokens = self.tokenizer.idx_to_token_seq(data['indices'].flatten().tolist())
        n_bars = data['indices'].shape[0]
        pianoroll = Pianoroll([])
        for bar in range(n_bars):
            pr = self.tokenizer.detokenize(tokens[bar*self.token_per_bar:(bar+1)*self.token_per_bar])
            pianoroll += (pr >> bar*32)

        return pianoroll
    
class PianoRollFeature(Feature):
    def __init__(self, 
        embed_size: int,
        condition_size: Dict[str, int],
        token_per_bar:int=64,
        hidden_dim: int=384,
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
        self.embed_embedding = nn.Embedding(len(self.tokenizer.vocab), embed_hidden_dim - 7)
        self.embed_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_hidden_dim, nhead=8, dim_feedforward=256, batch_first=True),
            num_layers=embed_num_layers
        )
        self.embed_output_layer = nn.Linear(embed_hidden_dim, embed_size)
    
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

        for _ in range(self.token_per_bar):
    
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
    def __init__(self, dim_input:int, max_len:int, dim_model: int, num_layers: int, dim_feedforward: int, num_heads: int=8, dropout: float=0.1):
        super().__init__()
        self.input_layer = nn.Linear(dim_input, dim_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim_model, num_heads, dim_feedforward, dropout, batch_first=True),
            num_layers
        )
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
        return x

def all_same(items):
    return all(x == items[0] for x in items)

def apply_tensors_in_nested_dict(d: Tensor|Mapping[str, Tensor|dict[str,Tensor]], f: Callable[[Tensor], Tensor]) -> Tensor|dict[str, Tensor]:
    '''
    Apply a function to all tensors in a nested dictionary.
    '''
    res = {}
    if isinstance(d, Tensor):
        return f(d)
    for k, v in d.items():
        if isinstance(v, dict):
            res[k] = apply_tensors_in_nested_dict(v, f)
        else:
            res[k] = f(v)
    return res

def concat_tensors_in_nested_dict(d: list[dict]|list[Tensor], dim:int, drop_missing=False) -> dict[str, Tensor]|Tensor:
    '''
    Concatenate the tensors in a list of nested dictionaries.
    '''
    res = {}
    if isinstance(d[0], Tensor):
        d: list[Tensor]
        return torch.cat(d, dim=dim)
    for k in d[0].keys():
        if drop_missing and any(k not in x for x in d):
            continue
        res[k] = concat_tensors_in_nested_dict([x[k] for x in d], dim, drop_missing)

    return res


class Cake(nn.Module):
    '''Propagate information using the stem, then predict the features' values locally one by one.
    '''

    def __init__(self, a0_size, max_len:int, dim_model: int, num_layers: int, dim_feedforward: int, num_heads: int=8, dropout: float=0.1):
        super().__init__()

        self.stem_input_dim = 256+3+256
        self.a0_size = a0_size
        self.stem = Stem(self.stem_input_dim,  max_len,a0_size, num_layers, dim_feedforward, num_heads, dropout)

        condition_size = {'a0': a0_size}
        self.chord = ChordFeature(256, condition_size.copy(), 256)
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

        self.first_bar_a0 = nn.Parameter(torch.zeros(a0_size))
        # initialize the first bar a0 using xavier initialization
        nn.init.uniform_(self.first_bar_a0, -1, 1)

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
        # the a0 of the first bar is defined by the first_bar_a0 parameter.

        a0 = torch.cat([
            self.first_bar_a0.view(1, 1, -1).expand(a0.shape[0], 1, -1), # [B, 1, a0_size]
            a0[:,:-1]
        ], dim=1) # [B, L, a0_size]

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

    
    def generate(self, n_bars: int, batch_size: int=1,
        prompt:Mapping[str, Tensor|Dict[str,Tensor]]|None = None,
        prompt_bars:int=0, given_features:dict|None=None
    ) -> Dict[str, Any]:
        '''
        Generate a song with n_bars.
        Each dict value in prompt is [L, ...] or Dict[str, Tensor]: [L, ...]
        '''
        device = next(self.parameters()).device
        if prompt is not None and prompt_bars == 0:
            assert isinstance(prompt['piano_roll'], dict)
            prompt_bars = int(prompt['piano_roll']['playing_mask'].sum().item())

        if prompt is not None:
            # convert the prompt to embeddings
            prompt_embeds = OrderedDict()
            prompt = apply_tensors_in_nested_dict(prompt, lambda x: x[: prompt_bars].to(device)) # truncate the prompt to prompt_bars
            # prompt: nested dict of [L, ...]
            
            for name, feature_data in prompt.items():
                prompt_embeds[name] = self.features[name].embed(feature_data).unsqueeze(0).expand(batch_size, -1, -1) # [1, L, D]

            history = torch.cat([prompt_embeds[key] for key in prompt_embeds], dim=2) # [B, L, D]

            generated_all = [apply_tensors_in_nested_dict(prompt, lambda x: x.unsqueeze(0).expand(batch_size, *([-1]*len(x.shape))))] # [B, L, ...]

        else:
            history: Tensor = torch.zeros(batch_size, 0, self.stem_input_dim,device=device) # [B, L=0, D]

            generated_all = []

        for _ in tqdm(range(prompt_bars, n_bars)):
            if history.shape[1] == 0:
                # a0 of the first bar is zero
                a0 = self.first_bar_a0.view(1, -1).expand(batch_size, -1) # [B, a0_size]
            else:
                a0 = self.stem(history)[:,-1] # [B, a0_size]

            # generate the features
            conditions = OrderedDict(a0=a0)
            generated = {}
            for name, model in self.features.items():
                print(name, given_features, name in given_features)
                if given_features is not None and name in given_features:
                    human_rep = [given_features[name]] # The list means bar dimension
                    feature_generated = self.features[name].get_loader().to_tensor_representation(human_rep).to(device)
                else:
                    feature_generated = model.generate(conditions) # [B, ...]
                # print(self.features[name].get_loader().to_human_representation(feature_generated))
                generated[name] = apply_tensors_in_nested_dict(feature_generated, lambda x: x.unsqueeze(1)) # [B, 1, ...]
                conditions[name] = self.features[name].embed(feature_generated) # [B, D]

            generated_all.append(generated)
            embeds = conditions.copy()
            embeds.pop('a0')

            embeds_cat = torch.cat([embeds[key] for key in embeds], dim=1) # [B, D]
            embeds_cat = embeds_cat.unsqueeze(1) # [B, 1, D]
            history = torch.cat([history, embeds_cat], dim=1) # [B, L+1, D]

        return concat_tensors_in_nested_dict(generated_all, dim=1, drop_missing=True) # concat L dimension
    
    def indices_to_pianoroll(self, indices: Tensor) -> Pianoroll:
        '''
        indices: [L, self.token_per_bar] where L is bar dimension
        '''
        n_bars = indices.shape[0]
        tokens = self.piano_roll.tokenizer.idx_to_token_seq(indices.view(-1).cpu().tolist())
        
        pianoroll = Pianoroll([])
        for bar in range(n_bars):
            pr = self.piano_roll.tokenizer.detokenize(tokens[bar*self.piano_roll.token_per_bar:(bar+1)*self.piano_roll.token_per_bar])
            pianoroll += (pr >> bar*32)

        return pianoroll

    def sample_and_save(self, save_path:str|Path, n_bars:int, batch_size:int=1, prompt:Mapping[str, Tensor|Dict[str,Tensor]]|None=None, prompt_bars:int=0):
        '''
        Sample a song and save it to the save_path.
        '''
        save_path = Path(save_path)
        generated = self.generate(n_bars, batch_size, prompt, prompt_bars)
        batch = generated['piano_roll']['indices']
        for i, indices in enumerate(batch):
            pianoroll = self.indices_to_pianoroll(indices)
                
            if batch_size == 1:
                song_save_path = save_path
            else:
                song_save_path = save_path.with_name(save_path.stem + f'_{i}.mid')
            pianoroll.to_midi(song_save_path)

        # print('Generated: ', tokens[:10])
        # print('Chords: ', self.chord.vocab.indices_to_tokens(generated['chord'].view(B,-1)[0,:32].cpu()))
        
    def get_loaders(self) -> dict[str, FeatureLoader]:
        return {key: value.get_loader() for key, value in self.features.items()}