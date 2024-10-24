from itertools import product
from typing import Any, Dict, Generic, List, OrderedDict, TypeVar
from torch import Tensor, nn
import torch
from tqdm import tqdm

from pianogen.data.vocab import Vocabulary
from pianogen.dataset.pianorolldataset import Sample
from pianogen.pe import binary_positional_encoding
from pianogen.tokenizer import PianoRollTokenizer

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
    def __init__(self, file_name: str, granularity: int):
        self.file_name = file_name
        self.granularity = granularity

    def load(self, sample: Sample, pad_to: int) -> Tensor:
        return torch.tensor(sample.get_feature_slice(self.file_name, self.granularity, pad_to=pad_to // self.granularity))

class ScalarFeature(Feature[Tensor]):
    '''
    data_type: Tensor
    '''
    def __init__(self, condition_size: Dict[str,int], num_classes: int, hidden_dim: int, file_name: str, granularity: int):
        super().__init__(1, condition_size)

        self.file_name = file_name
        self.granularity = granularity

        cond_size = sum(condition_size.values()) # use all conditions
        
        self.classifier = nn.Sequential(
            nn.Linear(cond_size, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def get_loader(self):
        return ScalarLoader(self.file_name, self.granularity)

    def get_logits(self, conditions: Dict[str, Any]) -> Tensor:
        condition = torch.cat([conditions[key] for key in conditions], dim=1)
        return self.classifier(condition)

    def calculate_loss(self, conditions: Dict[str, Any], gt: Tensor) -> Tensor:
        logits = self.get_logits(conditions)
        return nn.CrossEntropyLoss()(logits, gt)
        
    def generate(self, conditions: Dict[str, Any]) -> Tensor:
        logits = self.get_logits(conditions)
        return torch.argmax(logits, dim=1)
    
    def embed(self, data: Tensor) -> Tensor:
        return data

class ChordLoader(FeatureLoader):

    def __init__(self, vocab: Vocabulary):
        self.vocab = vocab

    def load(self, sample: Sample, pad_to: int) -> Tensor:
        chords = sample.get_feature_slice('chords', 8, pad_to=pad_to//8, pad_value='None')
        assert isinstance(chords, dict)
        chords = chords['name']
        return self.vocab.tokens_to_indices(chords)
class ChordFeature(Feature):
    
    def __init__(self, embed_size: int, condition_size: Dict[str, int], hidden_dim: int):
        super().__init__(embed_size, condition_size)
        qualities = ['M', 'm', 'o', '+', '7', 'M7', 'm7', 'o7', '/o7', 'sus2', 'sus4']
        roots = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        self.num_classes = len(qualities) * len(roots)
        
        chord_names = [f'{root}_{quality}' for root, quality in product(roots, qualities)]
        self.vocab = Vocabulary(['None']+chord_names) # None is when no note is played

        cond_size = sum(condition_size.values()) # use all conditions

        self.classifier = nn.Sequential(
            nn.Linear(cond_size, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, len(self.vocab))
        )

        self.embedder = nn.Embedding(len(self.vocab), embed_size)
    
    def get_loader(self) -> FeatureLoader:
        return ChordLoader(self.vocab)
    
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

    def load(self, sample: Sample, pad_to: int) -> Dict[str, Any]:
        pr = sample.song.read_pianoroll('pianoroll').slice(sample.start, sample.end)
        tokens = self.tokenizer.tokenize(pr, token_per_bar=self.token_per_bar, token_seq_len=self.token_per_bar*(pad_to//32))
        indices = self.tokenizer.token_to_idx_seq(tokens)
        pos = self.tokenizer.get_frame_indices(tokens)
        output_mask = self.tokenizer.get_output_mask(tokens[:-1])
        return {
            "indices": indices,
            "pos_frame": pos,
            "pos_pitch": torch.tensor([token['pitch'] for token in tokens]),
            "output_mask": output_mask,
        }

class PianoRollFeature(Feature):
    def __init__(self, embed_size: int, condition_size: Dict[str, int], token_per_bar:int=64, hidden_dim: int=256, num_layers: int=6, n_pitch: int=88, n_velocity: int=32):
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
        
        cond_size = sum(condition_size.values()) # use all conditions
        self.cond_adapter = nn.Linear(cond_size, hidden_dim - 12)
        self.input_embedding = nn.Embedding(len(self.tokenizer.vocab), hidden_dim - 12)
        self.pe = BinaryPositionalEncoding(5, token_per_bar)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=1024, batch_first=True),
            num_layers=num_layers
        )
        self.output_layer = nn.Linear(hidden_dim, len(self.tokenizer.vocab))

        self.crit = nn.CrossEntropyLoss(ignore_index=0)
    
    def get_loader(self) -> FeatureLoader:
        return PianoRollLoader(self.tokenizer, self.token_per_bar)
    
    def get_logits(self, conditions: Dict[str, Any], indices: Tensor, pos_frame: Tensor, pos_pitch: Tensor) -> Tensor:
        '''
        indices: B, L
        pos_frame: B, L (0-31)
        pos_token: B, L (0-self.n_pitch)
        '''
        
        condition = torch.cat([conditions[key] for key in conditions], dim=1)
        start = self.cond_adapter(condition) # B, 1, D-12

        # get the token embeddings. The last token is not needed for the input
        x = self.input_embedding(indices[:-1]) # B, L-1, D-12

        # prepend the start token
        x = torch.cat([start, x], dim=1) # B, L, D-12

        # add the positional encoding
        pos_enc_this = self.pe(pos_frame[:-1]) # B, L-1, 5
        # pos_enc of start token is zero
        pos_enc_this = torch.cat([torch.zeros_like(pos_enc_this[:,0:1]), pos_enc_this], dim=1) # B, L, 5
        pos_enc_next = self.pe(pos_frame) # B, L, 5

        pos_enc_pitch = pos_pitch.unsqueeze(-1).float() / (self.n_pitch - 1) # B, L-1 , 1
        pos_enc_pitch = torch.cat([torch.zeros_like(pos_enc_pitch[:,0:1]), pos_enc_pitch], dim=1) # B, L, 1

        # start token mark
        start_token_mark = torch.zeros_like(x[:,:,0:1]) # B, L, 1
        start_token_mark[:,0] = 1 # the first token is the start token
        
        x = torch.cat([start_token_mark, pos_enc_this, pos_enc_next, pos_enc_pitch, x], dim=-1) # B, L, D

        x = self.transformer(x, mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1]))

        x = self.output_layer(x) # B, L, out

        return x
    
    def calculate_loss(self, conditions: Dict[str, Any], gt: Dict[str, Any]) -> Tensor:
        logits = self.get_logits(conditions, gt['indices'], gt['pos_frame'], gt['pos_pitch'])
        return self.crit((logits+gt['output_mask']).transpose(1,2), gt['indices'])
    
    def generate(self, conditions: Dict[str, Any]) -> torch.Tensor:
        device = next(self.parameters()).device
        indices = torch.zeros(1,0,dtype=torch.long, device=device) # B=1, L=0
        pos_frame = torch.zeros(1,0,dtype=torch.long, device=device) # B=1, L=0
        pos_pitch = torch.zeros(1,0,dtype=torch.long, device=device) # B=1, L=0
        tokens = []
    
        last_token = None

        max_length = 64 # maximum length
        for _ in tqdm(range(max_length)):
    
            logits = self.get_logits(conditions, indices, pos_frame, pos_pitch).squeeze(0)[-1].detach().cpu()
            new_token = self.tokenizer.sample_from_logits(logits, last_token, method='nucleus', p=0.9)
            tokens.append(new_token)
            last_token = new_token
    
            # update indices and pos
    
            new_token_idx = self.tokenizer.vocab.get_idx(new_token)
            indices = torch.cat([indices, torch.tensor([[new_token_idx]]).to(device)], dim=-1)
            if new_token['type'] == 'next_frame':
                new_pos_frame = pos_frame[0,-1] + 1
            else:
                new_pos_frame = pos_frame[0,-1]
            pos_frame = torch.cat([pos_frame, torch.tensor([[new_pos_frame]]).to(device)], dim=-1)

            if new_token['type'] == 'pitch':
                new_pos_pitch = new_token['value']
            else:
                new_pos_pitch = pos_pitch[0,-1]
            pos_pitch = torch.cat([pos_pitch, torch.tensor([[new_pos_pitch]]).to(device)], dim=-1)
    
            if new_pos_frame >= 32:
                break

        return {
            "indices": indices,
            "pos_frame": pos_frame,
            "pos_pitch": pos_pitch,
            "tokens": tokens
        }
                                 
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
        self.pe = sinusoidal_positional_encoding(max_len, dim_model)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_layer(x) + self.pe[:x.shape[1]]
        x = self.transformer(x, mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1]))
        x = self.output_layer(x)
        return x

class Cake(nn.Module):
    '''Propagate information using the stem, then predict the features' values locally one by one.
    '''

    def __init__(self, a0_size, max_len:int, dim_model: int, num_layers: int, dim_feedforward: int, num_heads: int=8, dropout: float=0.1):
        super().__init__()

        self.stem = Stem(64+3+256, a0_size, max_len, dim_model, num_layers, dim_feedforward, num_heads, dropout)

        condition_size = {'a0': a0_size}
        self.chord = ChordFeature(64, condition_size.copy(), 128)
        condition_size['chord'] = 64
        self.velocity = ScalarFeature(condition_size.copy(), 128, dim_model, 'velocity', 32)
        condition_size['velocity'] = 1
        self.polyphony = ScalarFeature(condition_size.copy(), 128, dim_model , 'polyphony', 32)
        condition_size['polyphony'] = 1
        self.density = ScalarFeature(condition_size.copy(), 128, dim_model , 'note_density', 32)
        condition_size['density'] = 1
        self.piano_roll = PianoRollFeature(256, condition_size.copy(), 64)
        condition_size['piano_roll'] = 256

        self.features: OrderedDict[str, Feature] = OrderedDict(
            chord = self.chord,
            velocity = self.velocity,
            polyphony = self.polyphony,
            density = self.density
        )

    def calculate_loss(self, gt_feature: Dict[str, List[Any]]) -> Tensor:

        # convert all features to embeddings

        all_feature_embeds = {}
        for feature_name, feature_data in gt_feature.items():
            all_feature_embeds[feature_name] = self.features[feature_name].embed(feature_data)

        a0 = self.stem(all_feature_embeds)

        loss = 0
        conditions = {'a0': a0} # the global information

        for feature_name, feature_model in self.features.items():
            # calculate the loss of the feature based on the current conditions
            loss += feature_model.calculate_loss(conditions, gt_feature[feature_name])

            # use the ground truth feature as condition for the following features
            conditions[feature_name] = gt_feature[feature_name] 

        assert isinstance(loss, Tensor)
        return loss

            


