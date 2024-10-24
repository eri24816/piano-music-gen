from itertools import product
from typing import Any, Dict, Generic, OrderedDict, TypeVar
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
        condition = torch.cat([conditions[key] for key in conditions], dim=1)
        return self.classifier(condition)

    def calculate_loss(self, conditions: Dict[str, Any], gt: Tensor) -> Tensor:
        logits = self.get_logits(conditions)
        return nn.CrossEntropyLoss()(logits, gt)
        
    def generate(self, conditions: Dict[str, Any]) -> Tensor:
        logits = self.get_logits(conditions)
        return torch.argmax(logits, dim=1)
    
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
            nn.Linear(hidden_dim, len(self.vocab)*4)
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
        '''
        B = next(iter(conditions.values())).shape[0]
        condition = torch.cat([conditions[key] for key in conditions], dim=1)
        
        results: list[Tensor] = [] # each is [B, 1]
        embeds: list[Tensor] = [] # each is [B, 1, 32]

        for i in range(4):
            inp = torch.cat([
                condition, # B, D
                *embeds, # B, 32*i
                torch.zeros(B, 32*(4-i)).to(condition.device) # B, 32*(4-i)
            ], dim=1)

            logits = self.classifier(inp).view(B, -1) # B, out
            new_chord = nucleus_sampling_batch(logits, 0.9) # B
            results.append(new_chord)
            embeds.append(self.recurrent_embed(new_chord).unsqueeze(1)) # B, 1, 32

        return torch.stack(results, dim=1)
    
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
                output_mask: Tensor [L,3self.token_per_bar2, len(tokenizer.vocab)]
        '''
        pr = sample.song.read_pianoroll('pianoroll').slice(sample.start, sample.end)
        all_tokens = self.tokenizer.tokenize(pr,
                                            token_per_bar=self.token_per_bar,
                                            token_seq_len=self.token_per_bar*(pad_to//32))

        #result:list[Dict[str, Any]] = []
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

        return {
            "indices": torch.stack(indices_list, dim=0),
            "pos_frame": torch.stack(pos_frame_list, dim=0),
            "pos_pitch": torch.stack(pos_pitch_list, dim=0),
            "output_mask": torch.stack(output_mask_list, dim=0)
        }

class PianoRollFeature(Feature):
    def __init__(self, 
        embed_size: int,
        condition_size: Dict[str, int],
        token_per_bar:int=64,
        hidden_dim: int=256,
        num_layers: int=6,
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
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=1024, batch_first=True),
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
        indices: B, L
        pos_frame: B, L (0-31)
        pos_pitch: B, L (0-self.n_pitch)
        '''
        
        condition = torch.cat([conditions[key] for key in conditions], dim=1) # B, D_cond
        start = self.cond_adapter(condition).unsqueeze(1) # B, 1, D-12

        # get the token embeddings. The last token is not needed for the input
        x = self.input_embedding(indices[:,:-1]) # B, L-1, D-12

        # prepend the start token
        x = torch.cat([start, x], dim=1) # B, L, D-12

        # add the positional encoding
        pos_enc_this = self.pe(pos_frame[:,:-1]) # B, L-1, 5
        # pos_enc of start token is zero
        pos_enc_this = torch.cat([torch.zeros_like(pos_enc_this[:,0:1]), pos_enc_this], dim=1) # B, L, 5
        pos_enc_next = self.pe(pos_frame) # B, L, 5

        pos_enc_pitch = pos_pitch[:,:-1].unsqueeze(-1).float() / (self.n_pitch - 1) # B, L-1 , 1
        pos_enc_pitch = torch.cat([torch.zeros_like(pos_enc_pitch[:,0:1]), pos_enc_pitch], dim=1) # B, L, 1

        # start token mark
        start_token_mark = torch.zeros_like(x[:,:,0:1]) # B, L, 1
        start_token_mark[:,0] = 1 # the first token is the start token
        
        x = torch.cat([start_token_mark, pos_enc_this, pos_enc_next, pos_enc_pitch, x], dim=-1) # B, L, D

        x = self.transformer(x, mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1]))

        x = self.output_layer(x) # B, L, out

        return x
    
    def calculate_loss(self, conditions: Dict[str, Any], gt: Dict[str, Any]) -> Tensor:
        logits = self.get_logits(conditions, gt['indices'], gt['pos_frame'], gt['pos_pitch']) # B, L, out
        return self.crit((logits+gt['output_mask']).transpose(1,2), gt['indices'])
    
    def generate(self, conditions: Dict[str, Any]):
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

        return x


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
        self.register_buffer('pe', sinusoidal_positional_encoding(max_len, dim_model).unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        '''_summary_

        Args:
            x (Tensor): [B, L, D]

        Returns:
            Tensor: [B, L, D]
        '''
        print(x.shape)
        x = self.input_layer(x) + self.pe[:x.shape[1]]
        x = self.transformer(x, mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1]))
        x = self.output_layer(x)
        return x

def all_same(items):
    return all(x == items[0] for x in items)

class Cake(nn.Module):
    '''Propagate information using the stem, then predict the features' values locally one by one.
    '''

    def __init__(self, a0_size, max_len:int, dim_model: int, num_layers: int, dim_feedforward: int, num_heads: int=8, dropout: float=0.1):
        super().__init__()

        self.stem = Stem(256+3+256, a0_size, max_len, dim_model, num_layers, dim_feedforward, num_heads, dropout)

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
        self.piano_roll = PianoRollFeature(256, condition_size.copy(), 64)
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
        
        gt_feature_merged = {} # gt_feature, but the batch and bar dimensions are merged. [B*L, ...] or Dict[str, Tensor]: [B*L, ...]
        all_feature_embeds = {} # the embeddings of all features. [B, L, D]

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

        for key, value in all_feature_embeds.items():
            print(key, value.shape)
        all_feature_embeds_cat = torch.cat([all_feature_embeds[key] for key in all_feature_embeds], dim=2) # [B, L, D]
        a0 = self.stem(all_feature_embeds_cat) # [B, L, a0_size]

        # after the stem, all the features are predicted locally (bar-wise). So we can again merge the batch and bar dimensions
        B, L = a0.shape[:2]

        a0 = a0.view(B*L, -1)

        # accumulate the loss of predicting each feature

        conditions = {'a0': a0} # the global information

        losses = {}
        for feature_name, feature_model in self.features.items():
            # calculate the loss of the feature based on the current conditions

            # conditions: Dict[str, Tensor]: [B*L, D]
            # gt_feature_merged[feature_name]: [B*L, ...] or Dict[str, Tensor]: [B*L, ...]
            losses[feature_name] = feature_model.calculate_loss(conditions, gt_feature_merged[feature_name])

            # use the ground truth feature as condition for the following features
            conditions[feature_name] = all_feature_embeds[feature_name].view(B*L, -1)

        losses['total'] = sum(losses.values())
        return losses

    

