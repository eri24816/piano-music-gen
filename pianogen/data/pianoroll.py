from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Generator, Tuple
from matplotlib import pyplot as plt
import numpy as np
from math import ceil
import miditoolkit.midi.parser
import json

INF = 2147483647

HAS_TORCH = False
try:
    import torch

    HAS_TORCH = True
except ImportError:
    pass


def json_load(f):
    return json.load(open(f, "r"))


def json_dump(obj, f):
    json.dump(obj, open(f, "w"))


class Note:
    def __init__(self, onset, pitch, velocity, offset=None) -> None:
        self.onset = onset
        self.pitch = pitch
        self.velocity = velocity
        self.offset = offset

    def __repr__(self) -> str:
        return f"Note({self.onset},{self.pitch},{self.velocity},{self.offset})"

    def __gt__(self, other):
        if self.onset == other.onset:
            return self.pitch > other.pitch
        return self.onset > other.onset


scale = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
quality = {
    "": [0, 4, 7],
    "m": [0, 3, 7],
    "7": [0, 4, 7, 10, 10],  # the 7th note is important
    "m7": [0, 3, 7, 10, 10],
}
chord_to_chroma = {}
for i, s in enumerate(scale):
    for q in quality:
        chord_to_chroma[f"{s}{q}"] = [(x + i) % 12 for x in quality[q]]


def chroma_to_chord(query):
    scores: dict[str, float] = {}
    for chord, chroma in chord_to_chroma.items():
        score = 0
        for i in chroma:
            score += query[i]
        scores[chord] = score / len(chroma)
    argmax = max(scores, key=scores.__getitem__)
    return argmax


@dataclass
class PRMetadata:
    name: str = ""
    start_time: int = 0
    end_time: int = 0


class PianoRoll:
    """ """

    @staticmethod
    def load(path):
        """
        Load a pianoroll from a json file
        """
        if not isinstance(path, Path):
            path = Path(path)
        pr = PianoRoll(json_load(path))
        pr.set_metadata(name=path.stem)
        return pr

    @staticmethod
    def from_tensor(tens: "torch.Tensor", thres=5, normalized=False):
        """
        Convert a tensor to a pianoroll
        """
        if not HAS_TORCH:
            raise ImportError(
                "PianoRoll.from_tensor requires torch. Please install torch."
            )
        if normalized:
            tens = (tens + 1) * 64
        tens = tens.cpu().to(torch.int32).clamp(0, 127)
        data = {"onset_events": [], "pedal_events": []}
        for t in range(tens.shape[0]):
            for p in range(tens.shape[1]):
                if tens[t, p] > thres:
                    data["onset_events"].append([t, p + 21, int(tens[t, p])])

        return PianoRoll(data)

    @staticmethod
    def from_midi(path):
        """
        Load a pianoroll from a midi file
        """
        midi = miditoolkit.midi.parser.MidiFile(path)
        data = {"onset_events": [], "pedal_events": []}
        if len(midi.instruments) > 0:
            for note in midi.instruments[0].notes:
                note: miditoolkit.Note
                data["onset_events"].append(
                    [
                        int(note.start * 8 / midi.ticks_per_beat),
                        note.pitch,
                        note.velocity,
                        int(note.end * 8 / midi.ticks_per_beat),
                    ]
                )
        pr = PianoRoll(data)
        pr.set_metadata(name=path.split("/")[-1].split(".mid")[0])
        return pr

    def __init__(self, data: dict | list[Note]):
        # [onset time, pitch, velocity]
        if isinstance(data, dict):
            self.notes = [Note(*note) for note in data["onset_events"]]
            self.notes = sorted(self.notes)  # ensure the event is sorted by time

            if "pedal_events" in data:
                # Timestamps of pedal up events. (Otherwise the pedal is always down)
                self.pedal = data["pedal_events"]
                self.pedal = sorted(self.pedal)
            else:
                self.pedal = None

        else:
            self.notes = data
            self.pedal = None

        if len(self.notes):
            if self.notes[-1].offset is None:
                self.duration = ceil((self.notes[-1].onset + 1) / 32) * 32
            else:
                self.duration = ceil((self.notes[-1].offset) / 32) * 32
        else:
            self.duration = 0

        self._have_offset = len(self.notes) == 0 or self.notes[0].offset is not None

        self.metadata = PRMetadata("", 0, self.duration)

    def __repr__(self) -> str:
        return f"PianoRoll Bar {self.metadata.start_time//32:03d} - {ceil(self.metadata.end_time/32):03d} of {self.metadata.name}"

    """
    ==================
    Utils
    ==================
    """

    def set_metadata(self, name=None, start_time=None, end_time=None):
        if name is not None:
            self.metadata.name = name
        if start_time is not None:
            self.metadata.start_time = start_time
        if end_time is not None:
            self.metadata.end_time = end_time

    def iter_over_notes_unpack(self, notes=None):
        """
        generator that yields (onset, pitch, velocity, offset iterator)
        """
        if notes is None:
            notes = self.notes
        for note in notes:
            yield note.onset, note.pitch, note.velocity, note.offset

    def iter_over_bars_unpack(
        self, bar_length=32
    ) -> Generator[list[Tuple[int, int, int, int]]]:
        """
        generator that yields (onset, pitch, velocity, offset iterator)
        """

        iterator = iter(self.notes)
        for bar_start in range(0, self.duration, bar_length):
            list_of_notes = []
            try:
                while True:
                    note = next(iterator)
                    if note.onset >= bar_start + bar_length:
                        break
                    list_of_notes.append(
                        (note.onset, note.pitch, note.velocity, note.offset)
                    )
            except StopIteration:
                pass
            yield list_of_notes

    def iter_over_bars(self, bar_length=32) -> Generator[list[Note]]:
        """
        generator that yields list of notes in each bar
        """

        iterator = iter(self.notes)
        for bar_start in range(0, self.duration, bar_length):
            list_of_notes = []
            try:
                while True:
                    note = next(iterator)
                    if note.onset >= bar_start + bar_length:
                        break
                    list_of_notes.append(note)
            except StopIteration:
                pass
            yield list_of_notes

    def get_offsets_with_pedal(self, pedal) -> list[int]:
        offsets = []
        next_onset = [INF] * 88
        i = len(pedal)
        for onset, pitch, vel, _ in reversed(
            list(self.iter_over_notes_unpack())
        ):  # TODO: handle offsets if there are ones
            pitch -= 21  # midi number to piano
            while i > 0 and pedal[i - 1] > onset:
                i -= 1
            if i == len(pedal):
                next_pedal_up = self.duration
            else:
                next_pedal_up = pedal[i]

            offset = min(next_onset[pitch], next_pedal_up)

            offsets.append(offset)
            next_onset[pitch] = onset
        offsets = list(reversed(offsets))
        return offsets

    """
    ==================
    Type conversion
    ==================
    """

    def to_dict(self):
        data = {"onset_events": [], "pedal_events": []}
        for note in self.notes:
            data["onset_events"].append([note.onset, note.pitch, note.velocity])
        data["pedal_events"] = self.pedal if self.pedal else []
        return data

    def save(self, path):
        data = {"onset_events": [], "pedal_events": []}
        for note in self.notes:
            data["onset_events"].append([note.onset, note.pitch, note.velocity])
        data["pedal_events"] = self.pedal if self.pedal else []

        json_dump({"onset_events": self.notes, "pedal_events": self.pedal}, path)

    def to_tensor(
        self,
        start_time: int = 0,
        end_time: int = INF,
        padding=False,
        normalized=False,
        chromagram=False,
    ) -> "torch.Tensor":
        """
        Convert the pianoroll to a tensor
        """
        if not HAS_TORCH:
            raise ImportError(
                "PianoRoll.to_tensor requires torch. Please install torch."
            )

        n_features = 88 if not chromagram else 12

        if padding:
            # zero pad to end_time
            assert end_time != INF
            length = end_time - start_time
        else:
            length = min(self.duration, end_time) - start_time

        size = [length, n_features]
        piano_roll = torch.zeros(size)

        for time, pitch, vel, _ in self.iter_over_notes_unpack():
            rel_time = time - start_time
            # only contain notes between start_time and end_time
            if rel_time < 0:
                continue
            if rel_time >= length:
                break
            pitch -= 21  # midi to piano
            if chromagram:
                pitch = (pitch + 9) % 12
            piano_roll[rel_time, pitch] = vel

        if normalized:
            piano_roll = piano_roll / 64 - 1
        return piano_roll

    def to_midi(
        self, path=None, apply_pedal=True, bpm=105
    ) -> miditoolkit.midi.parser.MidiFile:
        """
        Convert the pianoroll to a midi file
        """
        notes = deepcopy(self.notes)
        if apply_pedal:
            if self.pedal:
                pedal = self.pedal
            else:
                pedal = list(range(0, self.duration, 32))
            offsets = self.get_offsets_with_pedal(pedal)
            for i, note in enumerate(notes):
                note.offset = offsets[i]
        else:
            assert self._have_offset, "Offset not found"
        return self._save_to_midi([notes], path, bpm)

    def _save_to_midi(self, instrs, path, bpm=105):
        midi = miditoolkit.midi.parser.MidiFile()
        midi.instruments = [
            miditoolkit.Instrument(program=0, is_drum=False, name=f"Piano{i}")
            for i in range(len(instrs))
        ]
        midi.tempo_changes.append(miditoolkit.TempoChange(bpm, 0))
        for i, notes in enumerate(instrs):
            for onset, pitch, vel, offset in self.iter_over_notes_unpack(notes):
                assert offset is not None, "Offset not found"
                midi.instruments[i].notes.append(
                    miditoolkit.Note(
                        vel,
                        pitch,
                        int(onset * midi.ticks_per_beat / 8),
                        int(offset * midi.ticks_per_beat / 8),
                    )
                )

        if path:
            midi.dump(path)
        return midi

    def save_to_pretty_score(
        self,
        path,
        separate_point=60,
        position_weight=3,
        mode="combine",
        make_pretty_voice=True,
    ):
        notes = deepcopy(self.notes)
        # separate left and right hand
        left_hand: list[Note] = []
        right_hand: list[Note] = []

        def loss(
            note,
            prev_notes,
            which_hand,
            max_dist=16,
            separate_point=60,
            position_weight=3,
        ):
            res = 0

            for prev_note in reversed(prev_notes):
                dt = note.onset - prev_note.onset
                dp = note.pitch - prev_note.pitch
                if dt > max_dist:
                    break
                loss = max(0, abs(dp) - 5 - 8 * dt)
                res += loss

            if which_hand == "l":
                res += (note.pitch - separate_point) * position_weight
            elif which_hand == "r":
                res -= (note.pitch - separate_point) * position_weight
            else:
                raise ValueError("which_hand must be 'l' or 'r'")
            return res

        # recursively search for min loss
        def cummulative_loss(
            past_notes_l, past_notes_r, future_notes, max_depth=4, discount_factor=0.9
        ):
            future_notes = future_notes[:max_depth]
            if len(future_notes) == 0:
                return 0, "l"
            else:
                future_loss_l = cummulative_loss(
                    past_notes_l + [future_notes[0]], past_notes_r, future_notes[1:]
                )[0]
                future_loss_r = cummulative_loss(
                    past_notes_l, past_notes_r + [future_notes[0]], future_notes[1:]
                )[0]
                loss_l = future_loss_l * discount_factor + loss(
                    future_notes[0],
                    past_notes_l,
                    "l",
                    16,
                    separate_point,
                    position_weight,
                )
                loss_r = future_loss_r * discount_factor + loss(
                    future_notes[0],
                    past_notes_r,
                    "r",
                    16,
                    separate_point,
                    position_weight,
                )
                if loss_l < loss_r:
                    return loss_l, "l"
                else:
                    return loss_r, "r"

        while len(notes):
            _, hand = cummulative_loss(left_hand, right_hand, notes)
            if hand == "l":
                left_hand.append(notes.pop(0))
            else:
                right_hand.append(notes.pop(0))

        def pretty_voice(voice: list[Note]):
            current = []
            for note in voice:
                if len(current) == 0:
                    current.append(note)
                else:
                    if note.onset == current[-1].onset:
                        current.append(note)
                    else:
                        stop_time = note.onset
                        for c in current:
                            c.offset = stop_time
                        current = [note]

        if make_pretty_voice:
            pretty_voice(left_hand)
            pretty_voice(right_hand)
        res = [right_hand, left_hand]
        print("left hand notes:", len(left_hand))
        print("right hand notes:", len(right_hand))
        if mode == "combine":
            self._save_to_midi(res, path)
        elif mode == "separate":
            self._save_to_midi([left_hand], path + "_left.mid")
            self._save_to_midi([right_hand], path + "_right.mid")

    def to_img(self, path):
        """
        Convert the pianoroll to a image
        """
        img = np.zeros((88, self.duration))
        for time, pitch, vel, offset in self.iter_over_notes_unpack():
            img[pitch - 21, time] = vel
        # enlarge the image
        img = np.repeat(img, 8, axis=0)
        img = np.repeat(img, 8, axis=1)
        for t in range(self.duration):
            if t % 32 == 0:
                img[:, t * 8] = 120

        # inverse y
        img = np.flip(img, axis=0)

        plt.imsave(path, img, cmap="gray", vmin=0, vmax=127)

    """
    ==================
    Basic operations
    ==================
    """

    def slice(self, start_time: int = 0, end_time: int = INF) -> "PianoRoll":
        """
        Slice a pianoroll from start_time to end_time
        """
        length = end_time - start_time
        sliced_notes = []
        sliced_pedal = []
        for time, pitch, vel, offset in self.iter_over_notes_unpack():
            rel_time = time - start_time
            if rel_time < 0:
                continue
            if rel_time >= length:
                break

            if offset is not None:
                rel_offset = offset - start_time
                rel_offset = min(rel_offset, length)
            else:
                rel_offset = None
            # only contain notes between start_time and end_time
            sliced_notes.append([rel_time, pitch, vel, rel_offset])

        if self.pedal:
            for pedal in self.pedal:
                time = pedal
                rel_time = time - start_time
                # only contain pedal between start_time and end_time
                if rel_time < 0:
                    continue
                if rel_time >= length:
                    break
                sliced_pedal.append(rel_time)
            new_pr = PianoRoll(
                {"onset_events": sliced_notes, "pedal_events": sliced_pedal}
            )
        else:
            new_pr = PianoRoll({"onset_events": sliced_notes})

        new_pr.set_metadata(
            self.metadata.name,
            self.metadata.start_time + start_time,
            self.metadata.start_time + end_time,
        )
        return new_pr

    def random_slice(self, length: int = 128) -> "PianoRoll":
        """
        Randomly slice a pianoroll with length
        """
        start_time = random.randint(0, max(0, (self.duration - length) // 32)) * 32
        return self.slice(start_time, start_time + length)

    def get_random_tensor_clip(self, duration, normalized=False):
        """
        Get a random clip of the pianoroll
        """
        start_time = (
            random.randint(0, (self.duration - duration) // 32) * 32
        )  # snap to bar
        return self.to_tensor(start_time, start_time + duration, normalized=normalized)

    def get_chord_sequence(self, granularity=32):
        """
        Get the chord sequence of the pianoroll
        """
        chords = []
        for bar in self.iter_over_bars_unpack(granularity):
            chroma = [0] * 12
            for time, pitch, vel, offset in bar:
                chroma[pitch % 12] += 1
            chord = chroma_to_chord(chroma)
            chords.append(chord)
        return chords

    def get_polyphony(self, granularity=32):
        """
        Get the polyphony of the pianoroll
        """
        polyphony = []
        for bar in self.iter_over_bars_unpack(granularity):
            to_be_reduced = []
            last_note_frame = 0
            poly = 0
            for frame, pitch, vel, offset in bar:
                if frame > last_note_frame:
                    to_be_reduced.append(poly)
                    last_note_frame = frame
                    poly = 0
                poly += 1
            max_3 = sorted(to_be_reduced, reverse=True)[:3]
            if len(max_3) == 0:
                polyphony.append(0)
            else:
                polyphony.append(sum(max_3) / len(max_3))
        return polyphony

    def get_density(self, granularity=32):
        """
        Get the density of the pianoroll
        """
        density = []
        for bar in self.iter_over_bars_unpack(granularity):
            frames = set()
            for frame, pitch, vel, offset in bar:
                frames.add(frame)
            density.append(len(frames))
        return density

    def get_velocity(self, granularity=32, num_vel=128):
        """
        Get the velocity of the pianoroll. Average velocity of each bar
        """
        velocity = []
        for bar in self.iter_over_bars_unpack(granularity):
            vel_sum = 0
            count = 0
            for frame, pitch, vel, offset in bar:
                vel_sum += vel
                count += 1
            if count == 0:
                count = 1
            velocity.append(int(vel_sum / 128 * 32 / count))
        return velocity
