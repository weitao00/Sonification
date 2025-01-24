# functions.py

import random
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from midiutil import MIDIFile
import pygame
import time
import threading


def str2midi(note_string):
    """
    Converts a musical note string to its corresponding MIDI number.

    Parameters:
    - note_string (str): Musical note (e.g., 'C4', 'F#3').

    Returns:
    - int: MIDI number corresponding to the note.
    """
    note_str = note_string.upper()
    note_map = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6,
                'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
    octave = int(note_str[-1])
    note = note_str[:-1]
    if note not in note_map:
        raise ValueError(f"Invalid note: {note}")
    midi_number = 12 * (octave + 1) + note_map[note]
    return midi_number


def assign_time_beats(df, desired_duration_beats, max_per_beat=5):
    """
    Assigns time_beats to data points based on a scaled PCA component with controlled duplicates.

    Parameters:
    - df (DataFrame): The dataset.
    - desired_duration_beats (int): Total number of beats for the timeframe.
    - max_per_beat (int): Maximum number of data points allowed per beat.

    Returns:
    - DataFrame: Updated DataFrame with assigned time_beats.
    """
    df = df.copy()
    df['time_beats'] = (df['PCA2_scaled'] * desired_duration_beats).astype(int)

    # Sort by PCA2_scaled to distribute duplicates
    df = df.sort_values('PCA2_scaled')

    # Initialize a dictionary to track beat assignments
    beat_assignments = {}

    for idx, row in df.iterrows():
        beat = row['time_beats']
        # Allow a maximum of 'max_per_beat' data points per beat
        if beat in beat_assignments:
            if len(beat_assignments[beat]) < max_per_beat:
                beat_assignments[beat].append(idx)
            else:
                # Assign to the next available beat
                beat += 1
                while beat in beat_assignments and len(beat_assignments[beat]) >= max_per_beat:
                    beat += 1
                if beat > desired_duration_beats:
                    beat = desired_duration_beats  # Assign to the last beat if overflow
                df.at[idx, 'time_beats'] = beat
                beat_assignments.setdefault(beat, []).append(idx)
        else:
            beat_assignments[beat] = [idx]

    return df


def play_midi(midi_file):
    """
    Plays a MIDI file using pygame.

    Parameters:
    - midi_file (str): Path to the MIDI file.
    """
    pygame.init()
    pygame.mixer.init()
    try:
        pygame.mixer.music.load(midi_file)
    except pygame.error as e:
        print(f"Cannot load {midi_file}: {e}")
        return
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)
    pygame.mixer.quit()


def play_midi_threaded(midi_file):
    """
    Plays a MIDI file in a separate thread to allow concurrent execution.

    Parameters:
    - midi_file (str): Path to the MIDI file.
    """
    midi_thread = threading.Thread(target=play_midi, args=(midi_file,))
    midi_thread.start()
