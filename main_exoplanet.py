# main_exoplanet.py

import os
import random
from data import generate_exoplanet_data
from functions import assign_time_beats, play_midi_threaded
from midiutil import MIDIFile
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import plotly.express as px


def main():
    # ============================
    # Settings and Paths
    # ============================
    seed = 6
    num_planets = 50
    desired_duration_beats = 100  # Reduced beat range for higher simultaneity
    max_per_beat = 8  # Increased maximum number of exoplanets per beat
    tempo = 180  # Tempo in BPM
    duration = 0.4  # Note duration in beats

    # Define directories
    midi_dir = os.path.join('midi', 'exoplanet')
    plots_dir = os.path.join('plots', 'exoplanet')

    # Create directories if they don't exist
    os.makedirs(midi_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # ============================
    # Data Generation
    # ============================
    df = generate_exoplanet_data(num_planets=num_planets, seed=seed)

    # ============================
    # Mapping Clusters to Musical Scales
    # ============================
    scales = {
        0: ['C4', 'E4', 'G4', 'B4', 'D#5'],  # Added D#5 for dissonance
        1: ['D4', 'F#4', 'A4', 'C#5', 'E5'],  # Added E5 for dissonance
        2: ['E4', 'G#4', 'B4', 'D#5', 'F#5']  # Added F#5 for dissonance
    }

    # Convert scales to MIDI numbers
    def str2midi(note_string):
        note_str = note_string.upper()
        note_map = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6,
                    'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
        octave = int(note_str[-1])
        note = note_str[:-1]
        if note not in note_map:
            raise ValueError(f"Invalid note: {note}")
        midi_number = 12 * (octave + 1) + note_map[note]
        return midi_number

    scales_midi = {cluster: [str2midi(note) for note in notes] for cluster, notes in scales.items()}

    # Assign a random pitch from the cluster's scale to each exoplanet
    df['pitch'] = df['cluster'].apply(lambda x: random.choice(scales_midi[x]))

    # ============================
    # Normalizing PCA Components
    # ============================
    scaler_pca = MinMaxScaler()
    pca_scaled = scaler_pca.fit_transform(df[['PCA1', 'PCA2']])
    df['PCA1_scaled'] = pca_scaled[:, 0]
    df['PCA2_scaled'] = pca_scaled[:, 1]

    # ============================
    # Mapping PCA Components to Musical Parameters
    # ============================
    df['velocity'] = (110 + df['PCA1_scaled'] * (127 - 110)).astype(int)  # Velocity range: 110-127

    # ============================
    # Controlled Timeframe Assignment
    # ============================
    df_realistic = assign_time_beats(df, desired_duration_beats, max_per_beat=max_per_beat)

    # ============================
    # Creating the MIDI File
    # ============================
    track = 0
    channel = 0

    midi_file = MIDIFile(1)  # One track
    midi_file.addTempo(track, 0, tempo)

    for _, row in df_realistic.iterrows():
        pitch = row['pitch']
        velocity = row['velocity']
        time_beat = row['time_beats']
        midi_file.addNote(track, channel, pitch, time_beat, duration, velocity)

    midi_filename = os.path.join(midi_dir, 'exoplanet_sonification_tenser.mid')
    with open(midi_filename, 'wb') as f:
        midi_file.writeFile(f)

    print(f"MIDI file '{midi_filename}' created successfully.")

    # ============================
    # Plotly Animated Scatter Plot
    # ============================
    animation_df = df_realistic.copy()
    animation_df['frame'] = animation_df['time_beats']

    fig = px.scatter(
        animation_df, x='PCA1_scaled', y='PCA2_scaled',
        size='velocity', color='pitch',
        animation_frame='frame',
        range_x=[animation_df['PCA1_scaled'].min() - 0.1, animation_df['PCA1_scaled'].max() + 0.1],
        range_y=[animation_df['PCA2_scaled'].min() - 0.1, animation_df['PCA2_scaled'].max() + 0.1],
        title='Exoplanet Data Sonification Visualization',
        labels={'PCA1_scaled': 'PCA1', 'PCA2_scaled': 'PCA2'},
        color_continuous_scale='Viridis',
        size_max=60,  # Increased maximum size for better visibility
        opacity=0.7  # Added opacity for overlapping visibility
    )

    # Update layout to increase plot height and width
    fig.update_layout(
        height=800,  # Increased height from default (e.g., 600) to 800 pixels
        width=1000  # Optional: You can also adjust width if desired
    )

    # Save plot as HTML and PNG
    plot_html = os.path.join(plots_dir, 'exoplanet_sonification_visualization.html')
    plot_png = os.path.join(plots_dir, 'exoplanet_sonification_visualization.png')
    fig.write_html(plot_html)
    fig.write_image(plot_png)

    print(f"Plot saved as '{plot_html}' and '{plot_png}'.")

    # ============================
    # Synchronizing Animation and MIDI Playback
    # ============================
    from functions import play_midi_threaded  # Ensure it's imported

    def start_sonification(midi_file, plotly_fig, tempo_bpm):
        """
        Starts MIDI playback and Plotly animation in synchronization.

        Parameters:
        - midi_file (str): Path to the MIDI file.
        - plotly_fig (plotly.graph_objs._figure.Figure): Plotly figure object.
        - tempo_bpm (int): Tempo in beats per minute.
        """
        # Start MIDI playback
        play_midi_threaded(midi_file)

        # Start the Plotly animation
        plotly_fig.show()

    # Execute synchronization
    start_sonification(midi_filename, fig, tempo)


if __name__ == "__main__":
    main()
