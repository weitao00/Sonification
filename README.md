# Multi-Domain Data Sonification and Visualization

This project explores the **sonification** and **visualization** of two distinct datasets: **Exoplanets** and **Wines**. By leveraging **machine learning** techniques to analyze and reduce data dimensionality, and by mapping data attributes to musical parameters, the project creates an immersive multi-sensory experience. Users can **listen** to data-driven compositions and **visualize** data interactions simultaneously, discovering hidden patterns and relationships within the datasets.
The use of **animated visualizations** captures the temporal aspects of the data, allowing users to observe changes and interactions over time. This dual representation aids in data interpretation and provides an engaging way to interact with information.
## Video Demonstrations

### Exoplanet Sonification with Animation
[[Exoplanet Sonification]](https://youtu.be/LSw6supn1M0)

### Wine Sonification with Animation
[[Wine Dataset Visualization]]( https://youtu.be/jxLosmN3pR4)
## Table of Contents

- [Project Overview](#project-overview)
  - [Exoplanet Dataset](#exoplanet-dataset)
  - [Wine Dataset](#wine-dataset)
- [Machine Learning Methods](#machine-learning-methods)
  - [Feature Scaling](#feature-scaling)
  - [Clustering](#clustering)
  - [Dimensionality Reduction](#dimensionality-reduction)
- [Sonification Process](#sonification-process)
  - [Mapping Clusters to Musical Scales](#mapping-clusters-to-musical-scales)
  - [Assigning Pitches](#assigning-pitches)
  - [Mapping PCA Components to Musical Parameters](#mapping-pca-components-to-musical-parameters)
  - [Controlled Timeframe Assignment](#controlled-timeframe-assignment)
  - [MIDI File Creation](#midi-file-creation)
- [Visualization](#visualization)
- [Folder Structure](#folder-structure)
- [Setup Instructions](#setup-instructions)
- [Usage Guidelines](#usage-guidelines)
- [Customization](#customization)

## Project Overview

This project demonstrates how to transform complex datasets into both **auditory** and **visual** representations. By applying machine learning techniques, the data is preprocessed and analyzed to uncover underlying structures, which are then mapped to musical and visual elements. This dual representation aids in data interpretation and provides an engaging way to interact with information.

### Exoplanet Dataset

The **Exoplanet Dataset** simulates characteristics of 50 fictional exoplanets, including:

- **Mass (Earth Masses):** Represents the mass of the exoplanet relative to Earth.
- **Radius (Earth Radii):** Indicates the size of the exoplanet compared to Earth.
- **Orbital Period (Days):** The time an exoplanet takes to complete one orbit around its star.
- **Equilibrium Temperature (K):** The temperature of the exoplanet assuming it is a black body.
- **Semi-Major Axis (AU):** The longest radius of the exoplanet's elliptical orbit, measured in Astronomical Units.
- **Eccentricity:** Describes the shape of the exoplanet's orbit.

### Wine Dataset

The **Wine Dataset** emulates characteristics of 50 fictional wines, encompassing:

- **Alcohol (% by Volume):** The alcohol content in the wine.
- **Malic Acid (g/L):** Amount of malic acid present.
- **Ash (g/L):** Ash content in the wine.
- **Alcalinity of Ash (mg/L):** Measures the alkalinity of the ash in the wine.
- **Magnesium (mg/L):** Magnesium content.
- **Total Phenols (g/L):** Total phenolic compounds.
- **Flavanoids (g/L):** A subclass of flavonoids.
- **Nonflavanoid Phenols (g/L):** Phenolic compounds that are not flavonoids.
- **Proanthocyanins (g/L):** A type of polyphenol.
- **Color Intensity:** The intensity of the wine's color.
- **Hue:** The hue of the wine.
- **OD280/OD315 (Optical Density Ratio):** A measure related to phenolic content.
- **Proline (mg/L):** Amino acid content.

## Machine Learning Methods

The project utilizes several machine learning techniques to preprocess and analyze the datasets, enabling effective sonification and visualization.

### Feature Scaling

Before applying any machine learning algorithms, it's crucial to **standardize** the data to ensure that each feature contributes equally to the analysis. This project utilizes **StandardScaler** from scikit-learn to scale the features to have a mean of 0 and a standard deviation of 1.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])
df_scaled = pd.DataFrame(scaled_features, columns=features)
```

### Clustering

To identify inherent groupings within the datasets, K-Means Clustering is employed. This algorithm partitions the data into a specified number of clusters (3 in this case), grouping similar data points together based on feature similarity.

```python
from sklearn.cluster import KMeans

k = 3
kmeans = KMeans(n_clusters=k, random_state=6)
kmeans.fit(df_scaled)
df['cluster'] = kmeans.labels_
```

Purpose:
Clustering helps in categorizing the data into distinct groups, which can then be mapped to different musical scales for sonification. Each cluster represents a unique segment of the dataset with similar characteristics.

### Dimensionality Reduction

High-dimensional data can be challenging to visualize and interpret. Principal Component Analysis (PCA) reduces the dataset's dimensionality to two principal components, capturing the most significant variance and facilitating visualization.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=6)
pca_features = pca.fit_transform(df_scaled)
df_pca = pd.DataFrame(pca_features, columns=['PCA1', 'PCA2'])
df = pd.concat([df, df_pca], axis=1)
```

Purpose:
PCA simplifies the complexity of high-dimensional data while retaining the most critical information, making it easier to visualize patterns and relationships in two-dimensional plots.

## Sonification Process

Sonification involves converting data attributes into musical parameters, enabling auditory exploration of data.

### Mapping Clusters to Musical Scales

Each cluster is assigned a dissonant musical scale to enhance tension and distinguish between different data groups.

```python
scales = {
    0: ['C4', 'E4', 'G4', 'B4', 'D#5'],    # Cluster 0
    1: ['D4', 'F#4', 'A4', 'C#5', 'E5'],   # Cluster 1
    2: ['E4', 'G#4', 'B4', 'D#5', 'F#5']   # Cluster 2
}
```

Rationale:
Using dissonant scales adds musical tension, making the sonification more engaging and highlighting the distinctiveness of each cluster.

### Assigning Pitches

Random pitches from the assigned scales are selected for each data point.

```python
def str2midi(note_string):
    note_str = note_string.upper()
    note_map = {'C': 0, 'C#':1, 'D':2, 'D#':3, 'E':4, 'F':5, 'F#':6,
                'G':7, 'G#':8, 'A':9, 'A#':10, 'B':11}
    octave = int(note_string[-1])
    note = note_str[:-1]
    if note not in note_map:
        raise ValueError(f"Invalid note: {note}")
    midi_number = 12 * (octave + 1) + note_map[note]
    return midi_number

scales_midi = {cluster: [str2midi(note) for note in notes] for cluster, notes in scales.items()}

df['pitch'] = df['cluster'].apply(lambda x: random.choice(scales_midi[x]))
```

Purpose:
Assigning specific pitches based on clusters ensures that each data group is represented by a unique set of musical notes, enhancing the auditory differentiation between clusters.

### Mapping PCA Components to Musical Parameters

PCA1_scaled: Mapped to MIDI velocity (intensity).
PCA2_scaled: Mapped to timing (when the note is played).

```python
from sklearn.preprocessing import MinMaxScaler

scaler_pca = MinMaxScaler()
pca_scaled = scaler_pca.fit_transform(df[['PCA1', 'PCA2']])
df['PCA1_scaled'] = pca_scaled[:,0]
df['PCA2_scaled'] = pca_scaled[:,1]

df['velocity'] = (110 + df['PCA1_scaled'] * (127 - 110)).astype(int)  # Velocity range: 110-127
desired_duration_beats = 100
df['time_beats'] = (df['PCA2_scaled'] * desired_duration_beats).astype(int)
```

Rationale:
- Velocity: Controls the loudness of the note, allowing the most significant data points (with higher PCA1 values) to be more prominent.
- Timing: Determines when each note is played, mapping the data's temporal aspect to the musical timeline.

### Controlled Timeframe Assignment

Ensures a maximum number of data points (exoplanets/wines) are assigned to the same beat to maintain clarity in both visualization and sonification.

### MIDI File Creation

MIDI notes are added based on the mapped parameters and saved to designated directories.

```python
from midiutil import MIDIFile

def create_midi(df_realistic, midi_filename, tempo=180, duration=0.4):
    track = 0
    channel = 0
    
    midi_file = MIDIFile(1)  # One track
    midi_file.addTempo(track, 0, tempo)
    
    for _, row in df_realistic.iterrows():
        pitch = row['pitch']
        velocity = row['velocity']
        time_beat = row['time_beats']
        midi_file.addNote(track, channel, pitch, time_beat, duration, velocity)
    
    with open(midi_filename, 'wb') as f:
        midi_file.writeFile(f)
    
    print(f"MIDI file '{midi_filename}' created successfully.")
```

Functionality:
This function creates a MIDI file by iterating over each data point, adding notes with specified pitch, velocity, timing, and duration. The resulting MIDI file can be played to audibly experience the data's structure.

## Visualization

The project employs Plotly to create animated scatter plots that visualize the datasets in two dimensions based on PCA results.

```python
import plotly.express as px

def create_plot(animation_df, plot_filename_html, plot_filename_png):
    fig = px.scatter(
        animation_df, x='PCA1_scaled', y='PCA2_scaled',
        size='velocity', color='pitch',
        animation_frame='frame',
        range_x=[animation_df['PCA1_scaled'].min()-0.1, animation_df['PCA1_scaled'].max()+0.1],
        range_y=[animation_df['PCA2_scaled'].min()-0.1, animation_df['PCA2_scaled'].max()+0.1],
        title='Data Sonification Visualization',
        labels={'PCA1_scaled': 'PCA1', 'PCA2_scaled': 'PCA2'},
        color_continuous_scale='Viridis',
        size_max=60,      # Increased maximum size for better visibility
        opacity=0.7       # Added opacity for overlapping visibility
    )
    
    # Update layout to increase plot height and width
    fig.update_layout(
        height=800,       # Increased height from default (e.g., 600) to 800 pixels
        width=1000        # Optional: You can also adjust width if desired
    )
    
    # Save plot as HTML and PNG
    fig.write_html(plot_filename_html)
    fig.write_image(plot_filename_png)
    
    print(f"Plot saved as '{plot_filename_html}' and '{plot_filename_png}'.")
```

Features:
- Animated Frames: Each frame corresponds to a specific beat, displaying the data points assigned to that timeframe.
- Dynamic Sizing and Coloring: Points vary in size based on velocity (intensity) and color based on pitch, providing visual cues that correlate with the sonified data.
- Interactivity: Plotly's interactive plots allow users to hover over points for more information and control the animation playback.

## Folder Structure

Organizing the project into a clear and modular structure enhances maintainability and scalability.

```
project_root/
├── data.py
├── functions.py
├── main_exoplanet.py
├── main_wine.py
├── midi/
│   ├── exoplanet/
│   └── wine/
├── plots/
│   ├── exoplanet/
│   └── wine/
├── README.md
└── requirements.txt
```

Explanation of Each Component:
- `data.py`: Contains functions to generate synthetic datasets for both exoplanets and wines.
- `functions.py`: Houses utility functions such as MIDI conversion, playback, and timeframe assignment.
- `main_exoplanet.py`: Orchestrates the exoplanet data processing, sonification, and visualization.
- `main_wine.py`: Manages the wine data processing, sonification, and visualization.
- `midi/`: Directory to store generated MIDI files.
- `plots/`: Directory to store generated plots.
- `README.md`: Provides an overview of the project, setup instructions, and usage guidelines.
- `requirements.txt`: Lists all Python dependencies required to run the project.


## Usage Guidelines

### MIDI Files
Generated MIDI files are saved in the respective directories:
- Exoplanets: `midi/exoplanet/exoplanet_sonification_tenser.mid`
- Wines: `midi/wine/wine_sonification_tenser.mid`

### Plots
Generated plots are saved as both HTML and PNG files:
- Exoplanets:
  - `plots/exoplanet/exoplanet_sonification_visualization.html`
  - `plots/exoplanet/exoplanet_sonification_visualization.png`
- Wines:
  - `plots/wine/wine_sonification_visualization.html`
  - `plots/wine/wine_sonification_visualization.png`

### Listening and Viewing
Running `main_exoplanet.py` or `main_wine.py` will automatically play the generated MIDI file and display the animated Plotly scatter plot.
- MIDI Playback: Ensure your system's audio is enabled. The MIDI files will play automatically upon running the scripts.
- Interactive Plots: Open the HTML files in a web browser for an interactive experience or view the PNG files as static images.

## Customization

### Random Seed
The random seed is set to 6 for reproducibility. You can change this value in `main_exoplanet.py` and `main_wine.py` to generate different datasets.
```python
seed = 6  # Change to any integer for different randomization
```

### Maximum Points per Beat
Adjust the `max_per_beat` parameter in both `main_exoplanet.py` and `main_wine.py` to control the number of points occurring simultaneously.
```python
max_per_beat = 8  # Increase or decrease based on desired simultaneity
```

### Tempo and Duration
Modify the `tempo`
