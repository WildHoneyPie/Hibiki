import json
import numpy as np
from pydub import AudioSegment
import os
from typing import Dict, List, Tuple, Set
import random

def load_json_file(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        return json.load(f)

def find_closest_music(emotions: Dict[str, float], music_library: Dict[str, Dict[str, float]], used_music: Set[str]) -> str:
    """Find the music file with the closest emotional values, excluding already used music."""
    min_distance = float('inf')
    closest_music = None
    
    target_valence = emotions['valence']
    target_arousal = emotions['arousal']
    
    for music_file, music_emotions in music_library.items():
        # Skip if this music has been used before
        if music_file in used_music:
            continue
            
        distance = np.sqrt(
            (target_valence - music_emotions['valence'])**2 +
            (target_arousal - music_emotions['arousal'])**2
        )
        if distance < min_distance:
            min_distance = distance
            closest_music = music_file
            
    return closest_music

def create_music_sequence(
    text_emotions: dict,
    music_library: dict,
    total_duration: int = 60000,  # 60 seconds in milliseconds
    crossfade_duration: int = 5000,  # 5 seconds crossfade
    fade_duration: int = 3000  # 3 seconds fade in/out
) -> AudioSegment:
    """Create a music sequence based on text emotions."""
    # Load all music files
    music_files = {}
    for music_file in music_library.keys():
        file_path = os.path.join('music_library', music_file)
        if os.path.exists(file_path):
            music_files[music_file] = AudioSegment.from_wav(file_path)
    
    # Keep track of used music files
    used_music = set()
    
    # Store music segments for each paragraph
    music_segments = []
    
    # Process each paragraph
    for i, paragraph in enumerate(text_emotions['paragraphs']):
        # Calculate duration for this paragraph
        start_proportion = paragraph['proportion']['start']
        end_proportion = paragraph['proportion']['end']
        duration = int((end_proportion - start_proportion) * total_duration)
        
        # 為了補償crossfade造成的長度縮短，除了第一段以外，每段都要加上crossfade的長度
        if i > 0:
            duration += crossfade_duration
        
        print(f"\nProcessing paragraph {i+1}:")
        print(f"Duration (including crossfade compensation): {duration/1000:.2f}s")
        
        # Find closest matching music
        closest_music = find_closest_music(paragraph['emotions'], music_library, used_music)
        if closest_music not in music_files:
            continue
            
        # Add to used music set
        used_music.add(closest_music)
        
        music_segment = music_files[closest_music]
        print(f"Selected music: {closest_music}")
        print(f"Original music length: {len(music_segment)/1000:.2f}s")
        
        # Loop the music if needed to match the required duration
        while len(music_segment) < duration:
            music_segment = music_segment + music_segment
            
        # Trim to exact duration
        music_segment = music_segment[:duration]
        print(f"Final segment length: {len(music_segment)/1000:.2f}s")
        
        music_segments.append(music_segment)
    
    # Combine all segments using crossfade
    if not music_segments:
        return AudioSegment.silent(duration=total_duration)
    
    final_audio = music_segments[0]
    print(f"\nStarting with first segment: {len(final_audio)/1000:.2f}s")
    
    for i, segment in enumerate(music_segments[1:], 1):
        print(f"Adding segment {i+1} with {crossfade_duration/1000:.2f}s crossfade")
        final_audio = final_audio.append(segment, crossfade=crossfade_duration)
        print(f"Combined length after segment {i+1}: {len(final_audio)/1000:.2f}s")
    
    # Adjust final audio to target duration
    print(f"\nFinal audio length before adjustment: {len(final_audio)/1000:.2f}s")
    print(f"Target length: {total_duration/1000:.2f}s")
    
    if len(final_audio) > total_duration:
        final_audio = final_audio[:total_duration]
        print(f"Trimmed audio to target length: {len(final_audio)/1000:.2f}s")
    elif len(final_audio) < total_duration:
        # Pad with silence if needed
        silence_needed = total_duration - len(final_audio)
        final_audio = final_audio + AudioSegment.silent(duration=silence_needed)
        print(f"Padded audio with {silence_needed/1000:.2f}s silence")
    
    # Apply fade in/out
    final_audio = final_audio.fade_in(fade_duration).fade_out(fade_duration)
    
    print(f"Final audio length after fade: {len(final_audio)/1000:.2f}s")
    
    return final_audio

def main(ambient_volume: float = 1.0, music_volume: float = 1.0):
    """
    Generate and mix music with ambient sound.
    
    Args:
        ambient_volume (float): Volume multiplier for ambient sound (0.0 to 2.0)
            where 0.0 means silent, 1.0 means original volume, 2.0 means double volume.
            Default is 1.0.
        music_volume (float): Volume multiplier for generated music (0.0 to 2.0)
            where 0.0 means silent, 1.0 means original volume, 2.0 means double volume.
            Default is 1.0.
    """
    # Load emotion data
    text_emotions = load_json_file('text_emotion.json')
    music_library = load_json_file('music_emotion_results.json')
    
    # Generate music sequence
    final_audio = create_music_sequence(text_emotions, music_library)
    
    # Export the generated music
    final_audio.export('generated_music.wav', format='wav')
    
    # Get a random ambient sound from the ambience folder
    ambience_files = [f for f in os.listdir('ambience') if f.endswith('.wav')]
    if ambience_files:
        random_ambience = random.choice(ambience_files)
        ambience_path = os.path.join('ambience', random_ambience)
        
        # Load ambient sound
        ambient_sound = AudioSegment.from_wav(ambience_path)
        
        # Ensure ambient sound is exactly 60 seconds
        if len(ambient_sound) > 60000:  # If longer than 60 seconds
            ambient_sound = ambient_sound[:60000]  # Trim to 60 seconds
        elif len(ambient_sound) < 60000:  # If shorter than 60 seconds
            # Loop until we have enough length
            while len(ambient_sound) < 60000:
                ambient_sound = ambient_sound + ambient_sound
            ambient_sound = ambient_sound[:60000]  # Trim to exactly 60 seconds
        
        # Apply fade in/out to ambient sound (3 seconds = 3000ms)
        ambient_sound = ambient_sound.fade_in(3000).fade_out(3000)
        
        # Convert volume multipliers to dB changes
        # Volume multiplier of 1.0 = 0dB change
        # Volume multiplier of 2.0 = +6dB
        # Volume multiplier of 0.5 = -6dB
        ambient_db = 20 * np.log10(ambient_volume)
        music_db = 20 * np.log10(music_volume)
        
        # Apply volume adjustments
        final_audio = final_audio + music_db
        ambient_sound = ambient_sound + ambient_db
        
        # Mix the sounds
        mixed_audio = final_audio.overlay(ambient_sound)
        
        # Export the final mixed audio
        mixed_audio.export('final_output_audio.wav', format='wav')
        print(f"Mixed with ambient sound: {random_ambience}")
        print(f"Ambient volume multiplier: {ambient_volume:.2f} ({ambient_db:.1f}dB)")
        print(f"Music volume multiplier: {music_volume:.2f} ({music_db:.1f}dB)")
        print(f"Final output length: {len(mixed_audio)/1000:.2f} seconds")
    else:
        print("No ambient sounds found in the ambience folder")
        final_audio.export('final_output_audio.wav', format='wav')

if __name__ == '__main__':
    # Example usage with different volume settings:
    # main(1.0, 1.0)  # Original volumes (default)
    # main(2.0, 1.0)  # Double ambient volume
    # main(1.0, 0.5)  # Half music volume
    # main(1.5, 0.8)  # Ambient 50% louder, music 20% quieter
    main(2.0, 0.5)  # Example mix