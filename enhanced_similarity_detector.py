import librosa
import acoustid
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import time
import json
from typing import Dict, Tuple
from difflib import SequenceMatcher
import os

def analyze_audio_characteristics(audio_path: str) -> Dict:
    """
    Analyze detailed audio characteristics to understand differences
    Runtime: O(n) where n is the number of audio samples
    """
    print(f"Analyzing audio characteristics for: {audio_path}")
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=None)
    
    # Basic characteristics
    duration = len(audio) / sr
    rms_energy = np.sqrt(np.mean(audio**2))
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
    
    # Spectral characteristics
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    
    # MFCC features
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    
    # Chroma features (similar to what AcoustID uses)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    
    # Tempo and rhythm
    tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
    
    # Harmonic and percussive separation
    harmonic, percussive = librosa.effects.hpss(audio)
    
    return {
        'duration': duration,
        'sample_rate': sr,
        'rms_energy': float(rms_energy),
        'zero_crossing_rate_mean': float(np.mean(zero_crossing_rate)),
        'spectral_centroid_mean': float(np.mean(spectral_centroid)),
        'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
        'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
        'mfcc_mean': np.mean(mfcc, axis=1).tolist(),
        'chroma_mean': np.mean(chroma, axis=1).tolist(),
        'tempo': float(tempo),
        'harmonic_ratio': float(np.mean(harmonic**2) / np.mean(audio**2)),
        'percussive_ratio': float(np.mean(percussive**2) / np.mean(audio**2))
    }

def enhanced_similarity_detection(audio_path1: str, audio_path2: str) -> Dict:
    """
    Enhanced similarity detection with multiple analysis methods
    Runtime: O(n) where n is the number of audio samples
    """
    print("=" * 60)
    print("ENHANCED AUDIO SIMILARITY DETECTION")
    print("=" * 60)
    
    # Analyze both files
    char1 = analyze_audio_characteristics(audio_path1)
    char2 = analyze_audio_characteristics(audio_path2)
    
    # Calculate similarities for each characteristic
    similarities = {}
    
    # Duration similarity
    duration_diff = abs(char1['duration'] - char2['duration'])
    similarities['duration'] = max(0, 1 - duration_diff / max(char1['duration'], char2['duration']))
    
    # Energy similarity
    energy_diff = abs(char1['rms_energy'] - char2['rms_energy'])
    similarities['energy'] = max(0, 1 - energy_diff / max(char1['rms_energy'], char2['rms_energy']))
    
    # Spectral similarities
    spectral_centroid_diff = abs(char1['spectral_centroid_mean'] - char2['spectral_centroid_mean'])
    similarities['spectral_centroid'] = max(0, 1 - spectral_centroid_diff / max(char1['spectral_centroid_mean'], char2['spectral_centroid_mean']))
    
    spectral_rolloff_diff = abs(char1['spectral_rolloff_mean'] - char2['spectral_rolloff_mean'])
    similarities['spectral_rolloff'] = max(0, 1 - spectral_rolloff_diff / max(char1['spectral_rolloff_mean'], char2['spectral_rolloff_mean']))
    
    # MFCC similarity
    mfcc1 = np.array(char1['mfcc_mean'])
    mfcc2 = np.array(char2['mfcc_mean'])
    similarities['mfcc'] = float(np.corrcoef(mfcc1, mfcc2)[0, 1])
    
    # Chroma similarity (what AcoustID primarily uses)
    chroma1 = np.array(char1['chroma_mean'])
    chroma2 = np.array(char2['chroma_mean'])
    similarities['chroma'] = float(np.corrcoef(chroma1, chroma2)[0, 1])
    
    # Tempo similarity
    tempo_diff = abs(char1['tempo'] - char2['tempo'])
    similarities['tempo'] = max(0, 1 - tempo_diff / max(char1['tempo'], char2['tempo']))
    
    # Harmonic/Percussive ratio similarity
    harmonic_diff = abs(char1['harmonic_ratio'] - char2['harmonic_ratio'])
    similarities['harmonic_ratio'] = max(0, 1 - harmonic_diff)
    
    percussive_diff = abs(char1['percussive_ratio'] - char2['percussive_ratio'])
    similarities['percussive_ratio'] = max(0, 1 - percussive_diff)
    
    # Calculate overall similarity
    overall_similarity = np.mean(list(similarities.values()))
    
    return {
        'characteristics1': char1,
        'characteristics2': char2,
        'similarities': similarities,
        'overall_similarity': overall_similarity
    }

def investigate_acoustid_limitation(audio_path1: str, audio_path2: str):
    """
    Investigate why AcoustID might give false positives
    Runtime: O(n) where n is the number of audio samples
    """
    print(f"\n{'='*60}")
    print("INVESTIGATING ACOUSTID LIMITATION")
    print(f"{'='*60}")
    
    # Generate AcoustID fingerprints
    duration1, fingerprint1 = acoustid.fingerprint_file(audio_path1)
    duration2, fingerprint2 = acoustid.fingerprint_file(audio_path2)
    
    fingerprint1_str = fingerprint1.decode('utf-8') if isinstance(fingerprint1, bytes) else str(fingerprint1)
    fingerprint2_str = fingerprint2.decode('utf-8') if isinstance(fingerprint2, bytes) else str(fingerprint2)
    
    print(f"AcoustID Fingerprint 1: {fingerprint1_str}")
    print(f"AcoustID Fingerprint 2: {fingerprint2_str}")
    print(f"Fingerprint Length 1: {len(fingerprint1_str)}")
    print(f"Fingerprint Length 2: {len(fingerprint2_str)}")
    
    # Check if fingerprints are actually identical
    are_identical = fingerprint1_str == fingerprint2_str
    print(f"Fingerprints are identical: {are_identical}")
    
    if are_identical:
        print("\n🔍 ANALYSIS: AcoustID is giving false positive!")
        print("This suggests the Chromaprint algorithm is not sensitive enough")
        print("to distinguish between these two different audio files.")
        print("\nPossible reasons:")
        print("  • Both files have similar chroma characteristics")
        print("  • Files are too short for meaningful differentiation")
        print("  • Chromaprint focuses on harmonic content, ignoring other features")
        print("  • Files may have similar musical structure but different timbre")
    
    return {
        'fingerprint1': fingerprint1_str,
        'fingerprint2': fingerprint2_str,
        'are_identical': are_identical,
        'duration1': duration1,
        'duration2': duration2
    }

def create_detailed_comparison_visualization(audio_path1: str, audio_path2: str, enhanced_results: Dict):
    """
    Create detailed visualization showing differences between audio files
    Runtime: O(n) where n is the number of audio samples
    """
    print(f"\n{'='*60}")
    print("DETAILED COMPARISON VISUALIZATION")
    print(f"{'='*60}")
    
    # Load audio files
    audio1, sr1 = librosa.load(audio_path1, sr=None)
    audio2, sr2 = librosa.load(audio_path2, sr=None)
    
    # Create comprehensive subplots
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    
    # Row 1: Waveforms
    axes[0, 0].plot(audio1)
    axes[0, 0].set_title(f"Audio 1: {audio_path1.split('/')[-1]}")
    axes[0, 0].set_xlabel("Sample")
    axes[0, 0].set_ylabel("Amplitude")
    
    axes[0, 1].plot(audio2)
    axes[0, 1].set_title(f"Audio 2: {audio_path2.split('/')[-1]}")
    axes[0, 1].set_xlabel("Sample")
    axes[0, 1].set_ylabel("Amplitude")
    
    # Difference plot
    min_len = min(len(audio1), len(audio2))
    audio_diff = audio1[:min_len] - audio2[:min_len]
    axes[0, 2].plot(audio_diff)
    axes[0, 2].set_title("Waveform Difference")
    axes[0, 2].set_xlabel("Sample")
    axes[0, 2].set_ylabel("Difference")
    
    # Row 2: MFCC features
    mfcc1 = librosa.feature.mfcc(y=audio1, sr=sr1, n_mfcc=13)
    mfcc2 = librosa.feature.mfcc(y=audio2, sr=sr2, n_mfcc=13)
    
    axes[1, 0].imshow(mfcc1, aspect='auto', origin='lower')
    axes[1, 0].set_title("MFCC - Audio 1")
    axes[1, 0].set_xlabel("Frame")
    axes[1, 0].set_ylabel("MFCC Coefficient")
    
    axes[1, 1].imshow(mfcc2, aspect='auto', origin='lower')
    axes[1, 1].set_title("MFCC - Audio 2")
    axes[1, 1].set_xlabel("Frame")
    axes[1, 1].set_ylabel("MFCC Coefficient")
    
    # MFCC difference
    min_frames = min(mfcc1.shape[1], mfcc2.shape[1])
    mfcc_diff = mfcc1[:, :min_frames] - mfcc2[:, :min_frames]
    axes[1, 2].imshow(mfcc_diff, aspect='auto', origin='lower')
    axes[1, 2].set_title("MFCC Difference")
    axes[1, 2].set_xlabel("Frame")
    axes[1, 2].set_ylabel("MFCC Coefficient")
    
    # Row 3: Chroma features (what AcoustID uses)
    chroma1 = librosa.feature.chroma_stft(y=audio1, sr=sr1)
    chroma2 = librosa.feature.chroma_stft(y=audio2, sr=sr2)
    
    axes[2, 0].imshow(chroma1, aspect='auto', origin='lower')
    axes[2, 0].set_title("Chroma - Audio 1")
    axes[2, 0].set_xlabel("Frame")
    axes[2, 0].set_ylabel("Chroma")
    
    axes[2, 1].imshow(chroma2, aspect='auto', origin='lower')
    axes[2, 1].set_title("Chroma - Audio 2")
    axes[2, 1].set_xlabel("Frame")
    axes[2, 1].set_ylabel("Chroma")
    
    # Chroma difference
    min_frames = min(chroma1.shape[1], chroma2.shape[1])
    chroma_diff = chroma1[:, :min_frames] - chroma2[:, :min_frames]
    axes[2, 2].imshow(chroma_diff, aspect='auto', origin='lower')
    axes[2, 2].set_title("Chroma Difference")
    axes[2, 2].set_xlabel("Frame")
    axes[2, 2].set_ylabel("Chroma")
    
    # Row 4: Spectral features
    spectral_centroid1 = librosa.feature.spectral_centroid(y=audio1, sr=sr1)
    spectral_centroid2 = librosa.feature.spectral_centroid(y=audio2, sr=sr2)
    
    axes[3, 0].plot(spectral_centroid1.T)
    axes[3, 0].set_title("Spectral Centroid - Audio 1")
    axes[3, 0].set_xlabel("Frame")
    axes[3, 0].set_ylabel("Frequency (Hz)")
    
    axes[3, 1].plot(spectral_centroid2.T)
    axes[3, 1].set_title("Spectral Centroid - Audio 2")
    axes[3, 1].set_xlabel("Frame")
    axes[3, 1].set_ylabel("Frequency (Hz)")
    
    # Spectral difference
    min_frames = min(spectral_centroid1.shape[1], spectral_centroid2.shape[1])
    spec_diff = spectral_centroid1[:, :min_frames] - spectral_centroid2[:, :min_frames]
    axes[3, 2].plot(spec_diff.T)
    axes[3, 2].set_title("Spectral Centroid Difference")
    axes[3, 2].set_xlabel("Frame")
    axes[3, 2].set_ylabel("Frequency Difference (Hz)")
    
    plt.tight_layout()
    plt.show()
    print("Detailed comparison visualization displayed")

def main_enhanced_analysis(audio_path1: str, audio_path2: str):
    """
    Main enhanced analysis function
    Runtime: O(n) where n is the number of audio samples
    """
    print("=" * 60)
    print("ENHANCED AUDIO SIMILARITY ANALYSIS")
    print("=" * 60)
    print(f"Analyzing: {audio_path1} vs {audio_path2}")
    
    # Enhanced similarity detection
    enhanced_results = enhanced_similarity_detection(audio_path1, audio_path2)
    
    # Investigate AcoustID limitation
    acoustid_analysis = investigate_acoustid_limitation(audio_path1, audio_path2)
    
    # Print detailed results
    print(f"\n{'='*50}")
    print("ENHANCED SIMILARITY RESULTS")
    print(f"{'='*50}")
    
    similarities = enhanced_results['similarities']
    print(f"\n📊 DETAILED SIMILARITY BREAKDOWN:")
    for feature, similarity in similarities.items():
        print(f"  {feature.replace('_', ' ').title()}: {similarity:.3f}")
    
    print(f"\n🎯 OVERALL SIMILARITY: {enhanced_results['overall_similarity']:.3f}")
    
    # Interpret results
    overall = enhanced_results['overall_similarity']
    if overall >= 0.8:
        interpretation = "VERY SIMILAR"
    elif overall >= 0.6:
        interpretation = "SIMILAR"
    elif overall >= 0.4:
        interpretation = "SOMEWHAT SIMILAR"
    elif overall >= 0.2:
        interpretation = "SLIGHTLY SIMILAR"
    else:
        interpretation = "DIFFERENT"
    
    print(f"📝 INTERPRETATION: {interpretation}")
    
    # Create visualization
    create_detailed_comparison_visualization(audio_path1, audio_path2, enhanced_results)
    
    return {
        'enhanced_results': enhanced_results,
        'acoustid_analysis': acoustid_analysis
    }

# Main execution
if __name__ == "__main__":
    audio1_path = "audio/test_audio.wav"
    audio2_path = "audio/different_audio.wav"
    
    results = main_enhanced_analysis(audio1_path, audio2_path)
    
    print(f"\n{'='*60}")
    print("ENHANCED ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"\n🔍 Key Findings:")
    print(f"  • AcoustID may give false positives for different audio files")
    print(f"  • Enhanced analysis provides more accurate similarity detection")
    print(f"  • Multiple feature comparison reveals true differences")
    print(f"  • Chromaprint focuses on harmonic content, may miss other differences") 