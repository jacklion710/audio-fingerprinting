import librosa
import acoustid
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import time
from typing import Dict, Tuple
from difflib import SequenceMatcher

def calculate_fingerprint_similarity(fingerprint1: str, fingerprint2: str) -> float:
    """
    Calculate similarity between two fingerprints using sequence matching
    Runtime: O(n) where n is the length of the shorter fingerprint
    """
    # Use sequence matcher for character-by-character comparison
    similarity = SequenceMatcher(None, fingerprint1, fingerprint2).ratio()
    return similarity

def compare_audio_files_librosa(audio_path1: str, audio_path2: str) -> Dict:
    """
    Compare two audio files using librosa features
    Runtime: O(n) where n is the number of audio samples
    """
    print(f"Comparing audio files using Librosa method...")
    start_time = time.time()
    
    # Load both audio files
    audio1, sr1 = librosa.load(audio_path1, sr=None)
    audio2, sr2 = librosa.load(audio_path2, sr=None)
    
    # Extract features for both files
    mfcc1 = librosa.feature.mfcc(y=audio1, sr=sr1, n_mfcc=13)
    mfcc2 = librosa.feature.mfcc(y=audio2, sr=sr2, n_mfcc=13)
    
    spectral_centroid1 = librosa.feature.spectral_centroid(y=audio1, sr=sr1)
    spectral_centroid2 = librosa.feature.spectral_centroid(y=audio2, sr=sr2)
    
    spectral_rolloff1 = librosa.feature.spectral_rolloff(y=audio1, sr=sr1)
    spectral_rolloff2 = librosa.feature.spectral_rolloff(y=audio2, sr=sr2)
    
    # Create fingerprints
    features1 = np.concatenate([
        mfcc1.flatten()[:1000],
        spectral_centroid1.flatten()[:100],
        spectral_rolloff1.flatten()[:100]
    ])
    
    features2 = np.concatenate([
        mfcc2.flatten()[:1000],
        spectral_centroid2.flatten()[:100],
        spectral_rolloff2.flatten()[:100]
    ])
    
    fingerprint1 = hashlib.md5(features1.tobytes()).hexdigest()
    fingerprint2 = hashlib.md5(features2.tobytes()).hexdigest()
    
    # Calculate similarities
    fingerprint_similarity = calculate_fingerprint_similarity(fingerprint1, fingerprint2)
    
    # Calculate feature similarities
    mfcc_similarity = np.corrcoef(mfcc1.flatten()[:1000], mfcc2.flatten()[:1000])[0, 1]
    centroid_similarity = np.corrcoef(spectral_centroid1.flatten()[:100], spectral_centroid2.flatten()[:100])[0, 1]
    rolloff_similarity = np.corrcoef(spectral_rolloff1.flatten()[:100], spectral_rolloff2.flatten()[:100])[0, 1]
    
    computation_time = time.time() - start_time
    
    return {
        'method': 'librosa',
        'fingerprint1': fingerprint1,
        'fingerprint2': fingerprint2,
        'fingerprint_similarity': fingerprint_similarity,
        'mfcc_similarity': mfcc_similarity,
        'centroid_similarity': centroid_similarity,
        'rolloff_similarity': rolloff_similarity,
        'overall_similarity': (fingerprint_similarity + mfcc_similarity + centroid_similarity + rolloff_similarity) / 4,
        'computation_time': computation_time,
        'duration1': len(audio1) / sr1,
        'duration2': len(audio2) / sr2
    }

def compare_audio_files_acoustid(audio_path1: str, audio_path2: str) -> Dict:
    """
    Compare two audio files using AcoustID fingerprints
    Runtime: O(n) where n is the number of audio samples
    """
    print(f"Comparing audio files using AcoustID method...")
    start_time = time.time()
    
    # Generate fingerprints using AcoustID
    duration1, fingerprint1 = acoustid.fingerprint_file(audio_path1)
    duration2, fingerprint2 = acoustid.fingerprint_file(audio_path2)
    
    # Convert to strings for comparison
    fingerprint1_str = fingerprint1.decode('utf-8') if isinstance(fingerprint1, bytes) else str(fingerprint1)
    fingerprint2_str = fingerprint2.decode('utf-8') if isinstance(fingerprint2, bytes) else str(fingerprint2)
    
    # Calculate similarity
    fingerprint_similarity = calculate_fingerprint_similarity(fingerprint1_str, fingerprint2_str)
    
    computation_time = time.time() - start_time
    
    return {
        'method': 'acoustid',
        'fingerprint1': fingerprint1_str,
        'fingerprint2': fingerprint2_str,
        'fingerprint_similarity': fingerprint_similarity,
        'overall_similarity': fingerprint_similarity,
        'computation_time': computation_time,
        'duration1': duration1,
        'duration2': duration2
    }

def interpret_similarity(similarity_score: float) -> str:
    """
    Interpret similarity score and provide human-readable description
    Runtime: O(1)
    """
    if similarity_score >= 0.95:
        return "IDENTICAL - Same audio file"
    elif similarity_score >= 0.8:
        return "VERY SIMILAR - Likely same content with minor differences"
    elif similarity_score >= 0.6:
        return "SIMILAR - Related audio content"
    elif similarity_score >= 0.4:
        return "SOMEWHAT SIMILAR - Some common characteristics"
    elif similarity_score >= 0.2:
        return "SLIGHTLY SIMILAR - Minimal common features"
    else:
        return "DIFFERENT - Unrelated audio content"

def create_similarity_visualization(audio_path1: str, audio_path2: str, results: Dict):
    """
    Create visualization comparing the two audio files
    Runtime: O(n) where n is the number of audio samples
    """
    print(f"\n{'='*60}")
    print("SIMILARITY VISUALIZATION")
    print(f"{'='*60}")
    
    # Load audio files
    audio1, sr1 = librosa.load(audio_path1, sr=None)
    audio2, sr2 = librosa.load(audio_path2, sr=None)
    
    # Create subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # Plot 1: Audio waveforms
    axes[0, 0].plot(audio1)
    axes[0, 0].set_title(f"Audio 1: {audio_path1.split('/')[-1]}")
    axes[0, 0].set_xlabel("Sample")
    axes[0, 0].set_ylabel("Amplitude")
    
    axes[0, 1].plot(audio2)
    axes[0, 1].set_title(f"Audio 2: {audio_path2.split('/')[-1]}")
    axes[0, 1].set_xlabel("Sample")
    axes[0, 1].set_ylabel("Amplitude")
    
    # Plot 2: MFCC features
    mfcc1 = librosa.feature.mfcc(y=audio1, sr=sr1, n_mfcc=13)
    mfcc2 = librosa.feature.mfcc(y=audio2, sr=sr2, n_mfcc=13)
    
    axes[1, 0].imshow(mfcc1, aspect='auto', origin='lower')
    axes[1, 0].set_title("MFCC Features - Audio 1")
    axes[1, 0].set_xlabel("Frame")
    axes[1, 0].set_ylabel("MFCC Coefficient")
    
    axes[1, 1].imshow(mfcc2, aspect='auto', origin='lower')
    axes[1, 1].set_title("MFCC Features - Audio 2")
    axes[1, 1].set_xlabel("Frame")
    axes[1, 1].set_ylabel("MFCC Coefficient")
    
    # Plot 3: Spectral centroids
    spectral_centroid1 = librosa.feature.spectral_centroid(y=audio1, sr=sr1)
    spectral_centroid2 = librosa.feature.spectral_centroid(y=audio2, sr=sr2)
    
    axes[2, 0].plot(spectral_centroid1.T)
    axes[2, 0].set_title("Spectral Centroid - Audio 1")
    axes[2, 0].set_xlabel("Frame")
    axes[2, 0].set_ylabel("Frequency (Hz)")
    
    axes[2, 1].plot(spectral_centroid2.T)
    axes[2, 1].set_title("Spectral Centroid - Audio 2")
    axes[2, 1].set_xlabel("Frame")
    axes[2, 1].set_ylabel("Frequency (Hz)")
    
    plt.tight_layout()
    plt.show()
    print("Similarity visualization displayed")

def main_comparison(audio_path1: str, audio_path2: str):
    """
    Main function to compare two audio files
    Runtime: O(n) where n is the number of audio samples
    """
    print("=" * 60)
    print("AUDIO SIMILARITY DETECTION")
    print("=" * 60)
    print(f"Comparing: {audio_path1} vs {audio_path2}")
    print("=" * 60)
    
    # Compare using both methods
    librosa_results = compare_audio_files_librosa(audio_path1, audio_path2)
    acoustid_results = compare_audio_files_acoustid(audio_path1, audio_path2)
    
    # Print results
    print(f"\n{'='*50}")
    print("SIMILARITY RESULTS")
    print(f"{'='*50}")
    
    print(f"\n📊 LIBROSA METHOD:")
    print(f"  Fingerprint Similarity: {librosa_results['fingerprint_similarity']:.3f}")
    print(f"  MFCC Similarity: {librosa_results['mfcc_similarity']:.3f}")
    print(f"  Centroid Similarity: {librosa_results['centroid_similarity']:.3f}")
    print(f"  Rolloff Similarity: {librosa_results['rolloff_similarity']:.3f}")
    print(f"  Overall Similarity: {librosa_results['overall_similarity']:.3f}")
    print(f"  Computation Time: {librosa_results['computation_time']:.3f}s")
    print(f"  Interpretation: {interpret_similarity(librosa_results['overall_similarity'])}")
    
    print(f"\n🎵 ACOUSTID METHOD:")
    print(f"  Fingerprint Similarity: {acoustid_results['fingerprint_similarity']:.3f}")
    print(f"  Overall Similarity: {acoustid_results['overall_similarity']:.3f}")
    print(f"  Computation Time: {acoustid_results['computation_time']:.3f}s")
    print(f"  Interpretation: {interpret_similarity(acoustid_results['overall_similarity'])}")
    
    print(f"\n📈 COMPARISON SUMMARY:")
    print(f"  Librosa Overall: {librosa_results['overall_similarity']:.3f}")
    print(f"  AcoustID Overall: {acoustid_results['overall_similarity']:.3f}")
    
    # Determine which method is more confident
    if abs(librosa_results['overall_similarity'] - acoustid_results['overall_similarity']) < 0.1:
        print(f"  Consensus: Both methods agree on similarity level")
    elif librosa_results['overall_similarity'] > acoustid_results['overall_similarity']:
        print(f"  Consensus: Librosa detects higher similarity")
    else:
        print(f"  Consensus: AcoustID detects higher similarity")
    
    # Create visualization
    create_similarity_visualization(audio_path1, audio_path2, librosa_results)
    
    return {
        'librosa': librosa_results,
        'acoustid': acoustid_results
    }

# Main execution
if __name__ == "__main__":
    audio1_path = "audio/test_audio.wav"
    audio2_path = "audio/different_audio.wav"
    
    results = main_comparison(audio1_path, audio2_path)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"\n💡 Key Insights:")
    print(f"  • Librosa provides detailed feature-by-feature comparison")
    print(f"  • AcoustID provides fast, industry-standard comparison")
    print(f"  • Both methods can detect audio similarity effectively")
    print(f"  • Use this for detecting duplicate audio, similar sounds, or audio variations") 