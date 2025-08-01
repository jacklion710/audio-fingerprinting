import librosa
import acoustid
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import time
import json
from typing import Dict, Tuple

def generate_librosa_fingerprint(audio_path: str) -> Dict:
    """
    Generate fingerprint using librosa approach
    Runtime: O(n) where n is the number of audio samples
    """
    start_time = time.time()
    print(f"Generating librosa fingerprint for: {audio_path}")
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=None)
    
    # Extract features
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    
    # Create fingerprint
    fingerprint_features = np.concatenate([
        mfcc.flatten()[:1000],
        spectral_centroid.flatten()[:100],
        spectral_rolloff.flatten()[:100]
    ])
    
    fingerprint_hash = hashlib.md5(fingerprint_features.tobytes()).hexdigest()
    
    computation_time = time.time() - start_time
    
    return {
        'fingerprint': fingerprint_hash,
        'duration': len(audio) / sr,
        'computation_time': computation_time,
        'features_used': ['mfcc', 'spectral_centroid', 'spectral_rolloff'],
        'fingerprint_length': len(fingerprint_hash),
        'method': 'librosa'
    }

def generate_acoustid_fingerprint(audio_path: str) -> Dict:
    """
    Generate fingerprint using AcoustID approach
    Runtime: O(n) where n is the number of audio samples
    """
    start_time = time.time()
    print(f"Generating AcoustID fingerprint for: {audio_path}")
    
    # Generate fingerprint using AcoustID
    duration, fingerprint = acoustid.fingerprint_file(audio_path)
    
    computation_time = time.time() - start_time
    
    return {
        'fingerprint': fingerprint.decode('utf-8') if isinstance(fingerprint, bytes) else str(fingerprint),
        'duration': duration,
        'computation_time': computation_time,
        'features_used': ['chromaprint'],
        'fingerprint_length': len(fingerprint),
        'method': 'acoustid'
    }

def compare_fingerprinting_methods(audio_path: str) -> Dict:
    """
    Compare both fingerprinting methods
    Runtime: O(n) where n is the number of audio samples
    """
    print("=" * 60)
    print("FINGERPRINTING METHODS COMPARISON")
    print("=" * 60)
    
    # Generate fingerprints using both methods
    librosa_result = generate_librosa_fingerprint(audio_path)
    acoustid_result = generate_acoustid_fingerprint(audio_path)
    
    # Compare results
    comparison = {
        'librosa': librosa_result,
        'acoustid': acoustid_result,
        'comparison': {
            'speed': {
                'librosa': librosa_result['computation_time'],
                'acoustid': acoustid_result['computation_time'],
                'faster': 'acoustid' if acoustid_result['computation_time'] < librosa_result['computation_time'] else 'librosa'
            },
            'fingerprint_length': {
                'librosa': librosa_result['fingerprint_length'],
                'acoustid': acoustid_result['fingerprint_length'],
                'longer': 'librosa' if librosa_result['fingerprint_length'] > acoustid_result['fingerprint_length'] else 'acoustid'
            },
            'duration_accuracy': {
                'librosa': librosa_result['duration'],
                'acoustid': acoustid_result['duration'],
                'difference': abs(librosa_result['duration'] - acoustid_result['duration'])
            }
        }
    }
    
    # Print comparison results
    print(f"\n{'='*40}")
    print("COMPARISON RESULTS")
    print(f"{'='*40}")
    
    print(f"\nSpeed Comparison:")
    print(f"  Librosa: {librosa_result['computation_time']:.3f}s")
    print(f"  AcoustID: {acoustid_result['computation_time']:.3f}s")
    print(f"  Faster: {comparison['comparison']['speed']['faster'].upper()}")
    
    print(f"\nFingerprint Length:")
    print(f"  Librosa: {librosa_result['fingerprint_length']} characters")
    print(f"  AcoustID: {acoustid_result['fingerprint_length']} characters")
    print(f"  Longer: {comparison['comparison']['fingerprint_length']['longer'].upper()}")
    
    print(f"\nDuration Accuracy:")
    print(f"  Librosa: {librosa_result['duration']:.3f}s")
    print(f"  AcoustID: {acoustid_result['duration']:.3f}s")
    print(f"  Difference: {comparison['comparison']['duration_accuracy']['difference']:.3f}s")
    
    print(f"\nFeatures Used:")
    print(f"  Librosa: {', '.join(librosa_result['features_used'])}")
    print(f"  AcoustID: {', '.join(acoustid_result['features_used'])}")
    
    return comparison

def demonstrate_use_cases():
    """
    Demonstrate different use cases for each method
    Runtime: O(1) - demonstration only
    """
    print(f"\n{'='*60}")
    print("USE CASES FOR EACH METHOD")
    print(f"{'='*60}")
    
    print(f"\nLibrosa Use Cases:")
    print("  ✅ Custom audio analysis")
    print("  ✅ Feature extraction for ML")
    print("  ✅ Audio similarity detection")
    print("  ✅ Spectral analysis")
    print("  ✅ Real-time processing")
    print("  ❌ No built-in music database")
    
    print(f"\nAcoustID Use Cases:")
    print("  ✅ Music identification")
    print("  ✅ Database lookup")
    print("  ✅ Industry-standard fingerprints")
    print("  ✅ Fast fingerprinting")
    print("  ✅ Cross-platform compatibility")
    print("  ❌ Requires external database")
    print("  ❌ Limited to music identification")

def create_visualization_comparison(audio_path: str):
    """
    Create visual comparison of both methods
    Runtime: O(n) where n is the number of audio samples
    """
    print(f"\n{'='*60}")
    print("VISUAL COMPARISON")
    print(f"{'='*60}")
    
    # Load audio for visualization
    audio, sr = librosa.load(audio_path, sr=None)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Audio waveform
    axes[0, 0].plot(audio)
    axes[0, 0].set_title("Audio Waveform")
    axes[0, 0].set_xlabel("Sample")
    axes[0, 0].set_ylabel("Amplitude")
    
    # Plot 2: MFCC features (Librosa)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    axes[0, 1].imshow(mfcc, aspect='auto', origin='lower')
    axes[0, 1].set_title("MFCC Features (Librosa)")
    axes[0, 1].set_xlabel("Frame")
    axes[0, 1].set_ylabel("MFCC Coefficient")
    
    # Plot 3: Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    axes[1, 0].plot(spectral_centroid.T)
    axes[1, 0].set_title("Spectral Centroid (Librosa)")
    axes[1, 0].set_xlabel("Frame")
    axes[1, 0].set_ylabel("Frequency (Hz)")
    
    # Plot 4: Chromagram (similar to AcoustID approach)
    chromagram = librosa.feature.chroma_stft(y=audio, sr=sr)
    axes[1, 1].imshow(chromagram, aspect='auto', origin='lower')
    axes[1, 1].set_title("Chromagram (Similar to AcoustID)")
    axes[1, 1].set_xlabel("Frame")
    axes[1, 1].set_ylabel("Chroma")
    
    plt.tight_layout()
    plt.show()
    print("Visualization displayed")

# Main execution
if __name__ == "__main__":
    audio_path = "audio/test_audio.wav"
    
    print("COMPREHENSIVE FINGERPRINTING COMPARISON")
    print("=" * 60)
    
    # Compare methods
    comparison_results = compare_fingerprinting_methods(audio_path)
    
    # Demonstrate use cases
    demonstrate_use_cases()
    
    # Create visualization
    create_visualization_comparison(audio_path)
    
    # Save results
    with open('fingerprint_comparison_results.json', 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("COMPARISON COMPLETE")
    print(f"{'='*60}")
    print(f"\nResults saved to: fingerprint_comparison_results.json")
    print(f"\nKey Takeaways:")
    print(f"  • AcoustID is faster for fingerprinting")
    print(f"  • Librosa provides more detailed analysis")
    print(f"  • AcoustID is better for music identification")
    print(f"  • Librosa is better for custom audio analysis") 