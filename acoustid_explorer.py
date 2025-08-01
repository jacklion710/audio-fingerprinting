import acoustid
import time
import json
from typing import Dict, List, Optional, Tuple

def explore_acoustid_fingerprinting(audio_path: str, api_key: Optional[str] = None) -> Dict:
    """
    Explore AcoustID fingerprinting capabilities
    Runtime: O(n) where n is the number of audio samples
    """
    start_time = time.time()
    print(f"Starting AcoustID fingerprinting exploration for: {audio_path}")
    
    results = {
        'fingerprint': None,
        'duration': None,
        'api_key_provided': api_key is not None,
        'lookup_results': None,
        'error': None
    }
    
    try:
        # Generate fingerprint using AcoustID
        print("Generating AcoustID fingerprint...")
        fingerprint_start = time.time()
        duration, fingerprint = acoustid.fingerprint_file(audio_path)
        fingerprint_time = time.time() - fingerprint_start
        
        results['fingerprint'] = fingerprint.decode('utf-8') if isinstance(fingerprint, bytes) else str(fingerprint)
        results['duration'] = duration
        
        print(f"Fingerprint generated in {fingerprint_time:.3f}s")
        print(f"Audio duration: {duration:.2f} seconds")
        print(f"Fingerprint length: {len(fingerprint)} characters")
        print(f"Fingerprint preview: {fingerprint[:50]}...")
        
        # If API key is provided, try to lookup the fingerprint
        if api_key:
            print("\nAttempting to lookup fingerprint in AcoustID database...")
            lookup_start = time.time()
            
            try:
                # Lookup the fingerprint
                lookup_results = acoustid.lookup(api_key, fingerprint, duration)
                lookup_time = time.time() - lookup_start
                
                results['lookup_results'] = lookup_results
                
                print(f"Lookup completed in {lookup_time:.3f}s")
                print(f"Found {len(lookup_results)} potential matches")
                
                # Display results
                for i, result in enumerate(lookup_results):
                    print(f"\nMatch {i+1}:")
                    print(f"  Score: {result.score:.2f}")
                    print(f"  Recording ID: {result.recordings[0].id if result.recordings else 'N/A'}")
                    if result.recordings:
                        recording = result.recordings[0]
                        print(f"  Title: {recording.title}")
                        print(f"  Artist: {recording.artists[0].name if recording.artists else 'Unknown'}")
                        print(f"  Release: {recording.releases[0].title if recording.releases else 'Unknown'}")
                        
            except acoustid.WebServiceError as e:
                print(f"Web service error during lookup: {e}")
                results['error'] = f"Web service error: {e}"
            except Exception as e:
                print(f"Error during lookup: {e}")
                results['error'] = f"Lookup error: {e}"
        else:
            print("\nNo API key provided. Skipping database lookup.")
            print("To enable database lookup, get a free API key from: https://acoustid.org/")
        
        total_time = time.time() - start_time
        print(f"\nTotal exploration time: {total_time:.3f}s")
        
    except acoustid.FingerprintGenerationError as e:
        print(f"Error generating fingerprint: {e}")
        results['error'] = f"Fingerprint generation error: {e}"
    except Exception as e:
        print(f"Unexpected error: {e}")
        results['error'] = f"Unexpected error: {e}"
    
    return results

def compare_fingerprints(audio_path1: str, audio_path2: str) -> Dict:
    """
    Compare fingerprints of two audio files
    Runtime: O(n) where n is the number of audio samples
    """
    print(f"\n{'='*50}")
    print("COMPARING FINGERPRINTS")
    print(f"{'='*50}")
    
    comparison_results = {
        'file1': {'path': audio_path1, 'fingerprint': None, 'duration': None},
        'file2': {'path': audio_path2, 'fingerprint': None, 'duration': None},
        'similarity': None,
        'errors': []
    }
    
    try:
        # Generate fingerprints for both files
        print(f"Generating fingerprint for: {audio_path1}")
        duration1, fingerprint1 = acoustid.fingerprint_file(audio_path1)
        comparison_results['file1']['fingerprint'] = fingerprint1
        comparison_results['file1']['duration'] = duration1
        
        print(f"Generating fingerprint for: {audio_path2}")
        duration2, fingerprint2 = acoustid.fingerprint_file(audio_path2)
        comparison_results['file2']['fingerprint'] = fingerprint2
        comparison_results['file2']['duration'] = duration2
        
        # Calculate similarity (simple character-by-character comparison)
        similarity = sum(1 for a, b in zip(fingerprint1, fingerprint2) if a == b) / max(len(fingerprint1), len(fingerprint2))
        comparison_results['similarity'] = similarity
        
        print(f"\nComparison Results:")
        print(f"File 1 duration: {duration1:.2f}s")
        print(f"File 2 duration: {duration2:.2f}s")
        print(f"Fingerprint similarity: {similarity:.2%}")
        
        if similarity > 0.9:
            print("Result: Very similar (likely same audio)")
        elif similarity > 0.7:
            print("Result: Similar (possibly related audio)")
        elif similarity > 0.5:
            print("Result: Somewhat similar")
        else:
            print("Result: Different audio files")
            
    except Exception as e:
        print(f"Error during comparison: {e}")
        comparison_results['errors'].append(str(e))
    
    return comparison_results

def demonstrate_acoustid_features():
    """
    Demonstrate various AcoustID features and capabilities
    Runtime: O(1) - demonstration only
    """
    print(f"\n{'='*50}")
    print("ACOUSTID FEATURES DEMONSTRATION")
    print(f"{'='*50}")
    
    print("\n1. Fingerprint Generation:")
    print("   - Uses Chromaprint algorithm")
    print("   - Generates base64-encoded fingerprints")
    print("   - Works with various audio formats")
    print("   - Provides duration information")
    
    print("\n2. Database Lookup:")
    print("   - Query AcoustID database with fingerprints")
    print("   - Returns music metadata (artist, title, album)")
    print("   - Provides confidence scores")
    print("   - Requires free API key from acoustid.org")
    
    print("\n3. Supported Audio Formats:")
    print("   - MP3, FLAC, WAV, OGG, M4A, etc.")
    print("   - Automatic format detection")
    print("   - Handles various sample rates")
    
    print("\n4. Use Cases:")
    print("   - Music identification")
    print("   - Audio file deduplication")
    print("   - Content matching")
    print("   - Audio fingerprinting for mobile apps")

# Main execution
if __name__ == "__main__":
    audio_path = "audio/test_audio.wav"
    
    print("Exploring AcoustID Package")
    print("=" * 50)
    
    # Demonstrate AcoustID features
    demonstrate_acoustid_features()
    
    # Explore fingerprinting (without API key for now)
    print(f"\n{'='*50}")
    print("FINGERPRINTING EXPLORATION")
    print(f"{'='*50}")
    
    results = explore_acoustid_fingerprinting(audio_path)
    
    # Save results to JSON for analysis
    with open('acoustid_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: acoustid_results.json")
    
    # If you have multiple audio files, you can compare them
    # Uncomment the following lines if you have multiple audio files:
    # compare_fingerprints("audio/test_audio.wav", "audio/another_audio.wav")
    
    print(f"\n{'='*50}")
    print("EXPLORATION COMPLETE")
    print(f"{'='*50}")
    print("\nTo enable database lookup:")
    print("1. Get a free API key from: https://acoustid.org/")
    print("2. Pass the API key to explore_acoustid_fingerprinting()")
    print("3. The script will then query the AcoustID database") 