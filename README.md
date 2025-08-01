# Audio Fingerprinting

A fast prototype for audio fingerprinting to test the Chromprint framework prior to implementation in Android and iOS mobile applications.

## Usage

Under `audio/` there exists a wave file `test_audio.wav` which is the power up sound for the Aero Pro. The current goal is to set up the fingerprinting pipeline and observe what happens when comparing the sound to a copy of itself, a completely different audio file, and a modified or similar copy of the sound as well as what happens when the subject audio file is padded with silence around the waveform, ambient sounds like room noises and reverberance and coloration due to the Aero speakers frequency profile.

Audio files are not included in the repository. Therefor if you are running the code as a third party you must provide your own audio files.

To run the script:

```bash
python acoustic_explorer.py
```