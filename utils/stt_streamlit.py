# utils/stt_streamlit.py

class AudioCaptureProcessor:
    """
    A Streamlit WebRTC audio processor that buffers audio using recv_queued().
    Ensures reliable audio capture for Whisper STT and supports waveform preview.
    """

    def __init__(self):
        self.buffer = b""  # Stores concatenated audio bytes

    def recv_queued(self, frames):
        """
        Streamlit calls this with a list of audio frames.
        This method prevents frame drops and enables waveform display.
        """
        for frame in frames:
            self.buffer += frame.to_ndarray().tobytes()
        return frames[-1]  # Returning the last frame preserves waveform display

    def get_audio(self):
        """
        Return the collected raw audio buffer for Whisper STT or export.
        """
        return self.buffer
