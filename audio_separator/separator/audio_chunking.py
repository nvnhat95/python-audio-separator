"""Audio chunking utilities for processing large audio files to prevent OOM errors."""

import os
import logging
from typing import List, Optional

import soundfile as sf
from pydub import AudioSegment


class AudioChunker:
    """
    Handles splitting and merging of large audio files.

    This class provides utilities to:
    - Split large audio files into fixed-duration chunks
    - Merge processed chunks back together with simple concatenation
    - Determine if a file should be chunked based on its duration

    Split/merge prefer SoundFile in a streaming fashion so the full input is not loaded
    into RAM (Pydub loads entire files). Pydub is used as a fallback when SoundFile cannot
    read the format (e.g. some MP3 builds).

    Example:
        >>> chunker = AudioChunker(chunk_duration_seconds=1800)  # 30-minute chunks
        >>> chunk_paths = chunker.split_audio("long_audio.wav", "/tmp/chunks")
        >>> # Process each chunk...
        >>> output_path = chunker.merge_chunks(processed_chunks, "output.wav")
    """

    _STREAM_READ_FRAMES = 262144

    @staticmethod
    def _soundfile_format_for_path(path: str) -> str:
        _, ext = os.path.splitext(path)
        return {".wav": "WAV", ".flac": "FLAC", ".ogg": "OGG", ".oga": "OGG"}.get(ext.lower(), "WAV")

    def __init__(self, chunk_duration_seconds: float, logger: logging.Logger = None):
        """
        Initialize the AudioChunker.

        Args:
            chunk_duration_seconds: Duration of each chunk in seconds
            logger: Optional logger instance for logging operations
        """
        self.chunk_duration_ms = int(chunk_duration_seconds * 1000)
        self.logger = logger or logging.getLogger(__name__)

    def split_audio(self, input_path: str, output_dir: str) -> List[str]:
        """
        Split audio file into fixed-size chunks.

        Args:
            input_path: Path to the input audio file
            output_dir: Directory where chunk files will be saved

        Returns:
            List of paths to the created chunk files

        Raises:
            FileNotFoundError: If input file doesn't exist
            IOError: If there's an error reading or writing audio files
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        streamed = self._split_audio_soundfile_streaming(input_path, output_dir)
        if streamed is not None:
            return streamed

        self.logger.debug(
            "SoundFile streaming split unavailable for this input; falling back to Pydub "
            "(loads the entire file into memory — use WAV/FLAC or set a smaller source file for very long audio)."
        )
        return self._split_audio_pydub(input_path, output_dir)

    def _split_audio_soundfile_streaming(self, input_path: str, output_dir: str) -> Optional[List[str]]:
        try:
            info = sf.info(input_path)
        except Exception as e:
            self.logger.debug(f"SoundFile info failed for streaming split: {e}")
            return None

        if info.frames <= 0:
            return None

        chunk_frames = int((self.chunk_duration_ms / 1000.0) * info.samplerate)
        if chunk_frames <= 0:
            return None

        frames_total = info.frames
        num_chunks = (frames_total + chunk_frames - 1) // chunk_frames
        self.logger.info(
            f"Streaming split: {frames_total / info.samplerate:.1f}s audio into {num_chunks} chunks "
            f"of {self.chunk_duration_ms / 1000:.1f}s each (SoundFile)"
        )

        chunk_paths: List[str] = []
        try:
            with sf.SoundFile(input_path) as infile:
                sr = infile.samplerate
                i = 0
                while infile.tell() < infile.frames:
                    nread = min(chunk_frames, infile.frames - infile.tell())
                    data = infile.read(nread, dtype="float32", always_2d=True)
                    if data.size == 0:
                        break
                    chunk_path = os.path.join(output_dir, f"chunk_{i:04d}.wav")
                    sf.write(chunk_path, data, sr, subtype="PCM_16", format="WAV")
                    chunk_paths.append(chunk_path)
                    self.logger.debug(f"Wrote chunk {i + 1}/{num_chunks}: {chunk_path}")
                    i += 1
        except Exception as e:
            self.logger.warning(f"SoundFile streaming split failed: {e}")
            for p in chunk_paths:
                try:
                    os.remove(p)
                except OSError:
                    pass
            return None

        return chunk_paths if chunk_paths else None

    def _split_audio_pydub(self, input_path: str, output_dir: str) -> List[str]:
        self.logger.debug(f"Loading audio file with Pydub: {input_path}")
        audio = AudioSegment.from_file(input_path)

        total_duration_ms = len(audio)
        chunk_paths = []

        num_chunks = (total_duration_ms + self.chunk_duration_ms - 1) // self.chunk_duration_ms
        self.logger.info(
            f"Splitting {total_duration_ms / 1000:.1f}s audio into {num_chunks} chunks of {self.chunk_duration_ms / 1000:.1f}s each"
        )

        _, ext = os.path.splitext(input_path)
        if not ext:
            ext = ".wav"

        for i in range(num_chunks):
            start_ms = i * self.chunk_duration_ms
            end_ms = min(start_ms + self.chunk_duration_ms, total_duration_ms)

            chunk = audio[start_ms:end_ms]
            chunk_filename = f"chunk_{i:04d}{ext}"
            chunk_path = os.path.join(output_dir, chunk_filename)

            self.logger.debug(
                f"Exporting chunk {i + 1}/{num_chunks}: {start_ms / 1000:.1f}s - {end_ms / 1000:.1f}s to {chunk_path}"
            )
            chunk.export(chunk_path, format=ext.lstrip("."))
            chunk_paths.append(chunk_path)

        return chunk_paths

    def merge_chunks(self, chunk_paths: List[str], output_path: str) -> str:
        """
        Merge processed chunks with simple concatenation.

        Args:
            chunk_paths: List of paths to chunk files to merge
            output_path: Path where the merged output will be saved

        Returns:
            Path to the merged output file

        Raises:
            ValueError: If chunk_paths is empty
            FileNotFoundError: If any chunk file doesn't exist
            IOError: If there's an error reading or writing audio files
        """
        if not chunk_paths:
            raise ValueError("Cannot merge empty list of chunks")

        for chunk_path in chunk_paths:
            if not os.path.exists(chunk_path):
                raise FileNotFoundError(f"Chunk file not found: {chunk_path}")

        self.logger.info(f"Merging {len(chunk_paths)} chunks into {output_path}")

        _, ext = os.path.splitext(output_path)
        ext_l = ext.lower()

        if ext_l in (".mp3", ".m4a", ".ogg", ".opus"):
            self.logger.debug("Lossy/container output format: using Pydub merge (may use substantial memory).")
            return self._merge_chunks_pydub(chunk_paths, output_path)

        if self._merge_chunks_soundfile_streaming(chunk_paths, output_path):
            return output_path

        self.logger.debug("SoundFile streaming merge failed; falling back to Pydub.")
        return self._merge_chunks_pydub(chunk_paths, output_path)

    def _merge_chunks_soundfile_streaming(self, chunk_paths: List[str], output_path: str) -> bool:
        try:
            info0 = sf.info(chunk_paths[0])
            sr = info0.samplerate
            channels = info0.channels
            subtype = info0.subtype if info0.subtype else "PCM_16"
            out_fmt = self._soundfile_format_for_path(output_path)

            out_dir = os.path.dirname(os.path.abspath(output_path))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

            with sf.SoundFile(
                output_path,
                mode="w",
                samplerate=sr,
                channels=channels,
                subtype=subtype,
                format=out_fmt,
            ) as outf:
                for i, chunk_path in enumerate(chunk_paths):
                    self.logger.debug(f"Streaming merge: reading chunk {i + 1}/{len(chunk_paths)}: {chunk_path}")
                    with sf.SoundFile(chunk_path) as inf:
                        if inf.samplerate != sr or inf.channels != channels:
                            self.logger.debug(
                                f"Chunk format mismatch (sr={inf.samplerate}, ch={inf.channels}) vs output "
                                f"(sr={sr}, ch={channels}); aborting SoundFile merge."
                            )
                            return False
                        while inf.tell() < inf.frames:
                            block = inf.read(self._STREAM_READ_FRAMES, dtype="float32", always_2d=True)
                            if block.size == 0:
                                break
                            outf.write(block)
            self.logger.info(f"Exported merged audio to {output_path} (streaming)")
            return True
        except Exception as e:
            self.logger.warning(f"SoundFile streaming merge failed: {e}")
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass
            return False

    def _merge_chunks_pydub(self, chunk_paths: List[str], output_path: str) -> str:
        combined = AudioSegment.empty()

        for i, chunk_path in enumerate(chunk_paths):
            self.logger.debug(f"Loading chunk {i + 1}/{len(chunk_paths)}: {chunk_path}")
            chunk = AudioSegment.from_file(chunk_path)
            combined += chunk

        _, ext = os.path.splitext(output_path)
        output_format = ext.lstrip(".").lower() if ext else "wav"
        if output_format == "m4a":
            output_format = "mp4"

        self.logger.info(f"Exporting merged audio ({len(combined) / 1000:.1f}s) to {output_path}")
        combined.export(output_path, format=output_format)
        return output_path

    def should_chunk(self, audio_duration_seconds: float) -> bool:
        """
        Determine if file is large enough to benefit from chunking.

        Args:
            audio_duration_seconds: Duration of the audio in seconds

        Returns:
            True if the file should be chunked, False otherwise
        """
        return audio_duration_seconds > (self.chunk_duration_ms / 1000)
