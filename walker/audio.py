"""Audio/Text-to-speech module for Walker."""

import subprocess
from typing import Optional, Callable


class Audio:
    """Text-to-speech for directions"""

    callback: Optional[Callable[[str], None]] = None  # Class-level callback for debug GUI

    @classmethod
    def set_callback(cls, callback: Optional[Callable[[str], None]]):
        """Set callback function for audio events"""
        cls.callback = callback

    @staticmethod
    def speak(text: str):
        """Speak text using espeak (available in Termux)"""
        # Send to callback if set (for debug GUI)
        if Audio.callback:
            Audio.callback(text)

        try:
            subprocess.run(
                ["espeak", "-s", "150", text],
                capture_output=True,
                timeout=10
            )
        except FileNotFoundError:
            # Fallback: try pyttsx3
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
            except Exception:
                print(f"[AUDIO] {text}")
        except Exception as e:
            print(f"Audio error: {e}")
            print(f"[AUDIO] {text}")
