"""face_tracking — open-source video face tracking based on Pixel3DMM.

This package provides a thin, self-contained re-implementation of the
Pixel3DMM tracking pipeline so that any 3D face landmarks (FLAME-aligned)
can be fitted to a video with no external infrastructure dependencies.
"""

from face_tracking.api import track_video, track_videos  # noqa: F401

__version__ = "0.1.0"
