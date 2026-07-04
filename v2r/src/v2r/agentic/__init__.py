"""Agentic labeler: an AI agent that inspects a video (AI-generated or real)
with perception tools — VLM frame analysis, MediaPipe pose/hands, YOLO object
detection, motion analysis — decides which tools apply, and writes the labels
into the episode workspace with honest confidence and source tags."""

from .labeler import run_agentic_labeler  # noqa: F401
