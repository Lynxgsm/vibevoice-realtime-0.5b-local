#!/usr/bin/env python3
"""
Download the VibeVoice-Realtime-0.5B model from Hugging Face.

This mirrors the Colab notebook's snapshot_download step.
"""
import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(
        description="Download VibeVoice-Realtime model from Hugging Face"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="microsoft/VibeVoice-Realtime-0.5B",
        help="Hugging Face model ID (default: microsoft/VibeVoice-Realtime-0.5B)",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default="models/VibeVoice-Realtime-0.5B",
        help="Local directory to save the model (default: models/VibeVoice-Realtime-0.5B)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token (or set HF_TOKEN env var). Required if model is gated.",
    )
    args = parser.parse_args()

    # Get token from args, env var, or None
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    local_dir = Path(args.local_dir).resolve()
    model_id = args.model_id

    print(f"📥 Downloading model: {model_id}")
    print(f"📁 Destination: {local_dir}")

    # Check if model already exists
    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"⚠️  Directory {local_dir} already exists and is not empty.")
        response = input("Do you want to re-download? (y/N): ").strip().lower()
        if response != "y":
            print("✅ Skipping download (using existing model)")
            return

    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            token=token,
            local_dir_use_symlinks=False,
        )
        print(f"✅ Downloaded model: {model_id}")
        print(f"✅ Model saved to: {local_dir}")
    except Exception as e:
        error_msg = str(e)
        if "authentication" in error_msg.lower() or "401" in error_msg or "403" in error_msg:
            print(f"❌ Authentication error: {error_msg}")
            print("\n💡 To download gated models, you need to:")
            print("   1. Log in to Hugging Face: huggingface-cli login")
            print("   2. Or set HF_TOKEN environment variable")
            print("   3. Or pass --token <your_token>")
            sys.exit(1)
        else:
            print(f"❌ Error downloading model: {error_msg}")
            raise


if __name__ == "__main__":
    main()

