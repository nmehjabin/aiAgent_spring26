# ============================================================
# Exercise 2: Video-Surveillance Agent (Gradio version)
# Works locally on Mac/PC — opens a web UI in your browser
# ============================================================
# SETUP: run once in terminal:
#   pip install opencv-python ollama gradio Pillow edge-tts
#
# RUN:
#   python3 video_surveillance_agent.py
#   Then open http://localhost:7860 in your browser
# ============================================================

import cv2
import base64
import io
import time
import asyncio
import subprocess
import threading
from datetime import timedelta

import ollama
import numpy as np
import gradio as gr
from PIL import Image as PILImage

# edge-tts is optional
try:
    import edge_tts
    import nest_asyncio
    nest_asyncio.apply()
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("  edge-tts not installed — TTS alerts disabled.")


# ============================================================
# HELPERS
# ============================================================

def frame_to_base64(frame: np.ndarray, max_size: int = 512) -> str:
    """Convert OpenCV BGR frame to resized base64 JPEG string."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = PILImage.fromarray(rgb)
    img.thumbnail((max_size, max_size), PILImage.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def seconds_to_timestamp(seconds: float) -> str:
    """Convert float seconds to MM:SS string."""
    return str(timedelta(seconds=int(seconds)))[2:]


def ask_llava(frame_b64: str, prompt: str, model: str = "llava") -> str:
    """Send a frame + prompt to LLaVA, return text response."""
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt, "images": [frame_b64]}]
    )
    return response["message"]["content"].strip()


def parse_detection(response: str) -> dict:
    """
    Parse LLaVA free-text into structured detection.
    LLaVA returns natural language so we use keyword search.
    """
    text = response.lower()

    person = (
        any(w in text for w in ["person", "people", "human", "man", "woman", "someone", "figure"])
        and not any(w in text for w in ["no person", "no people", "nobody", "no human"])
    )

    cat_count = 0
    if "cat" in text and "no cat" not in text:
        for word, val in [("two", 2), ("three", 3), ("2", 2), ("3", 3), ("one", 1), ("1", 1)]:
            if word in text:
                cat_count = val
                break
        if cat_count == 0:
            cat_count = 1

    dog_count = 0
    if "dog" in text and "no dog" not in text:
        for word, val in [("two", 2), ("three", 3), ("2", 2), ("3", 3), ("one", 1), ("1", 1)]:
            if word in text:
                dog_count = val
                break
        if dog_count == 0:
            dog_count = 1

    return {"person": person, "cat": cat_count, "dog": dog_count, "raw": response}


def speak(text: str):
    """TTS alert using edge-tts (background thread, non-blocking)."""
    if not TTS_AVAILABLE:
        return

    def _run():
        # Create a NEW event loop for this thread — fixes the RuntimeError
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _speak():
            communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
            audio_bytes = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_bytes += chunk["data"]
            with open("/tmp/tts_alert.mp3", "wb") as f:
                f.write(audio_bytes)
            subprocess.run(["afplay", "/tmp/tts_alert.mp3"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        loop.run_until_complete(_speak())
        loop.close()

    threading.Thread(target=_run, daemon=True).start()

# ============================================================
# PROMPTS
# ============================================================

PERSON_PROMPT = (
    "Is there a person visible in this image? "
    "Answer YES or NO first, then briefly describe what you see."
)

FULL_PROMPT = (
    "Look carefully at this image. "
    "1. Is there a person/human? Answer YES or NO. "
    "2. Are there cats? If yes, how many? "
    "3. Are there dogs? If yes, how many? "
    "Be concise. Example: 'Person: YES. Cats: 0. Dogs: 1.'"
)


# ============================================================
# CORE: EXTRACT FRAMES
# ============================================================

def extract_frames(video_path: str, interval_sec: int):
    """
    Read video with OpenCV and sample one frame every interval_sec seconds.
    Returns: (frames list, fps, frame_interval)
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(fps * interval_sec))

    frames = []
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % frame_interval == 0:
            frames.append(frame)
        frame_num += 1
    cap.release()

    return frames, fps, frame_interval


# ============================================================
# CORE: ANALYSE VIDEO FRAMES
# ============================================================

def analyse_video(video_path, interval_sec, model, detect_animals, progress=gr.Progress()):
    """
    Main Gradio function — analyses all frames, yields log progressively.
    Using 'yield' means Gradio streams updates to the UI in real time.
    """
    if video_path is None:
        yield "Please upload a video file first.", ""
        return

    yield "⏳ Extracting frames from video...", ""

    frames, fps, frame_interval = extract_frames(video_path, int(interval_sec))
    total = len(frames)
    duration = (total * frame_interval) / fps

    log = (
        f" Extracted **{total} frames** from a **{duration:.1f}s** video at {fps:.1f} fps\n\n"
        f"Analysing one frame every {interval_sec}s using model `{model}`...\n\n---\n\n"
    )
    yield log, ""

    prompt = FULL_PROMPT if detect_animals else PERSON_PROMPT
    person_present = False
    events = []

    for i, frame in enumerate(frames):
        timestamp = (i * frame_interval) / fps
        ts = seconds_to_timestamp(timestamp)

        progress(i / total, desc=f"Frame {i+1}/{total} at {ts}")

        frame_b64 = frame_to_base64(frame, max_size=384)
        raw = ask_llava(frame_b64, prompt, model=model)
        detection = parse_detection(raw)

        # --- State machine: log only when state CHANGES ---
        if detection["person"] and not person_present:
            person_present = True
            events.append((ts, " PERSON ENTERED"))
            log += f"###  `{ts}` — Person **ENTERED** the scene\n\n"
            if detect_animals and TTS_AVAILABLE:
                speak("Intruder Alert! A person has entered the scene.")

        elif not detection["person"] and person_present:
            person_present = False
            events.append((ts, "🚶 PERSON EXITED"))
            log += f"### 🚶 `{ts}` — Person **EXITED** the scene\n\n"

        if detect_animals:
            if detection["cat"] > 0:
                events.append((ts, f" {detection['cat']} cat(s)"))
                log += f"-  `{ts}` — **{detection['cat']} cat(s)** detected\n\n"
            if detection["dog"] > 0:
                events.append((ts, f"🐶 {detection['dog']} dog(s)"))
                log += f"-  `{ts}` — **{detection['dog']} dog(s)** detected\n\n"

        # Collapsible raw response
        log += (
            f"<details><summary>Frame {i+1} raw LLaVA response</summary>\n\n"
            f"`{raw}`\n\n</details>\n\n"
        )

        yield log, ""

    # Build final summary table
    summary = "##  Detection Summary\n\n"
    if not events:
        summary += "No events detected."
    else:
        summary += "| Time | Event |\n|---|---|\n"
        for ts, event in events:
            summary += f"| `{ts}` | {event} |\n"

    yield log, summary


# ============================================================
# WEBCAM MODE — Optional Task 4
# ============================================================

_webcam_stop = threading.Event()


def start_webcam(sample_interval, model, detect_animals):
    """Streams webcam frames to LLaVA. Yields log lines for Gradio."""
    _webcam_stop.clear()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        yield " Could not open webcam. Check it's not in use by another app."
        return

    prompt = FULL_PROMPT if detect_animals else PERSON_PROMPT
    person_present = False
    log = "📷 **Webcam started.** Click Stop to end.\n\n---\n\n"
    yield log

    while not _webcam_stop.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        ts = time.strftime("%H:%M:%S")
        frame_b64 = frame_to_base64(frame, max_size=384)
        raw = ask_llava(frame_b64, prompt, model=model)
        detection = parse_detection(raw)

        if detection["person"] and not person_present:
            person_present = True
            log += f"###  `{ts}` — Person **ENTERED**\n\n"
            if detect_animals and TTS_AVAILABLE:
                speak("Intruder Alert!")
        elif not detection["person"] and person_present:
            person_present = False
            log += f"### 🚶 `{ts}` — Person **EXITED**\n\n"

        if detect_animals:
            if detection["cat"] > 0:
                log += f"-  `{ts}` — {detection['cat']} cat(s)\n\n"
            if detection["dog"] > 0:
                log += f"-  `{ts}` — {detection['dog']} dog(s)\n\n"

        yield log

        for _ in range(int(sample_interval * 2)):
            if _webcam_stop.is_set():
                break
            time.sleep(0.5)

    cap.release()
    log += "\n\n **Webcam stopped.**"
    yield log


def stop_webcam():
    _webcam_stop.set()
    return " Stop signal sent — finishing current frame..."


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(title="Video Surveillance Agent", theme=gr.themes.Soft()) as demo:

    gr.Markdown("#  Video Surveillance Agent\nPowered by **LLaVA** via Ollama")

    with gr.Tabs():

        # ── Tab 1: Video File ──────────────────────────────────
        with gr.TabItem(" Video File"):
            gr.Markdown(
                "Upload a video clip. The agent samples a frame every N seconds "
                "and detects when people enter or exit the scene."
            )

            video_input = gr.Video(label="Upload Video (MP4, MOV, AVI)")

            with gr.Row():
                interval_slider = gr.Slider(
                    minimum=1, maximum=10, value=2, step=1,
                    label="Seconds between sampled frames  (lower = more accurate but slower)"
                )
                model_input = gr.Textbox(value="llava", label="Ollama Model")

            detect_chk = gr.Checkbox(
                label="Also detect cats & dogs + TTS 'Intruder Alert' (Task 5)",
                value=False
            )

            run_btn = gr.Button("▶ Run Analysis", variant="primary", size="lg")

            log_box     = gr.Markdown(label="Event Log")
            summary_box = gr.Markdown(label="Summary")

            run_btn.click(
                fn=analyse_video,
                inputs=[video_input, interval_slider, model_input, detect_chk],
                outputs=[log_box, summary_box],
            )

        # ── Tab 2: Webcam ─────────────────────────────────────
        with gr.TabItem("📷 Webcam — Real Time (Optional)"):
            gr.Markdown(
                "Captures from your Mac webcam every N seconds and analyses with LLaVA.\n\n"
                " LLaVA takes ~8s per frame on a MacBook — increase the interval accordingly."
            )

            with gr.Row():
                webcam_interval = gr.Slider(
                    minimum=5.0, maximum=30.0, value=10.0, step=1.0,
                    label="Seconds between samples"
                )
                webcam_model = gr.Textbox(value="llava", label="Ollama Model")

            webcam_detect_chk = gr.Checkbox(
                label="Detect cats & dogs + TTS alerts",
                value=False
            )

            with gr.Row():
                webcam_start_btn = gr.Button("▶ Start Webcam", variant="primary")
                webcam_stop_btn  = gr.Button("⏹ Stop", variant="stop")

            webcam_log = gr.Markdown(label="Live Log")

            webcam_start_btn.click(
                fn=start_webcam,
                inputs=[webcam_interval, webcam_model, webcam_detect_chk],
                outputs=webcam_log,
            )
            webcam_stop_btn.click(fn=stop_webcam, outputs=webcam_log)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n Starting Video Surveillance Agent...")
    print(" Ollama must be running:  ollama serve")
    print(" LLaVA must be pulled:    ollama pull llava\n")
    demo.launch()
