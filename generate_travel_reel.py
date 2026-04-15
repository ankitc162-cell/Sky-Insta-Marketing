"""
Travel Destination Reel Generator
- Accepts destination name + optional audience type as CLI args
- Generates fast-paced Hindi voiceover with many segments via Gemini TTS (Erinome)
- Fetches multiple destination-specific Pexels video clips per segment (concatenated)
- Verifies clip relevance: Pexels metadata first, Gemini Vision fallback
- Falls back to destination-specific Pexels images with Ken Burns zoom
- Builds 9:16 reel with destination overlay + logo
- Mixes viral background music at 15% volume
- Uses Whisper for audio‑aligned segment durations (perfect sync)
- Sends final video to Telegram
- Falls back to ChatGPT if Gemini text generation fails
"""

import os
import sys
import re
import time
import requests
import random
import json
import shutil
import subprocess
import openai
from google import genai
from google.genai import types
from pathlib import Path
from datetime import datetime
from moviepy.editor import (
    VideoFileClip, ColorClip, CompositeVideoClip,
    AudioFileClip, concatenate_videoclips, ImageClip,
    CompositeAudioClip
)
from moviepy.audio.AudioClip import concatenate_audioclips
from moviepy.video.fx.all import crop, resize
from moviepy.audio.fx.all import volumex

import PIL.Image
if not hasattr(PIL.Image, "ANTIALIAS"):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

# --- Constants ---
GEMINI_API_KEY  = os.environ["GEMINI_API_KEY"]
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
PEXELS_API_KEY  = os.environ["PEXELS_API_KEY"]
REEL_W, REEL_H  = 1080, 1920

CACHE_DIR       = Path(os.environ.get("PEXELS_CACHE_DIR", "/tmp/pexels_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BG_MUSIC_PATH   = os.environ.get("BG_MUSIC_PATH", "bg_music.mp3")
BG_MUSIC_VOLUME = 0.15
VOICEOVER_SPEED = 1.20          # increased for ~60s reels
LOGO_PATH       = os.environ.get("LOGO_PATH", "logo.png")

VIRAL_MUSIC_OPTIONS = [
    "https://cdn.pixabay.com/download/audio/2024/03/13/audio_3d5b6e6e21.mp3",
    "https://cdn.pixabay.com/download/audio/2023/06/05/audio_0bbfe02b94.mp3",
    "https://cdn.pixabay.com/download/audio/2022/10/25/audio_946f989511.mp3",
]

if len(sys.argv) > 1:
    DESTINATION   = sys.argv[1]
    AUDIENCE_TYPE = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("AUDIENCE_TYPE", "general travelers")
else:
    DESTINATION   = os.environ.get("DESTINATION", "")
    AUDIENCE_TYPE = os.environ.get("AUDIENCE_TYPE", "general travelers")
    if not DESTINATION:
        print("ERROR: Provide destination as argument or DESTINATION env var.")
        sys.exit(1)

# Cache for verification results (video_id -> bool)
VERIFY_CACHE = {}

# --- Telegram Sender ---
def send_video_telegram(video_path: str, caption: str = ""):
    token   = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("[ERROR] Missing Telegram credentials")
        return False
    url = f"https://api.telegram.org/bot{token}/sendVideo"
    try:
        with open(video_path, "rb") as f:
            response = requests.post(
                url,
                files={"video": f},
                data={"chat_id": chat_id, "caption": caption},
                timeout=120
            )
        response.raise_for_status()
        print("[SUCCESS] Video sent to Telegram")
        return True
    except Exception as e:
        print(f"[ERROR] Telegram send failed: {e}")
        return False

# --- Script Generation (adjusted for ~60s) ---
def _build_prompt(destination: str, audience_type: str) -> str:
    return (
        "Aap ek premium travel company 'Sky Suffer Tourism Private Limited' ke liye viral Instagram Reels scriptwriter hain.\n"
        f"Destination: \"{destination}\"\n"
        f"Target audience: {audience_type}\n\n"

        "Sirf valid JSON return karo — koi preamble nahi, koi markdown fence nahi — is exact format mein:\n"
        "{\n"
        '  "hook": "First 3-second scroll-stopping line",\n'
        '  "viral_line": "One deeply emotional/quotable line",\n'
        '  "music_vibe": "cinematic / upbeat / romantic",\n'
        '  "script": "Complete voiceover script",\n'
        '  "segments": [\n'
        '    {"description": "What is shown", "keywords": "specific landmark name"}\n'
        '  ]\n'
        "}\n\n"

        "=== SCRIPT RULES ===\n"
        f"HOOK (first 3 sec — curiosity/FOMO):\n"
        f"  - Example: 'Kya aapne kabhi aisi jagah dekhi hai jahan...?'\n"
        f"  - THEN: 'Sky Suffer Tourism ke saath aaj hum aapko le ja rahe hain {destination}!'\n\n"

        f"CONTENT: {destination} ke 8-12 sabse iconic spots/activities cover karo.\n"
        "  - Har spot ke liye 1-2 sentences — fast, exciting.\n"
        "  - Adjectives: 'shaandar', 'adbhut', 'hairan kar dene wala', 'jaaduī'.\n\n"

        "CTA (urgent):\n"
        "  - 'Offers limited — abhi contact karein Sky Suffer Tourism: 9654100207'\n\n"

        "  - Total script 170-190 words. Fast-paced Hindi.\n\n"

        "=== KEYWORDS RULES ===\n"
        "  - Har segment ke liye EXACT landmark name in English (e.g., 'Bali Tegallalang Rice Terrace').\n"
        "  - 8-12 segments total.\n"
        "  - Keywords MUST be specific to {destination}.\n"
    )

def _parse_response(text: str) -> dict:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1).strip()
    return json.loads(text)

def generate_script(destination: str, audience_type: str) -> dict:
    prompt = _build_prompt(destination, audience_type)
    gemini_models = ["gemini-2.0-flash", "gemini-2.5-flash"]

    for model in gemini_models:
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            print(f"[INFO] Trying Gemini model: {model}")
            response = client.models.generate_content(model=model, contents=prompt)
            data = _parse_response(response.text)
            print(f"[INFO] Script generated: {len(data['script'].split())} words, {len(data['segments'])} segments")
            return data
        except Exception as e:
            print(f"[WARN] Gemini {model} failed: {e}")

    if not OPENAI_API_KEY:
        raise RuntimeError("All Gemini models failed and OPENAI_API_KEY not set.")
    try:
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Output only valid JSON, no markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1200
        )
        data = _parse_response(response.choices[0].message.content)
        print(f"[INFO] Script generated via ChatGPT: {len(data['script'].split())} words, {len(data['segments'])} segments")
        return data
    except Exception as e:
        print(f"[ERROR] ChatGPT failed: {e}")
        raise RuntimeError("Both Gemini and ChatGPT failed.")

# --- Background Music ---
def get_bg_music_path() -> str:
    if os.path.exists(BG_MUSIC_PATH):
        return BG_MUSIC_PATH
    for url in VIRAL_MUSIC_OPTIONS:
        try:
            r = requests.get(url, timeout=30, stream=True)
            r.raise_for_status()
            tmp_path = "/tmp/bg_music_dl.mp3"
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            return tmp_path
        except Exception as e:
            print(f"[WARN] Music download failed: {e}")
    return None

def load_bg_music(duration: float):
    music_path = get_bg_music_path()
    if not music_path:
        return None
    try:
        music = AudioFileClip(music_path)
        if music.duration < duration:
            loops = int(duration / music.duration) + 1
            music = concatenate_audioclips([music] * loops)
        music = music.subclip(0, duration)
        music = volumex(music, BG_MUSIC_VOLUME)
        return music
    except Exception as e:
        print(f"[WARN] Could not load BG music: {e}")
        return None

# --- Destination Query Helper ---
def build_query(base: str, dest: str) -> str:
    if dest.lower() in base.lower():
        return base
    return f"{dest} {base}".strip()

# --- Video Verification (metadata first, Gemini fallback) ---
def verify_video_metadata(video_data: dict, destination: str, expected_landmark: str) -> bool:
    """Check if Pexels metadata contains destination or landmark keywords."""
    try:
        text_fields = []
        if "url" in video_data:
            text_fields.append(video_data["url"])
        if "user" in video_data and "name" in video_data["user"]:
            text_fields.append(video_data["user"]["name"])
        if "video_files" in video_data:
            for vf in video_data["video_files"]:
                if "link" in vf:
                    text_fields.append(vf["link"])

        combined = " ".join(text_fields).lower()
        dest_lower = destination.lower()
        landmark_words = expected_landmark.lower().split()
        # Check for destination or first significant word of landmark (length > 3)
        if dest_lower in combined or any(word in combined for word in landmark_words if len(word) > 3):
            print(f"[VERIFY-META] Metadata contains destination/landmark → ACCEPT")
            return True
        else:
            print(f"[VERIFY-META] Metadata lacks relevant keywords → fallback to Gemini")
            return False
    except Exception as e:
        print(f"[WARN] Metadata verification error: {e}, falling back to Gemini")
        return False

def verify_video_content_gemini(video_path: str, expected_landmark: str, destination: str) -> bool:
    """Use Gemini Vision as secondary verification."""
    try:
        from PIL import Image
        clip = VideoFileClip(video_path)
        frame = clip.get_frame(0)
        clip.close()
        img = Image.fromarray(frame)

        client = genai.Client(api_key=GEMINI_API_KEY)
        prompt = f"Does this image show {expected_landmark} in {destination}? Answer only 'yes' or 'no'."
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt, img]
        )
        answer = response.text.strip().lower()
        print(f"[VERIFY-GEMINI] '{expected_landmark}': {answer}")
        return answer == 'yes'
    except Exception as e:
        print(f"[WARN] Gemini Vision verification failed: {e}, accepting by default")
        return True  # Accept if Gemini fails

def is_video_relevant(video_path: str, video_data: dict, expected_landmark: str, destination: str) -> bool:
    """Check relevance: metadata first, then Gemini fallback."""
    video_id = video_data["id"]
    if video_id in VERIFY_CACHE:
        print(f"[VERIFY-CACHE] Using cached result for {video_id}: {VERIFY_CACHE[video_id]}")
        return VERIFY_CACHE[video_id]

    # 1. Try metadata first
    if verify_video_metadata(video_data, destination, expected_landmark):
        VERIFY_CACHE[video_id] = True
        return True

    # 2. Fallback to Gemini Vision
    gemini_result = verify_video_content_gemini(video_path, expected_landmark, destination)
    VERIFY_CACHE[video_id] = gemini_result
    return gemini_result

# --- Pexels Video Fetch (with verification) ---
used_video_ids = set()

def fetch_multiple_pexels_videos(query: str, destination: str, count: int = 3, max_retries: int = 2) -> list:
    headers = {"Authorization": PEXELS_API_KEY}
    videos = []
    for orientation in ["portrait", "landscape"]:
        params = {"query": query, "per_page": 20, "orientation": orientation, "size": "medium"}
        for attempt in range(max_retries + 1):
            try:
                resp = requests.get("https://api.pexels.com/videos/search", headers=headers, params=params, timeout=15)
                resp.raise_for_status()
                videos = resp.json().get("videos", [])
                break
            except requests.HTTPError as e:
                if resp.status_code == 429:
                    time.sleep(5 * (attempt + 1))
                else:
                    break
            except Exception:
                break
        if videos:
            break

    if not videos:
        print(f"[WARN] No videos for '{query}'")
        return []

    available = [v for v in videos if v["id"] not in used_video_ids]
    if len(available) < count:
        available = videos
    random.shuffle(available)

    downloaded = []
    for video in available:
        if len(downloaded) >= count:
            break
        used_video_ids.add(video["id"])
        video_files = sorted(video["video_files"], key=lambda x: x.get("width", 0), reverse=True)
        chosen = next((v for v in video_files if v.get("width", 0) <= 1920), video_files[0])
        cache_file = CACHE_DIR / f"{video['id']}_{chosen.get('width', 0)}.mp4"
        output_path = f"/tmp/clip_{video['id']}.mp4"

        if cache_file.exists():
            shutil.copy2(str(cache_file), output_path)
            print(f"[CACHE HIT] {query} -> {cache_file.name}")
        else:
            print(f"[INFO] Downloading: {query} -> {chosen['link'][:60]}... (ID: {video['id']})")
            try:
                r = requests.get(chosen["link"], timeout=60, stream=True)
                r.raise_for_status()
                with open(output_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                shutil.copy2(output_path, str(cache_file))
            except Exception as e:
                print(f"[WARN] Download failed: {e}")
                continue

        # Verify relevance
        if is_video_relevant(output_path, video, query, destination):
            downloaded.append(output_path)
        else:
            print(f"[REJECT] Video {video['id']} not relevant to {query}")
            os.remove(output_path)

    return downloaded

# --- Pexels Image Fetch (fallback) ---
used_image_ids = set()

def fetch_pexels_image(query: str, destination: str, output_path: str) -> str:
    headers = {"Authorization": PEXELS_API_KEY}
    for orientation in ["portrait", "landscape"]:
        params = {"query": query, "per_page": 10, "orientation": orientation, "size": "large"}
        try:
            resp = requests.get("https://api.pexels.com/v1/search", headers=headers, params=params, timeout=15)
            resp.raise_for_status()
            photos = resp.json().get("photos", [])
            if photos:
                break
        except Exception:
            continue
    if not photos:
        return None
    available = [p for p in photos if p["id"] not in used_image_ids]
    if not available:
        available = photos
    photo = random.choice(available[:5])
    used_image_ids.add(photo["id"])
    image_url = photo["src"]["original"]
    try:
        r = requests.get(image_url, timeout=30, stream=True)
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return output_path
    except Exception as e:
        print(f"[WARN] Image download failed: {e}")
        return None

# --- Voiceover Generation ---
def generate_voiceover(script: str, output_path: str = "voiceover.mp3") -> str:
    import imageio_ffmpeg
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-tts",
        contents=script,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Erinome")
                )
            )
        )
    )
    audio_data = b""
    for part in response.candidates[0].content.parts:
        if hasattr(part, "inline_data") and part.inline_data:
            audio_data += part.inline_data.data

    raw_path  = output_path.replace(".mp3", ".raw")
    wav_path  = output_path.replace(".mp3", ".wav")
    sped_path = output_path.replace(".mp3", "_sped.wav")
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()

    with open(raw_path, "wb") as f:
        f.write(audio_data)

    subprocess.run([ffmpeg_bin, "-y", "-f", "s16le", "-ar", "24000", "-ac", "1",
                    "-i", raw_path, wav_path], check=True, capture_output=True)
    subprocess.run([ffmpeg_bin, "-y", "-i", wav_path,
                    "-filter:a", f"atempo={VOICEOVER_SPEED}", sped_path],
                   check=True, capture_output=True)
    subprocess.run([ffmpeg_bin, "-y", "-i", sped_path, output_path],
                   check=True, capture_output=True)

    for p in [raw_path, wav_path, sped_path]:
        if os.path.exists(p): os.remove(p)
    print(f"[INFO] Voiceover saved -> {output_path} ({VOICEOVER_SPEED}x speed)")
    return output_path

# --- Segment Duration Estimation ---
def estimate_segment_durations(data: dict, total_duration: float) -> list:
    segments = data["segments"]
    script = data["script"]
    n = len(segments)
    try:
        words = script.split()
        total_words = len(words)
        chunk_size = total_words // n
        counts = []
        for i in range(n):
            start = i * chunk_size
            end = start + chunk_size if i < n - 1 else total_words
            counts.append(max(end - start, 1))
        total = sum(counts)
        durations = [total_duration * (c / total) for c in counts]
        print(f"[INFO] Word-weighted durations: {[f'{d:.1f}s' for d in durations]}")
        return durations
    except Exception as e:
        print(f"[WARN] Word-weight failed ({e}), using equal split")
        return [total_duration / n] * n

def get_segment_durations_from_whisper(audio_path: str, data: dict, total_duration: float) -> list:
    try:
        import whisper
        print("[INFO] Running Whisper for audio-aligned timestamps...")
        model = whisper.load_model("tiny")
        result = model.transcribe(audio_path, language="hi", word_timestamps=True)

        words = []
        for seg in result["segments"]:
            for w in seg.get("words", []):
                words.append((w["start"], w["word"].strip()))

        if not words:
            raise ValueError("No word timestamps returned by Whisper")

        n = len(data["segments"])
        chunk_size = len(words) // n
        cut_times = [0.0]
        for i in range(1, n):
            idx = i * chunk_size
            if idx < len(words):
                cut_times.append(words[idx][0])
        cut_times.append(total_duration)

        durations = [max(cut_times[i+1] - cut_times[i], 1.0) for i in range(n)]
        print(f"[INFO] Whisper-aligned durations: {[f'{d:.1f}s' for d in durations]}")
        return durations

    except Exception as e:
        print(f"[WARN] Whisper failed ({e}), falling back to word-weight")
        return estimate_segment_durations(data, total_duration)

# --- Video Assembly ---
def make_destination_overlay(destination: str, duration: float):
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    font_paths = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
                  "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf",
                  "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    font = None
    for fp in font_paths:
        if os.path.exists(fp):
            font = ImageFont.truetype(fp, 75)
            break
    if font is None:
        font = ImageFont.load_default()
    spaced = " ".join(destination.upper())
    dummy = Image.new("RGBA", (1,1))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0,0), spaced, font=font)
    text_w, text_h = bbox[2]-bbox[0], bbox[3]-bbox[1]
    pad = 30
    img = Image.new("RGBA", (text_w+2*pad, text_h+2*pad), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    for dx in range(-2,3):
        for dy in range(-2,3):
            draw.text((pad+dx, pad+dy), spaced, font=font, fill=(200,200,200,179))
    draw.text((pad, pad), spaced, font=font, fill=(255,255,255,179))
    y_pos = int(REEL_H * 0.10)
    x_pos = (REEL_W - img.width)//2
    return [ImageClip(np.array(img)).set_duration(duration).set_position((x_pos, y_pos))]

def make_logo_overlay(duration: float):
    if not os.path.exists(LOGO_PATH):
        return []
    from PIL import Image
    import numpy as np
    logo = Image.open(LOGO_PATH).convert("RGBA")
    max_h = 100
    ratio = max_h / logo.height
    new_w = int(logo.width * ratio)
    logo = logo.resize((new_w, max_h), PIL.Image.LANCZOS)
    logo_clip = ImageClip(np.array(logo))
    x_pos = (REEL_W - new_w)//2
    y_pos = REEL_H - max_h - 40
    return [logo_clip.set_duration(duration).set_position((x_pos, y_pos))]

def crop_to_portrait(clip):
    target_ratio = REEL_W / REEL_H
    if clip.w / clip.h > target_ratio:
        new_w = int(clip.h * target_ratio)
        clip = crop(clip, width=new_w, x_center=clip.w/2)
    else:
        new_h = int(clip.w / target_ratio)
        clip = crop(clip, height=new_h, y_center=clip.h/2)
    return resize(clip, (REEL_W, REEL_H))

def apply_ken_burns(clip, duration, zoom=1.1):
    def make_frame(t):
        scale = 1 + (zoom-1) * t / duration
        frame = clip.get_frame(t)
        from PIL import Image
        import numpy as np
        img = Image.fromarray(frame)
        new_w, new_h = int(clip.w*scale), int(clip.h*scale)
        img = img.resize((new_w, new_h), PIL.Image.LANCZOS)
        left, top = (new_w - clip.w)//2, (new_h - clip.h)//2
        img = img.crop((left, top, left+clip.w, top+clip.h))
        return np.array(img)
    return VideoFileClip(None).set_duration(duration).fl(make_frame)

def build_video(data: dict, audio_path: str, destination: str, output_path: str = "reel.mp4") -> str:
    voiceover = AudioFileClip(audio_path)
    duration = voiceover.duration
    segments = data["segments"]
    n = len(segments)

    seg_durations = get_segment_durations_from_whisper(audio_path, data, duration)

    print(f"[INFO] {n} segments, durations: {[f'{d:.1f}s' for d in seg_durations]}")

    clips = []
    temp_files = []

    for i, seg in enumerate(segments):
        seg_duration = seg_durations[i]

        base_keywords = seg["keywords"]
        query = build_query(base_keywords, destination)
        print(f"[INFO] Segment {i}: '{query}' ({seg_duration:.1f}s)")

        video_paths = fetch_multiple_pexels_videos(query, destination, count=3)
        segment_clips = []
        for vp in video_paths:
            try:
                v = VideoFileClip(vp)
                v.get_frame(0)
                segment_clips.append(v)
                temp_files.append(vp)
            except Exception as e:
                print(f"[WARN] Video unplayable: {e}")

        if segment_clips:
            if len(segment_clips) > 1:
                visual = concatenate_videoclips(segment_clips)
            else:
                visual = segment_clips[0]
            if visual.duration < seg_duration:
                loops = int(seg_duration / visual.duration) + 1
                visual = concatenate_videoclips([visual] * loops)
            visual = visual.subclip(0, seg_duration)
        else:
            print(f"[INFO] No videos, trying image for '{query}'")
            img_path = f"img_{i}.jpg"
            img_local = fetch_pexels_image(query, destination, img_path)
            if img_local:
                temp_files.append(img_local)
                try:
                    img = ImageClip(img_local)
                    img = crop_to_portrait(resize(img, (REEL_W, REEL_H)))
                    visual = apply_ken_burns(img, seg_duration, 1.1)
                except Exception as e:
                    print(f"[WARN] Image processing failed: {e}")
                    visual = ColorClip(size=(REEL_W, REEL_H), color=(15,15,30)).set_duration(seg_duration)
            else:
                visual = ColorClip(size=(REEL_W, REEL_H), color=(15,15,30)).set_duration(seg_duration)

        visual = crop_to_portrait(visual).without_audio()
        overlay = ColorClip(size=(REEL_W, REEL_H), color=(0,0,0)).set_opacity(0.45).set_duration(seg_duration)
        clips.append(CompositeVideoClip([visual, overlay], size=(REEL_W, REEL_H)))

    base = concatenate_videoclips(clips, method="compose")
    overlays = make_destination_overlay(destination, duration) + make_logo_overlay(duration)
    bg = load_bg_music(duration)
    final_audio = CompositeAudioClip([voiceover, bg]) if bg else voiceover

    final = CompositeVideoClip([base] + overlays, size=(REEL_W, REEL_H)).set_audio(final_audio).set_duration(duration)
    final.write_videofile(output_path, fps=30, codec="libx264", audio_codec="aac",
                          temp_audiofile="temp_audio.m4a", remove_temp=True, logger=None,
                          ffmpeg_params=["-profile:v", "baseline", "-level", "3.0",
                                         "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                                         "-b:v", "3500k", "-b:a", "128k", "-ar", "44100", "-ac", "2"])

    voiceover.close()
    if bg: bg.close()
    final.close()
    base.close()
    for c in clips:
        try: c.close()
        except: pass
    for f in temp_files:
        if os.path.exists(f): os.remove(f)
    print(f"[INFO] Video built -> {output_path}")
    return output_path

def cleanup_temp_files(*paths):
    for p in paths:
        if p and os.path.exists(p):
            try: os.remove(p)
            except: pass

def main():
    import imageio_ffmpeg
    os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()

    print("=" * 60)
    print(f"Travel Reel Generator — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Destination:   {DESTINATION}")
    print(f"Audience:      {AUDIENCE_TYPE}")
    print("=" * 60)

    audio_path = "voiceover.mp3"
    video_path = "reel.mp4"

    try:
        data = generate_script(DESTINATION, AUDIENCE_TYPE)
        generate_voiceover(data["script"], audio_path)
        build_video(data, audio_path, DESTINATION, video_path)

        caption = (
            f"✨ {DESTINATION} awaits you with Skysafar Tourism! ✨\n\n"
            f"{data.get('viral_line', '')}\n\n"
            f"Discover the magic of {DESTINATION} — from breathtaking landscapes to rich culture.\n\n"
            f"📞 Contact us: 9654100207\n"
            f"📌 Save this reel and start planning your next adventure!\n\n"
            f"#SkySafarTourism #{DESTINATION.replace(' ', '')} #Travel #Wanderlust"
        )
        send_video_telegram(video_path, caption)
        print("\n✅ Done! Reel sent to Telegram.")
    finally:
        cleanup_temp_files(audio_path, video_path)

if __name__ == "__main__":
    main()
