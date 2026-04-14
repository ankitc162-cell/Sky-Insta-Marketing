"""
Travel Destination Reel Generator
- Accepts destination name as command-line argument
- Generates Hindi voiceover covering top attractions using Gemini TTS (Erinome)
- Fetches unique Pexels video clips for each segment
- Builds 9:16 reel with destination overlay
- Sends final video to Telegram chat
- Falls back to ChatGPT if Gemini text generation fails
"""

import os
import sys
import time
import requests
import random
import json
import openai
from google import genai
from pathlib import Path
from datetime import datetime
from moviepy.editor import (
    VideoFileClip, ColorClip, CompositeVideoClip,
    AudioFileClip, concatenate_videoclips, ImageClip
)
from moviepy.video.fx.all import crop, resize

# --- Constants ---
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
PEXELS_API_KEY = os.environ["PEXELS_API_KEY"]
REEL_W, REEL_H = 1080, 1920

# Read destination from command line
if len(sys.argv) > 1:
    DESTINATION = sys.argv[1]
else:
    DESTINATION = os.environ.get("DESTINATION", "")
    if not DESTINATION:
        print("ERROR: Provide destination as argument or DESTINATION env var.")
        sys.exit(1)

# --- Telegram Sender ---
def send_video_telegram(video_path: str, caption: str = ""):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
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

# --- Script Generation ---
def _build_prompt(destination: str) -> str:
    return (
        "Aap ek premium travel company 'Sky Safar Tourism Private Limited' ke liye Instagram Reels scriptwriter hain.\n"
        f"Aaj hum destination \"{destination}\" par ek exciting reel banayenge.\n\n"
        f"Ek Hindi voiceover script likho jo {destination} ke top tourist spots, unki khoobsurti, aur wahan jaane ke reasons ko highlight kare.\n"
        "Script exciting aur inviting tone mein honi chahiye, jaise ek tour guide apne guests ko roam karwa raha ho.\n\n"
        "IMPORTANT: Script ke saath Pexels video search keywords bhi provide karo — har location ya vibe ke liye alag keyword.\n"
        "Sirf valid JSON return karo is exact format mein:\n"
        '{{"script": "Poora voiceover script yahan...", "segments": [{{"description": "Is part mein kya dikhaya ja raha hai", "keywords": "pexels search term in English"}}, ...]}}\n\n'
        "Script rules:\n"
        f"- Shuru karo: 'Sky Safar Tourism ke saath aaj hum aapko le ja rahe hain {destination} ki yaadgaar yatra par!'\n"
        f"- {destination} ke 4-6 sabse famous aur beautiful tourist spots ko cover karo.\n"
        "- Har spot ke baare mein 2-3 sentences bolo — uska khaas attraction kya hai, kyun visit karna chahiye.\n"
        "- Exciting adjectives use karo: 'shaandar', 'manmohak', 'adbhut', 'swarg se kam nahi'.\n"
        "- Ant mein kaho: 'Toh der kis baat ki? Abhi contact karein Sky Safar Tourism Private Limited ko 9654100207 par aur book karein apna dream trip {destination} ka!'\n"
        "- Total script 200-250 words. Natural, flowing Hindi.\n\n"
        "Keywords rules:\n"
        "- Har tourist spot ya visual vibe ke liye ek segment banao.\n"
        "- Keywords ENGLISH mein, simple 1-3 word Pexels search terms."
    )

def _parse_response(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)

def generate_script(destination: str) -> dict:
    prompt = _build_prompt(destination)
    gemini_models = ["gemini-2.0-flash", "gemini-2.5-flash"]

    # Try Gemini models in sequence
    for model in gemini_models:
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            print(f"[INFO] Trying Gemini model: {model}")
            response = client.models.generate_content(model=model, contents=prompt)
            data = _parse_response(response.text)
            print(f"[INFO] Script generated using Gemini {model} ({len(data['script'].split())} words, {len(data['segments'])} segments)")
            return data
        except Exception as e:
            print(f"[WARN] Gemini {model} failed: {e}")

    # Fallback to ChatGPT
    if not OPENAI_API_KEY:
        raise RuntimeError("All Gemini models failed and OPENAI_API_KEY not set.")
    try:
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs only valid JSON in the exact format requested."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        response_text = response.choices[0].message.content
        data = _parse_response(response_text)
        print(f"[INFO] Script generated using ChatGPT ({len(data['script'].split())} words, {len(data['segments'])} segments)")
        return data
    except Exception as e:
        print(f"[ERROR] ChatGPT also failed: {e}")
        raise RuntimeError("Both Gemini and ChatGPT failed to generate script.")

# --- Pexels Video Fetch (with deduplication) ---
used_video_ids = set()

def fetch_pexels_video(query: str, output_path: str):
    headers = {"Authorization": PEXELS_API_KEY}
    for orientation in ["portrait", "landscape"]:
        params = {"query": query, "per_page": 15, "orientation": orientation, "size": "medium"}
        resp = requests.get("https://api.pexels.com/videos/search", headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        videos = resp.json().get("videos", [])
        if videos:
            break
    if not videos:
        print(f"[WARN] No videos for '{query}'")
        return None

    available = [v for v in videos if v["id"] not in used_video_ids]
    if not available:
        available = videos

    video = random.choice(available[:5])
    used_video_ids.add(video["id"])

    video_files = sorted(video["video_files"], key=lambda x: x.get("width", 0), reverse=True)
    chosen = next((v for v in video_files if v.get("width", 0) <= 1920), video_files[0])
    print(f"[INFO] Downloading: {query} -> {chosen['link'][:60]}... (ID: {video['id']})")
    r = requests.get(chosen["link"], timeout=60, stream=True)
    r.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    return output_path

# --- Voiceover Generation (Gemini TTS - Erinome) ---
def generate_voiceover(script: str, output_path: str = "voiceover.mp3") -> str:
    import struct
    import subprocess
    import imageio_ffmpeg

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-tts",
        contents=script,
        config={
            "response_modalities": ["AUDIO"],
            "speech_config": {
                "voice_config": {
                    "prebuilt_voice_config": {
                        "voice_name": "Erinome"
                    }
                }
            }
        }
    )

    audio_data = b""
    for part in response.candidates[0].content.parts:
        if hasattr(part, "inline_data") and part.inline_data:
            audio_data += part.inline_data.data

    raw_path = output_path.replace(".mp3", ".raw")
    wav_path = output_path.replace(".mp3", ".wav")

    with open(raw_path, "wb") as f:
        f.write(audio_data)

    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([
        ffmpeg_bin, "-y",
        "-f", "s16le", "-ar", "24000", "-ac", "1",
        "-i", raw_path,
        wav_path
    ], check=True, capture_output=True)
    subprocess.run([
        ffmpeg_bin, "-y", "-i", wav_path, output_path
    ], check=True, capture_output=True)

    os.remove(raw_path)
    os.remove(wav_path)

    print(f"[INFO] Gemini TTS voiceover (Erinome) saved -> {output_path}")
    return output_path

# --- Video Assembly ---
def make_text_image(text, fontsize, color, stroke_color="black", stroke_width=3):
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fontsize)
    except:
        font = ImageFont.load_default()
    dummy = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0] + stroke_width*2 + 20
    h = bbox[3] - bbox[1] + stroke_width*2 + 20
    img = Image.new("RGBA", (w, h), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    for dx in range(-stroke_width, stroke_width+1):
        for dy in range(-stroke_width, stroke_width+1):
            draw.text((stroke_width+10+dx, stroke_width+10+dy), text, font=font, fill=stroke_color)
    draw.text((stroke_width+10, stroke_width+10), text, font=font, fill=color)
    return np.array(img)

def make_destination_overlay(destination: str, duration: float):
    dest_text = make_text_image(destination, 80, "#FFC800")
    dest_clip = ImageClip(dest_text).set_duration(duration).set_position(("center", 0.25), relative=True)
    return [dest_clip]

def crop_to_portrait(clip):
    target_ratio = REEL_W / REEL_H
    clip_ratio = clip.w / clip.h
    if clip_ratio > target_ratio:
        new_w = int(clip.h * target_ratio)
        clip = crop(clip, width=new_w, x_center=clip.w/2)
    else:
        new_h = int(clip.w / target_ratio)
        clip = crop(clip, height=new_h, y_center=clip.h/2)
    return resize(clip, (REEL_W, REEL_H))

def build_video(data: dict, audio_path: str, destination: str, output_path: str = "reel.mp4") -> str:
    audio = AudioFileClip(audio_path)
    duration = audio.duration
    segments = data["segments"]
    n = len(segments)
    seg_duration = duration / n
    print(f"[INFO] Video duration: {duration:.1f}s, {n} segments x {seg_duration:.1f}s each")

    clips = []
    for i, seg in enumerate(segments):
        clip_path = f"clip_{i}.mp4"
        pexels_path = fetch_pexels_video(seg["keywords"], clip_path)
        if pexels_path:
            try:
                raw = VideoFileClip(pexels_path)
                if raw.duration < seg_duration:
                    loops = int(seg_duration / raw.duration) + 1
                    raw = concatenate_videoclips([raw] * loops)
                raw = raw.subclip(0, seg_duration)
                raw = crop_to_portrait(raw).without_audio()
            except Exception as e:
                print(f"[WARN] Clip {i} failed: {e}, using color fallback")
                raw = ColorClip(size=(REEL_W, REEL_H), color=(15,15,30)).set_duration(seg_duration)
        else:
            raw = ColorClip(size=(REEL_W, REEL_H), color=(15,15,30)).set_duration(seg_duration)
        overlay = ColorClip(size=(REEL_W, REEL_H), color=(0,0,0)).set_opacity(0.45).set_duration(seg_duration)
        clips.append(CompositeVideoClip([raw, overlay], size=(REEL_W, REEL_H)))

    base_video = concatenate_videoclips(clips, method="compose")
    overlays = make_destination_overlay(destination, duration)
    final = CompositeVideoClip([base_video] + overlays, size=(REEL_W, REEL_H)).set_audio(audio).set_duration(duration)
    final.write_videofile(
        output_path, fps=30, codec="libx264", audio_codec="aac",
        temp_audiofile="temp_audio.m4a", remove_temp=True, logger=None,
        ffmpeg_params=[
            "-profile:v", "baseline", "-level", "3.0",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-b:v", "3500k", "-b:a", "128k", "-ar", "44100", "-ac", "2"
        ]
    )
    print(f"[INFO] Video built -> {output_path}")
    return output_path

# --- Main ---
def main():
    import imageio_ffmpeg
    os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()

    print("=" * 60)
    print(f"Travel Reel Generator — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Destination: {DESTINATION}")
    print("=" * 60)

    data = generate_script(DESTINATION)
    audio_path = generate_voiceover(data["script"])
    video_path = build_video(data, audio_path, DESTINATION)

    caption = (
        f"✨ {DESTINATION} awaits you with Sky Safar Tourism! ✨\n\n"
        f"Discover the magic of {DESTINATION} — from breathtaking landscapes to rich culture.\n\n"
        f"📞 Contact us: 9654100207\n"
        f"📌 Save this reel and start planning your next adventure!\n\n"
        f"#SkySafarTourism #{DESTINATION.replace(' ', '')} #Travel #Wanderlust"
    )

    send_video_telegram(video_path, caption)
    print("\n✅ Done! Reel sent to Telegram.")

if __name__ == "__main__":
    main()
