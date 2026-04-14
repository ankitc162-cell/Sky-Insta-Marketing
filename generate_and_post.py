"""
Daily Instagram Reel Generator & Poster
- Fetches follower count using instagrapi
- Generates number facts using Google Gemini (free)
- Creates voiceover using Microsoft Edge TTS (free)
- Fetches video clips from Pexels (free API)
- Assembles engaging reel: video clips + number overlay + voiceover
- Posts reel to Instagram using instagrapi
"""

import os
import time
import asyncio
import textwrap
import requests
import random
import json
import edge_tts
from google import genai
from pathlib import Path
from datetime import datetime
from moviepy.editor import (
    VideoFileClip, ColorClip, TextClip, CompositeVideoClip,
    AudioFileClip, concatenate_videoclips, ImageClip
)
from moviepy.video.fx.all import crop, resize

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GROQ_API_KEY    = os.environ.get("GROQ_API_KEY", "")
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
PEXELS_API_KEY = os.environ["PEXELS_API_KEY"]
EDGE_TTS_VOICE = "hi-IN-MadhurNeural"
DAY_NUMBER     = int(os.environ.get("DAY_NUMBER", 1))
REEL_W, REEL_H = 1080, 1920


# ─────────────────────────────────────────────────────────────────────────────
# 1. Get follower count via instagrapi
# ─────────────────────────────────────────────────────────────────────────────
BUFFER_API_KEY    = os.environ["BUFFER_API_KEY"]
BUFFER_CHANNEL_ID = os.environ["BUFFER_CHANNEL_ID"]
BUFFER_API_URL    = "https://api.buffer.com"

def buffer_graphql(query: str, variables: dict = None) -> dict:
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    resp = requests.post(
        BUFFER_API_URL,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {BUFFER_API_KEY}"
        },
        json=payload,
        timeout=30
    )
    if not resp.ok:
        raise RuntimeError(f"Buffer API {resp.status_code}: {resp.text}")
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(f"Buffer API error: {data['errors']}")
    return data


def get_follower_count() -> int:
    # Fetch follower count via Instagram Graph API
    # Falls back to 0 if not configured
    ig_user_id = os.environ.get("INSTAGRAM_USER_ID", "")
    ig_token   = os.environ.get("INSTAGRAM_ACCESS_TOKEN", "")
    if ig_user_id and ig_token:
        try:
            url  = f"https://graph.facebook.com/v19.0/{ig_user_id}?fields=followers_count&access_token={ig_token}"
            resp = requests.get(url, timeout=15)
            if resp.ok:
                count = resp.json().get("followers_count", 0) or 0
                print(f"[INFO] Follower count: {count}")
                return count
        except Exception as e:
            print(f"[WARN] Could not get follower count: {e}")
    print("[INFO] Follower count: 0 (no Instagram credentials configured)")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# 2. Generate script + keywords
# ─────────────────────────────────────────────────────────────────────────────
def _build_prompt(day, followers):
    return (
        "Aap ek educational Instagram Reels page ke liye scriptwriter hain.\n"
        f"Aaj Day {day} hai. Page par {followers} followers hain.\n\n"
        f"Number {followers} ke baare mein ek Hindi voiceover script likho AUR har fact ke liye Pexels video search keywords do.\n\n"
        "Sirf valid JSON return karo is exact format mein:\n"
        '{{"script": "Poora voiceover script yahan...", "segments": [{{"fact": "Fact 1 ka ek sentence summary", "keywords": "pexels search term in English"}}, {{"fact": "Fact 2 ka ek sentence summary", "keywords": "pexels search term in English"}}, {{"fact": "Fact 3 ka ek sentence summary", "keywords": "pexels search term in English"}}]}}\n\n'
        "Script rules:\n"
        f"- BILKUL EXACTLY is tarah shuru karo: Day {day} of posting facts about the number of my followers. Aaj mere {followers} followers hain.\n"
        f"- Phir naturally kaho: aur aaj hum baat karenge number {followers} ke baare mein, jo sach mein bahut interesting hai.\n"
        f"- EXACTLY 3 informational facts likho number {followers} ke baare mein. Sirf 3, na zyada na kam.\n"
        "- Har fact clearly aur naturally explain karo — jaise ek knowledgeable dost baat kar raha ho.\n"
        "- Natural transitions use karo jaise: iske alawa, ek aur interesting baat, aur sabse khas baat yeh hai.\n"
        "- Phrases jaise 'yeh baat jaan ke aapko hairaani hogi' ya 'kya aap jaante hain' use karo — lekin natural taur pe, forced nahi.\n"
        "- Jahan fit ho, India, cricket, science, history ka reference do.\n"
        "- BILKUL EXACTLY is tarah khatam karo: Aisi aur rochak jaankari ke liye follow karo. Aapka ek follow kal ka number badal dega. Chalo mere saath 1 million tak pahunche.\n"
        "- 150-180 words total. Shuddh, natural, informational Hindi — na overdramatic na robotic.\n\n"
        "Keywords rules:\n"
        "- EXACTLY 3 segments, ek har fact ke liye\n"
        "- Keywords ENGLISH mein, simple 1-3 word Pexels search terms"
    )


def _parse_response(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    data = json.loads(text)
    print(f"[INFO] Script generated ({len(data['script'].split())} words)")
    print(f"[INFO] Segments: {[s['keywords'] for s in data['segments']]}")
    return data


def _generate_with_groq(day, followers):
    prompt = _build_prompt(day, followers)
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
            "temperature": 0.7
        },
        timeout=30
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"]
    return _parse_response(text)


def _generate_with_gemini(day, followers):
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = _build_prompt(day, followers)
    for attempt in range(3):
        try:
            response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            return _parse_response(response.text)
        except Exception as e:
            if attempt == 2:
                raise
            wait = 30 * (attempt + 1)
            print(f"[WARN] Gemini attempt {attempt+1} failed: retrying in {wait}s...")
            time.sleep(wait)


def _generate_with_chatgpt(day, followers):
    prompt = _build_prompt(day, followers)
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
            "temperature": 0.8
        },
        timeout=30
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"]
    return _parse_response(text)


def generate_script_and_keywords(day: int, followers: int) -> dict:
    try:
        print("[INFO] Using Gemini for script generation...")
        return _generate_with_gemini(day, followers)
    except Exception as e:
        print(f"[WARN] Gemini failed: {e}")
        if OPENAI_API_KEY:
            print("[INFO] Falling back to ChatGPT...")
            return _generate_with_chatgpt(day, followers)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# 3. Fetch video clips from Pexels
# ─────────────────────────────────────────────────────────────────────────────
def fetch_pexels_video(query: str, output_path: str):
    headers = {"Authorization": PEXELS_API_KEY}
    for orientation in ["portrait", "landscape"]:
        params = {"query": query, "per_page": 10, "orientation": orientation, "size": "medium"}
        resp   = requests.get("https://api.pexels.com/videos/search", headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        videos = resp.json().get("videos", [])
        if videos:
            break
    if not videos:
        print(f"[WARN] No videos found for '{query}'")
        return None
    video      = random.choice(videos[:5])
    video_files = sorted(video["video_files"], key=lambda x: x.get("width", 0), reverse=True)
    chosen     = next((v for v in video_files if v.get("width", 0) <= 1920), video_files[0])
    print(f"[INFO] Downloading Pexels clip: {query} -> {chosen['link'][:60]}...")
    r = requests.get(chosen["link"], timeout=60, stream=True)
    r.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"[INFO] Clip saved -> {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# 4. Generate voiceover with Gemini TTS (Enceladus voice)
# ─────────────────────────────────────────────────────────────────────────────
def generate_voiceover(script: str, output_path: str = "voiceover.mp3") -> str:
    return _gemini_tts(script, output_path)


def _gemini_tts(script: str, output_path: str) -> str:
    import struct
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-tts",
        contents=script,
        config={
            "response_modalities": ["AUDIO"],
            "speech_config": {
                "voice_config": {
                    "prebuilt_voice_config": {
                        "voice_name": "Enceladus"
                    }
                }
            }
        }
    )
    # Extract audio data from response
    audio_data = b""
    for part in response.candidates[0].content.parts:
        if hasattr(part, "inline_data") and part.inline_data:
            audio_data += part.inline_data.data

    # Convert raw PCM to WAV then to MP3 using ffmpeg
    import subprocess
    raw_path = output_path.replace(".mp3", ".raw")
    wav_path = output_path.replace(".mp3", ".wav")

    with open(raw_path, "wb") as f:
        f.write(audio_data)

    # Use imageio-ffmpeg binary (guaranteed to exist)
    import imageio_ffmpeg
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()

    # Convert raw PCM (24kHz, 16-bit, mono) to WAV
    subprocess.run([
        ffmpeg_bin, "-y",
        "-f", "s16le", "-ar", "24000", "-ac", "1",
        "-i", raw_path,
        wav_path
    ], check=True, capture_output=True)

    # Convert WAV to MP3
    subprocess.run([
        ffmpeg_bin, "-y", "-i", wav_path, output_path
    ], check=True, capture_output=True)

    import os as _os
    _os.remove(raw_path)
    _os.remove(wav_path)

    print(f"[INFO] Gemini TTS voiceover saved -> {output_path}")
    return output_path


async def _edge_tts_async(script: str, output_path: str):
    communicate = edge_tts.Communicate(script, EDGE_TTS_VOICE)
    await communicate.save(output_path)

def _edge_tts_fallback(script: str, output_path: str) -> str:
    asyncio.run(_edge_tts_async(script, output_path))
    print(f"[INFO] Edge TTS voiceover saved -> {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# 5. Build the reel video
# ─────────────────────────────────────────────────────────────────────────────
def make_text_image(text, fontsize, color, stroke_color="black", stroke_width=3):
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fontsize)
    except Exception:
        font = ImageFont.load_default()
    dummy = Image.new("RGBA", (1, 1))
    draw  = ImageDraw.Draw(dummy)
    bbox  = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0] + stroke_width * 2 + 20
    h = bbox[3] - bbox[1] + stroke_width * 2 + 20
    img  = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    for dx in range(-stroke_width, stroke_width + 1):
        for dy in range(-stroke_width, stroke_width + 1):
            draw.text((stroke_width + 10 + dx, stroke_width + 10 + dy), text, font=font, fill=stroke_color)
    draw.text((stroke_width + 10, stroke_width + 10), text, font=font, fill=color)
    return np.array(img)


def make_number_overlay(number: int, day: int, duration: float):
    day_arr      = make_text_image("Day " + str(day), 60, "#FFC800")
    followers_arr = make_text_image("Followers " + str(number), 60, "white")
    day_clip      = ImageClip(day_arr).set_duration(duration).set_position(("center", 0.22), relative=True)
    followers_clip = ImageClip(followers_arr).set_duration(duration).set_position(("center", 0.29), relative=True)
    return [day_clip, followers_clip]


def crop_to_portrait(clip):
    target_ratio = REEL_W / REEL_H
    clip_ratio   = clip.w / clip.h
    if clip_ratio > target_ratio:
        new_w = int(clip.h * target_ratio)
        clip  = crop(clip, width=new_w, x_center=clip.w / 2)
    else:
        new_h = int(clip.w / target_ratio)
        clip  = crop(clip, height=new_h, y_center=clip.h / 2)
    return resize(clip, (REEL_W, REEL_H))


def build_video(data: dict, audio_path: str, day: int, followers: int, output_path: str = "reel.mp4") -> str:
    audio        = AudioFileClip(audio_path)
    duration     = audio.duration
    segments     = data["segments"]
    n            = len(segments)
    seg_duration = duration / n
    print(f"[INFO] Total duration: {duration:.1f}s, {n} segments x {seg_duration:.1f}s each")

    clips = []
    for i, seg in enumerate(segments):
        clip_path   = f"clip_{i}.mp4"
        pexels_path = fetch_pexels_video(seg["keywords"], clip_path)
        if pexels_path:
            try:
                raw = VideoFileClip(pexels_path)
                if raw.duration < seg_duration:
                    loops = int(seg_duration / raw.duration) + 1
                    raw   = concatenate_videoclips([raw] * loops)
                raw = raw.subclip(0, seg_duration)
                raw = crop_to_portrait(raw).without_audio()
            except Exception as e:
                print(f"[WARN] Failed to process clip {i}: {e}, using color fallback")
                raw = ColorClip(size=(REEL_W, REEL_H), color=(15, 15, 30)).set_duration(seg_duration)
        else:
            raw = ColorClip(size=(REEL_W, REEL_H), color=(15, 15, 30)).set_duration(seg_duration)
        overlay = ColorClip(size=(REEL_W, REEL_H), color=(0, 0, 0)).set_opacity(0.45).set_duration(seg_duration)
        clips.append(CompositeVideoClip([raw, overlay], size=(REEL_W, REEL_H)))

    base_video = concatenate_videoclips(clips, method="compose")
    overlays   = make_number_overlay(followers, day, duration)
    final      = CompositeVideoClip([base_video] + overlays, size=(REEL_W, REEL_H)).set_audio(audio).set_duration(duration)
    final.write_videofile(
        output_path, fps=30, codec="libx264", audio_codec="aac",
        temp_audiofile="temp_audio.m4a", remove_temp=True, logger=None,
        ffmpeg_params=[
            "-profile:v", "baseline", "-level", "3.0",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-b:v", "3500k", "-b:a", "128k", "-ar", "44100", "-ac", "2"
        ]
    )
    print(f"[INFO] Final video built -> {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# 6. Post reel using instagrapi
# ─────────────────────────────────────────────────────────────────────────────
def upload_to_github_releases(video_path: str) -> str:
    """Upload video to GitHub Releases for temporary public URL."""
    gh_token  = os.environ["GH_TOKEN"]
    gh_repo   = os.environ["GITHUB_REPOSITORY"]
    tag       = f"reel-day-{DAY_NUMBER}-{int(time.time())}"
    headers   = {
        "Authorization": f"Bearer {gh_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    # Create release
    rel = requests.post(
        f"https://api.github.com/repos/{gh_repo}/releases",
        headers=headers,
        json={"tag_name": tag, "name": tag, "draft": False, "prerelease": True},
        timeout=30
    )
    rel.raise_for_status()
    upload_url = rel.json()["upload_url"].replace("{?name,label}", "")
    filename   = Path(video_path).name
    # Upload asset
    with open(video_path, "rb") as f:
        asset = requests.post(
            f"{upload_url}?name={filename}",
            headers={**headers, "Content-Type": "video/mp4"},
            data=f,
            timeout=300
        )
    asset.raise_for_status()
    # Use direct asset download URL (no redirects) for Buffer compatibility
    asset_data = asset.json()
    url = asset_data["browser_download_url"]
    # Convert to direct CDN URL that Buffer can access without redirect
    direct_url = f"https://objects.githubusercontent.com/github-production-release-asset-2e65be/{asset_data['id']}?X-Amz-Algorithm=AWS4-HMAC-SHA256"
    print(f"[INFO] Video uploaded to GitHub Releases -> {url}")
    return url


def post_reel_buffer(video_path: str, caption: str):
    """Post reel via Buffer GraphQL API — zero ban risk, official Meta partner."""
    video_url = upload_to_github_releases(video_path)

    mutation = """
    mutation CreatePost($input: CreatePostInput!) {
      createPost(input: $input) {
        ... on PostActionSuccess {
          post { id dueAt }
        }
        ... on MutationError {
          message
        }
      }
    }
    """
    variables = {
        "input": {
            "text": caption,
            "channelId": BUFFER_CHANNEL_ID,
            "schedulingType": "automatic",
            "mode": "addToQueue",
            "metadata": {
                "instagram": {
                    "type": "reel",
                    "shouldShareToFeed": True
                }
            },
            "assets": {
                "videos": [{"url": video_url}]
            }
        }
    }
    data   = buffer_graphql(mutation, variables)
    result = data["data"]["createPost"]
    if "post" in result:
        print(f"[INFO] Reel scheduled via Buffer! Post ID: {result['post']['id']}")
    else:
        raise RuntimeError(f"Buffer post failed: {result.get('message')}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 7. Orchestrate
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import imageio_ffmpeg
    os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()

    print("=" * 60)
    print("  Daily Reel Generator -- " + datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))
    print("=" * 60)

    followers  = get_follower_count()
    data       = generate_script_and_keywords(DAY_NUMBER, followers)
    audio_path = generate_voiceover(data["script"])
    video_path = build_video(data, audio_path, DAY_NUMBER, followers)

    caption = (
        "Day " + str(DAY_NUMBER) + " of posting facts about the number " + str(followers) + "\n\n"
        + "Today we have " + str(followers) + " followers and THIS number is fascinating!\n\n"
        + "Follow us and be part of this journey to 1,000,000\n\n"
        + "#numberfacts #didyouknow #educational #facts #math "
        + "#instagramreels #viral #mindblown #science"
    )

    post_reel_buffer(video_path, caption)
    print("\nDone! Reel posted successfully.")


if __name__ == "__main__":
    main()
