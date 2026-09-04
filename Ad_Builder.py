import os
import subprocess

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import math
import tempfile
import gc
import torch

torch.set_num_threads(1)
torch.set_grad_enabled(False)

import pandas as pd
import numpy as np
import gdown
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from moviepy import VideoFileClip, concatenate_videoclips
import imageio_ffmpeg as gp
from proglog import ProgressBarLogger

st.set_page_config(page_title="AI Video Ad Generator", page_icon="🎬", layout="wide")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def compute_embeddings(keywords_list):
    model = load_model()
    return model.encode(keywords_list, show_progress_bar=False, batch_size=32)

class StreamlitProgressLogger(ProgressBarLogger):
    def __init__(self, progress_bar, status_text, start_pct, end_pct):
        super().__init__()
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.start_pct = start_pct
        self.end_pct = end_pct

    def bars_callback(self, bar, attr, value, old_value=None):
        if bar == 't': 
            total = self.bars[bar].get('total', 1)
            if total > 0:
                fraction = value / total
                current_pct = int(self.start_pct + fraction * (self.end_pct - self.start_pct))
                current_pct = min(current_pct, 100)
                
                self.progress_bar.progress(current_pct)
                self.status_text.text(f"Rendering video... {current_pct}%")

st.title("🎬 AI Video Ad Generator")
st.markdown("Upload your clip selector Excel file, input your script, and select the best clips for your ad.")

uploaded_file = st.file_uploader("Upload 'Video Clip Selector.xlsx'", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file, sheet_name="VideoTimeStamps")
        df.columns = df.columns.str.strip()
        
        df_clips = df[['Clip Name', 'Clip Link']].copy()

        df_clips['Clip Name No Extension'] = df_clips['Clip Name'].str.split(".").str[0]
        df_clips['keywords'] = df_clips['Clip Name No Extension'].str.split("_")
        df_clips.rename(columns={
            'Clip Name': 'clip_id', 
            'Clip Link': 'gdrive_url'
        }, inplace=True)

        df_clips['unique_id'] = df_clips.index.astype(str)
        
        df_clips['Country'] = df['Country'] if 'Country' in df.columns else "Any"
        df_clips['Subject'] = df['Subject'] if 'Subject' in df.columns else "Any"
        
        df_clips['keywords'] = df_clips['keywords'].astype(str)
        
        with st.spinner("Calculating clip embeddings..."):
            clip_embeddings = compute_embeddings(df_clips["keywords"].tolist())
            
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        st.stop()

    st.write("---")
    
    st.header("1. Apply Filters & Enter Script")
    
    countries = ["Any"] + sorted(df_clips['Country'].dropna().astype(str).unique().tolist())
    subjects = ["Any"] + sorted(df_clips['Subject'].dropna().astype(str).unique().tolist())
    
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        selected_country = st.selectbox("Select Country", countries)
    with filter_col2:
        selected_subject = st.selectbox("Select Subject", subjects)

    user_input = st.text_area("Enter your ad script (one scene per line):", height=150)
    
    if st.button("**Find Clips**", type="primary", use_container_width=True):
        ad_script = [line.strip() for line in user_input.split("\n") if line.strip()]
        
        if not ad_script:
            st.warning("Please enter a script.")
        else:
            mask = pd.Series(True, index=df_clips.index)
            if selected_country != "Any":
                mask &= (df_clips['Country'].astype(str).str.strip().str.lower() == selected_country.strip().lower())
            if selected_subject != "Any":
                mask &= (df_clips['Subject'].astype(str).str.strip().str.lower() == selected_subject.strip().lower())
                
            filtered_df = df_clips[mask].copy()
            
            if filtered_df.empty:
                st.error("No clips match this exact Country and Subject combination.")
            else:
                subset_embeddings = clip_embeddings[mask.to_numpy()]
                model = load_model()
                
                scenes_data = []
                for line_number, script_line in enumerate(ad_script, start=1):
                    script_embedding = model.encode([script_line])
                    similarities = cosine_similarity(script_embedding, subset_embeddings)[0]
                    
                    top_subset_indices = similarities.argsort()[::-1]
                    
                    choices = []
                    for idx in top_subset_indices:
                        choices.append({
                            "unique_id": str(filtered_df.iloc[idx]["unique_id"]),
                            "clip_id": str(filtered_df.iloc[idx]["clip_id"]),
                            "keywords": str(filtered_df.iloc[idx]["keywords"]),
                            "gdrive_url": str(filtered_df.iloc[idx]["gdrive_url"]),
                            "score": f"{similarities[idx]:.1%}"
                        })
                        
                    scenes_data.append({
                        "line_number": line_number,
                        "script_line": script_line,
                        "choices": choices
                    })
                
                st.session_state["scenes_data"] = scenes_data
                st.session_state["ad_script"] = ad_script
                
                for i, scene in enumerate(scenes_data):
                    st.session_state[f"page_{i}"] = 0
                    st.session_state[f"scene_{scene['line_number']}_selected"] = scene["choices"][0]["unique_id"]

    if "scenes_data" in st.session_state:
        st.write("---")
        st.header("2. Select Clips for Your Script")
        
        for i, scene in enumerate(st.session_state["scenes_data"]):
            st.subheader(f"Scene {scene['line_number']}: {scene['script_line']}")
            
            total_choices = len(scene["choices"])
            total_pages = math.ceil(total_choices / 5)
            current_page = st.session_state.get(f"page_{i}", 0)
            
            start_idx = current_page * 5
            end_idx = min(start_idx + 5, total_choices)
            current_batch = scene["choices"][start_idx:end_idx]
            
            cols = st.columns(5)
            for col_idx, col in enumerate(cols):
                if col_idx < len(current_batch):
                    c = current_batch[col_idx]
                    with col:
                        scene_key = f"scene_{scene['line_number']}_selected"
                        is_selected = st.session_state.get(scene_key) == c['unique_id']
                        
                        btn_label = "✅ Selected" if is_selected else "Select"
                        btn_type = "primary" if is_selected else "secondary"
                        
                        if st.button(btn_label, key=f"sel_{i}_{start_idx+col_idx}", type=btn_type, use_container_width=True):
                            st.session_state[scene_key] = c['unique_id']
                            st.rerun()

                        st.markdown(f"**Score: {c['score']}**")
                        st.markdown(f"🔗[{c['clip_id']}]({c['gdrive_url']})")
                        st.caption(f"{c['keywords']}")
            
            nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
            with nav_col1:
                if st.button("⬅️ Previous 5", key=f"prev_{i}", use_container_width=True):
                    st.session_state[f"page_{i}"] = (current_page - 1) % total_pages
                    st.rerun()
            with nav_col2:
                st.markdown(f"<div style='text-align: center; font-size: 1.1rem; padding-top: 10px;'>Page {current_page + 1} of {total_pages}</div>", unsafe_allow_html=True)
            with nav_col3:
                if st.button("Next 5 ➡️", key=f"next_{i}", use_container_width=True):
                    st.session_state[f"page_{i}"] = (current_page + 1) % total_pages
                    st.rerun()
            
            st.write("---")

        st.write("")
        if st.button("**GENERATE AD**", type="primary", use_container_width=True):
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            temp_dir = tempfile.mkdtemp()
            downloaded_clip_paths = []
            video_clips = []
            final_video = None
            error_occurred = False
            total_scenes = len(st.session_state["ad_script"])
            
            try:
                for line_number in range(1, total_scenes + 1):
                    status_text.text(f"Downloading clip {line_number} of {total_scenes}...")
                    
                    selected_unique_id = st.session_state[f"scene_{line_number}_selected"]
                    match_row = df_clips[df_clips['unique_id'] == selected_unique_id]
                    
                    if match_row.empty:
                        st.error(f"Error: Clip metadata missing for selection in Scene {line_number}")
                        error_occurred = True
                        break

                    clip_name = match_row.iloc[0]['clip_id']
                    gdrive_url = match_row.iloc[0]['gdrive_url']

                    base, ext = os.path.splitext(clip_name)
                    safe_clip_name = f"{base}{ext.lower()}"

                    download_path = os.path.join(temp_dir, f"scene{line_number}_{safe_clip_name}")
                    final_mp4_path = os.path.join(temp_dir, f"scene{line_number}_{base}.mp4")

                    if not os.path.exists(final_mp4_path):
                        if not os.path.exists(download_path):
                            try:
                                gdown.download(gdrive_url, download_path, quiet=True)
                            except Exception as e:
                                st.error(f"Failed to download clip '{clip_name}'. Details: {e}")
                                error_occurred = True
                                break

                        if download_path.endswith('.mov'):
                            try:
                                ffmpeg_binary = gp.get_ffmpeg_exe()

                                cmd = [
                                    ffmpeg_binary, '-y', '-i', download_path,
                                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                                    '-c:a', 'aac', '-map_metadata', '-1',
                                    final_mp4_path
                                ]
                                
                                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                                
                                if os.path.exists(download_path):
                                    os.remove(download_path)
                                    
                            except Exception as e:
                                st.error(f"System FFmpeg failed to process '{clip_name}'. Details: {e}")
                                error_occurred = True
                                break
                        else:
                            os.rename(download_path, final_mp4_path)

                    downloaded_clip_paths.append(final_mp4_path)
                    
                    current_dl_progress = int((line_number / total_scenes) * 50)
                    progress_bar.progress(current_dl_progress)
                
                if not error_occurred:
                    status_text.text("Concatenating video clips...")
                    progress_bar.progress(55)
                    
                    for path in downloaded_clip_paths:
                        clip = VideoFileClip(path)
                        target_h = 1080 if clip.h > 1080 else clip.h
                        
                        target_w = int(round(clip.w * (target_h / clip.h)))
                        
                        if target_w % 2 != 0:
                            target_w -= 1
                        if target_h % 2 != 0:
                            target_h -= 1
                        
                        if target_w != clip.w or target_h != clip.h:
                            clip = clip.resized((target_w, target_h)) 
                            
                        video_clips.append(clip)
                        
                    final_video = concatenate_videoclips(video_clips, method="compose")
                    
                    final_output_path = os.path.join(temp_dir, "final_ad_output.mp4")
                    temp_audio_path = os.path.join(temp_dir, "temp_audio_build.m4a")

                    my_logger = StreamlitProgressLogger(
                        progress_bar=progress_bar, 
                        status_text=status_text, 
                        start_pct=55, 
                        end_pct=100
                    )

                    final_video.write_videofile(
                        final_output_path, 
                        fps=24, 
                        codec="libx264", 
                        audio_codec="aac",
                        preset="fast",
                        bitrate="4000k",
                        threads=1,
                        ffmpeg_params=["-pix_fmt", "yuv420p"],
                        temp_audiofile=temp_audio_path,
                        remove_temp=True,
                        logger=my_logger
                    )

                    progress_bar.progress(100)
                    st.success("🎉🎉 Video generated successfully! 🎉🎉")
                    
                    st.video(final_output_path)
                    
                    with open(final_output_path, "rb") as video_file:
                        st.download_button(
                            label="**DOWNLOAD FINAL AD**",
                            data=video_file,
                            file_name="final_ad.mp4",
                            mime="video/mp4",
                            type="primary",
                            use_container_width=True
                        )

            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"MoviePy compilation failed: {e}")

            finally:
                if final_video is not None:
                    try:
                        final_video.close()
                    except Exception:
                        pass
                
                for clip in video_clips:
                    try:
                        clip.close()
                    except Exception:
                        pass
                
                gc.collect()