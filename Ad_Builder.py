import os
import pandas as pd
import gdown
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from moviepy import VideoFileClip, concatenate_videoclips

st.set_page_config(page_title="Automated Ad Video Builder", layout="wide")
CLIP_PATH = "clip_library.xlsx"
TEMP_DIR = "./temp_clips"
FINAL_OUTPUT_PATH = "final_ad_output.mp4"

os.makedirs(TEMP_DIR, exist_ok=True)

@st.cache_data
def load_data():
    df = pd.read_excel(CLIP_PATH, sheet_name=1)
    df.columns = df.columns.str.strip()
    df_new = df[['Clip Name', 'Actions', 'Clip Link']].copy()
    df_new.rename(columns={'Clip Name': 'clip_id', 'Actions': 'keywords', 'Clip Link': 'gdrive_url'}, inplace=True)
    return df_new

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

df_new = load_data()
model = load_model()

if 'clip_embeddings' not in st.session_state:
    st.session_state.clip_embeddings = model.encode(df_new["keywords"].tolist(), show_progress_bar=False)

# APP
st.title("🎬 Automated Ad Video Builder")

default_script = ""

user_input = st.text_area(
    label="Paste your script (Put each scene/line on a new row, Hit Ctrl+Enter to submit):",
    value=default_script,
    height=150
)

ad_script = [line.strip() for line in user_input.split("\n") if line.strip()]

if 'selections' not in st.session_state:
    st.session_state.selections = {}

if ad_script:
    for line_number, script_line in enumerate(ad_script, start=1):
        st.write("---")
        st.subheader(f"Scene {line_number}: *\"{script_line}\"*")
        
        script_embedding = model.encode([script_line])
        similarities = cosine_similarity(script_embedding, st.session_state.clip_embeddings)[0]
        
        top_3_indices = similarities.argsort()[-5:][::-1]
        cols = st.columns(5)        
        choices_options = []
        
        for i, idx in enumerate(top_3_indices):
            clip_id = df_new.iloc[idx]["clip_id"]
            keywords = df_new.iloc[idx]["keywords"]
            gdrive_url = df_new.iloc[idx]["gdrive_url"]
            score = similarities[idx]
            
            choices_options.append({
                "clip_id": clip_id,
                "gdrive_url": gdrive_url,
                "label": f"Option {i+1}: {clip_id} (Match: {score:.1%})"
            })
            
            with cols[i]:
                st.metric(label=f"Match Confidence", value=f"{score:.1%}")
                st.caption(f"**Tags:** {keywords}")
                st.markdown(f"[🔗 View Video Source Link]({gdrive_url})")

        current_selection = st.radio(
            label=f"Choose clip for Scene {line_number}:",
            options=[0, 1, 2],
            format_func=lambda x: choices_options[x]["label"],
            key=f"scene_{line_number}_{hash(script_line)}_radio"
        )
        
        st.session_state.selections[line_number] = choices_options[current_selection]

    st.write("---")
    if st.button("Render Final Sequence", type="primary", use_container_width=True):
        downloaded_clip_paths = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        active_selections = {k: v for k, v in st.session_state.selections.items() if k <= len(ad_script)}
        
        for index, (line_num, asset_data) in enumerate(active_selections.items()):
            status_text.text(f"Processing Scene {line_num}: Fetching clip '{asset_data['clip_id']}'...")
            local_filename = os.path.join(TEMP_DIR, f"{asset_data['clip_id']}.mp4")
            
            if not os.path.exists(local_filename):
                try:
                    gdown.download(asset_data['gdrive_url'], local_filename, quiet=True)
                except Exception as e:
                    st.error(f"Failed download on '{asset_data['clip_id']}': Please check its share permissions.")
                    st.stop()
            
            downloaded_clip_paths.append(local_filename)
            progress_bar.progress(int((index + 1) / len(active_selections) * 50))
            
        status_text.text("Merging video files together...")
        try:
            video_clips = [VideoFileClip(path) for path in downloaded_clip_paths]
            final_video = concatenate_videoclips(video_clips, method="compose")
            final_video.write_videofile(
                FINAL_OUTPUT_PATH, 
                fps=24, 
                codec="libx264", 
                audio_codec="aac"
            )
            
            for clip in video_clips:
                clip.close()
                
            progress_bar.progress(100)
            status_text.success("Ad compiled successfully!")
            
            with open(FINAL_OUTPUT_PATH, 'rb') as video_file:
                st.video(video_file.read())
                
        except Exception as e:
            st.error(f"MoviePy failed compilation: {e}")
else:
    st.warning("Please input a script into the text area above to begin.")