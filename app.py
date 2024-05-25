import pandas as pd
import streamlit as st
import base64
import openai

# Set the OpenAI API key using st.secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

client = openai.Client()

st.set_page_config(
    page_title="Image to Text",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("AI Image Description Generator 🤖✍️")

def to_base64(uploaded_file):
    file_buffer = uploaded_file.read()
    b64 = base64.b64encode(file_buffer).decode()
    return f"data:image/png;base64,{b64}"

with st.sidebar:
    st.title("Upload Your Images")
    st.session_state.images = st.file_uploader(label=" ", accept_multiple_files=True)

def generate_df():
    current_df = pd.DataFrame(
        {
            "image_id": [img.file_id for img in st.session_state.images],
            "image": [to_base64(img) for img in st.session_state.images],
            "name": [img.name for img in st.session_state.images],
            "description": [""] * len(st.session_state.images),
        }
    )

    if "df" not in st.session_state:
        st.session_state.df = current_df
        return

    new_df = pd.merge(current_df, st.session_state.df, on=["image_id"], how="outer", indicator=True)
    new_df = new_df[new_df["_merge"] != "right_only"].drop(columns=["_merge", "name_y", "image_y", "description_x"])
    new_df = new_df.rename(columns={"name_x": "name", "image_x": "image", "description_y": "description"})
    new_df["description"] = new_df["description"].fillna("")

    st.session_state.df = new_df

def render_df():
    columns = st.columns(3)
    with columns[0]:
        st.data_editor(
            st.session_state.df,
            disabled=False,
            num_rows="dynamic",
            column_config={
                "image": st.column_config.ImageColumn(
                    "Preview Image", help="Image preview", width=100
                ),
                "name": st.column_config.TextColumn(
                    "Name", help="Image name", width=200
                ),
                "description": st.column_config.TextColumn(
                    "Description", help="Image description", width=800
                ),
            },
            hide_index=True,
            key="data_editor",
        )
    with columns[1]:
        st.markdown("---")
        st.subheader("Selected Rows")
        selected_rows = st.session_state.df[st.session_state.data_editor.selected_rows]
        st.write(selected_rows)
    with columns[2]:
        st.markdown("---")
        st.subheader("Edited Rows")
        edited_rows = st.session_state.df[st.session_state.data_editor.edited_rows]
        st.write(edited_rows)

def generate_description(image_base64):
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": st.session_state.text_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_base64,
                        },
                    },
                ],
            }
        ],
        max_tokens=50,
    )
    return response.choices[0].message.content
