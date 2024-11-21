import os
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
from PyPDF2 import PdfReader
import RAG
from compare import compare_responses
import shutil


# 遍歷資料夾並移動根目錄內的 .txt 檔案
def move_txt_files(source_folder, destination_folder):
    if not os.path.exists(source_folder):
        print(f"Source folder '{source_folder}' does not exist.")
        return

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    moved_files = []

    # 只遍歷 source_folder 的根目錄
    for file in os.listdir(source_folder):
        source_file = os.path.join(source_folder, file)
        if os.path.isfile(source_file) and file.endswith(".txt"):
            destination_file = os.path.join(destination_folder, file)
            
            # 確保目標檔案名唯一（避免覆蓋）
            counter = 1
            while os.path.exists(destination_file):
                name, ext = os.path.splitext(file)
                destination_file = os.path.join(destination_folder, f"{name}_{counter}{ext}")
                counter += 1
            
            # 移動檔案
            shutil.move(source_file, destination_file)
            moved_files.append(destination_file)

    if moved_files:
        print(f"Moved {len(moved_files)} .txt files to '{destination_folder}':")
        for file in moved_files:
            print(file)
    else:
        print("No .txt files found in the source folder.")

# Function to read file content based on file type
def read_file_content(uploaded_file):
    if uploaded_file is None:
        return ""
    file_type = uploaded_file.name.split(".")[-1].lower()
    if file_type == "txt":
        return uploaded_file.read().decode("utf-8")
    elif file_type == "csv":
        df = pd.read_csv(uploaded_file)
        return df.to_string(index=False)
    elif file_type == "pdf":
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    else:
        st.write("Unsupported file format.")
        return ""

def rag_process(rag):
    rag.setting()
    move_txt_files(rag.address, f"{rag.address}/input")
    rag.indexing()
    return rag

rag1 = RAG.RAG("test1")
rag2 = RAG.RAG("test2")



# 加載模型和分詞器（請確保模型已經下載到本地）
model_name = "gpt2"  # 替換成您的本地模型名稱
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 設定 GPU 支持
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)


st.title("本地 LLM 應用")

# 資料夾路徑輸入
folder_path_1 = st.text_input("Enter a folder path to scan:", key="1")
folder_path_2 = st.text_input("Enter a folder path to scan:", key="2")

if st.button("Select Documents Folder(relative path)"):
    rag1 = RAG.RAG(folder_path_1)
    rag2 = RAG.RAG(folder_path_2)
    st.write(f"Folder Selected: {folder_path_1}, {folder_path_2}")


if st.button("Generate Graph"):
    print("Start Generating Graph")
    rag_process(rag1)
    rag_process(rag2)
    print("Graph Generation Finished")
    st.write("Graph Generation Finished")

st.write("請在下面的輸入框中輸入您的問題")
# 輸入框：用戶問題
user_input = st.text_input("Your question:")
# 文件上傳
uploaded_file = st.file_uploader("Or upload a file as your question:", type=["txt", "csv", "pdf"])

# 提交按鈕
if st.button("Generate Response"):
    # Check if there's user input or an uploaded file
    combined_input = user_input if user_input else ""
    
    # Add file content if a file was uploaded
    if uploaded_file is not None:
        file_content = read_file_content(uploaded_file)
        combined_input += " " + file_content                                                                                                                                   
    # Only proceed if there's any input to process
    if combined_input.strip():
        st.write(f"Combined input: {combined_input}")
        response1 = rag1.global_search(combined_input)
        response2 = rag2.global_search(combined_input)
        response = compare_responses(combined_input, response1, response2)                
        # Display the response
        st.write("Response1:")
        st.write(response1)
        st.write("\n")
        st.write("Response2:")
        st.write(response2)
        st.write("\n")
        st.write("Comparison:")
        st.write(response)
    else:
        st.write("Please enter text or upload a file before generating a response.")
