# Topic 6: Vision-Language Models (VLM)

## Table of Contents

1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Setup & Installation](#setup--installation)
4. [Exercise 1: Vision-Language Chat Agent](#exercise-1-vision-language-chat-agent)
5. [Exercise 2: Video Surveillance Agent](#exercise-2-video-surveillance-agent)
6. [Results](#results)
7. [Challenges & Lessons Learned](#challenges--lessons-learned)

## Project Overview
This project explores **Vision-Language Models (VLMs)** - AI models that can understand both image and text simultaneously. Two agents were built using **LLaVA** (Large Language and Vision Assistant) running locally via **Ollama**, without any cloud API or cost.


**Tools & Libraries Used:**
- [Ollama](https://ollama.ai) — runs LLMs/VLMs locally
- [LLaVA](https://llava-vl.github.io) — open-source vision-language model
- [LangGraph](https://langchain-ai.github.io/langgraph/) — agent framework using state graphs
- [OpenCV (cv2)](https://opencv.org) — video frame extraction
- [Gradio](https://gradio.app) — web-based UI for local Python programs (Exercise 2)
- [ipywidgets](https://ipywidgets.readthedocs.io) — notebook-based UI (Exercise 1)

## Directory Structure


```
Topic6VLM/
|- task1.ipynb      # Exercise 1 — multi-turn image chat agent
|- cat.jpg          # example of uploaded image for task1.ipynb
|- task2.py             # Exercise 2 — video person detection agent, (video file too big to upload in github)
└── README.md                      # This file
```
## SetUp & Installation

### Requirements
- Python 3.10+
- [Ollama](https://ollama.ai/download) installed

###  Install Python dependencies
```bash
pip install opencv-python ollama gradio langgraph langchain-core ipywidgets Pillow
```

### Pull the LLaVA model
```bash
ollama pull llava
```

### Start Ollama (if not already running)
```bash
ollama serve
```
> If you see `address already in use`, Ollama is already running — this is fine.

## Exercise 1: Vision-Language Chat Agent

Purpose: A multi-turn conversational agent that can answer questions about an uploaded image, remembering the full conversation history across turns.

### How to Run 
```bash
# Run inside a Jupyter notebook or Google Colab
%run task1.py
```

### Architecture 

The agent is built as a **Langraph state graph**:
```
[user input] -> chat_node: calls LLaVA with image + history -> output
                        |--------------loop-----------------|

```

| Component | Role |
|---|---|
| `AgentState` | Holds conversation history (`messages`) and the image (`image_b64`) |
| `add_messages` | Appends new messages to history instead of overwriting - enables memory |
| `chat_node` | Formats history for Ollama, calls LLaVA, returns AI response |
| Image encoding | Uploaded image is resized and base64-encoded before being sent |

## Exercise 2: Video Surveillance Agent


An agent that analyses a video by extracting frames every N seconds and asking LLaVA whether a person is present, logging timestamps when people enter and exit the scene.

### How to Run

```bash

python3 task2.py
# Then open http://127.0.0.1:7860 in your browser
```

### Architecture 

```
Video File (is not uploaded in this github folder due to large size) -> OpenCV extracts 1 frame every N seconds -> Each frame (resized -> base64 encoded) -> LLaVA asked: "Is there a person in this image?" ->
parse_detection(): keyword search on free-text response ->
State Machine: absent -> present = ENTER event
              present -> absent = EXIT event   -> Events logged with timestamps



```

## Results 


<img width="970" height="731" alt="chat_histroy" src="https://github.com/user-attachments/assets/cbd3dffb-5c28-46eb-863f-22a292832a18" />



