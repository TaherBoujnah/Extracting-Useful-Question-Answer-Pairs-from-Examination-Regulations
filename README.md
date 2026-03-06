Extracting Useful Question–Answer Pairs from Examination Regulations

Bachelor Thesis Project – NLP & LLM Applications

This project builds a pipeline for automatically extracting useful Question–Answer (QA) pairs from university examination regulations and deploying them through a retrieval-based chatbot system.

The goal is to transform long and complex regulation documents into clear, searchable FAQ-style information for students.

Overview

University examination regulations contain important information about exams, grading rules, retakes, and academic policies. However, these documents are often lengthy and difficult to navigate.

This project builds a complete NLP pipeline that:

processes regulation documents

extracts structured regulation segments

generates candidate QA pairs using local LLMs

filters and evaluates generated outputs

builds an embedding-based retrieval system

serves answers through a chatbot interface

The repository includes both:

the research pipeline used in the thesis

a working FAQ chatbot prototype

Pipeline

The system follows a multi-stage pipeline:

Document preprocessing

Extract regulation text

Normalize formatting

Convert documents into structured markdown

Regulation chunking

Split regulations into meaningful sections

Preserve metadata such as section name, page numbers, and degree level

Question–Answer generation

Generate candidate QA pairs from regulation chunks using local LLMs

Filtering and selection

Remove duplicates and low-quality QA pairs

Select the most relevant ones

Evaluation

Compare generated QA pairs with a manually curated gold dataset

Retrieval system

Create embeddings for generated FAQ questions

Retrieve answers using semantic similarity

Chatbot interface

Provide an interactive FAQ assistant using Chainlit

Repository Structure
.
├── backend/
│   ├── api.py
│   ├── chunk_examregs.py
│   ├── config.py
│   ├── indexer.py
│   ├── evaluation/
│   ├── qa/
│   ├── retrieval/
│   └── unused_codes/
│
├── frontend/
│   └── chatbot_chainlit.py
│
├── data/
│   ├── eval/
│   ├── final/
│   ├── generated/
│   ├── gold/
│   ├── plots/
│   ├── ALL_Informatik_Exam_Regulations.md
│   ├── chunks.jsonl
│   ├── embeddings.npy
│   ├── embeddings_meta.json
│   └── exam_regulations.pdf
│
├── .chainlit/
├── chainlit.md
└── README.md
Models Explored

The project evaluates several local LLMs for QA generation:

Gemma 3 (4B)

Llama 3.1 (8B)

Qwen 2.5 (7B)

Generated outputs and evaluation results for these models are included in the repository.

Data Artifacts

The repository contains several datasets produced during the thesis.

Generated QA datasets
Located in data/generated/

These contain model outputs and generation metadata.

Gold dataset
A manually curated reference dataset used for evaluation.

data/gold/gold.json

Final selected QA dataset

data/final/

This contains the final selected FAQ pairs used by the chatbot.

Visualizations

The project includes visual analysis of generated QA clusters.

Cluster Overview


LDA Selected Cluster


PCA + KMeans Cluster


Example Generated QA Pair

Question

How many exam attempts are allowed per module?

Answer

Students typically have three attempts to pass a module examination.
If the module is failed after the final attempt, it is considered definitively failed.

Installation

Clone the repository:

git clone https://github.com/TaherBoujnah/Extracting-Useful-Question-Answer-Pairs-from-Examination-Regulations.git

cd Extracting-Useful-Question-Answer-Pairs-from-Examination-Regulations

Install dependencies:

pip install -r backend/requirements.txt

Running the Backend API

Start the FastAPI service:

uvicorn backend.api:app --reload

Running the Chatbot

Launch the Chainlit interface:

chainlit run frontend/chatbot_chainlit.py

Then open the local interface in your browser.

Retrieval System

The chatbot uses semantic retrieval to match user questions with generated FAQ entries.

Pipeline:

Embed FAQ questions

Compute similarity with the user query

Return the most relevant answer

Embeddings and metadata are stored in:

data/embeddings.npy
data/embeddings_meta.json

Thesis Contribution

This thesis demonstrates how large language models can be used to transform long academic regulation documents into structured FAQ knowledge bases.

The project combines:

document preprocessing

LLM-based QA generation

dataset evaluation

semantic retrieval

chatbot deployment

into a complete end-to-end system.

Future Work

Possible extensions include:

improving QA filtering methods

adding multilingual support

integrating stronger retrieval models

extending the system to other faculties or universities

improving hallucination detection

Author

Taher Boujnah
Bachelor Thesis – Natural Language Processing
