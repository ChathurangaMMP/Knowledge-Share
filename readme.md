# 📊 Data-Driven RAG Chunking Evaluation

This folder contains the source code and experimental data for my article: **"Stop Guessing: A Data-Driven Guide to RAG Chunking Strategies."**

It demonstrates how to quantitatively evaluate different text chunking strategies (Recursive, Token-based, and Semantic) using the **Ragas** framework and **LangChain**.

## 📂 Folder Structure

```text
RAGAS-evaluation/
├── data/                   # Contains raw source documents (e.g., Nexus-9 Datasheet)
├── extracted_data/         # Intermediate storage for processed chunks (optional debug logs)
├── output/                 # Final CSV results containing Ragas scores
└── ragas_evaluation_main.py # The main execution script
```

## 🚀 Overview

The script performs the following automated workflow:

1.  **Ingests** a complex technical document ("Nexus-9 QPU Datasheet").
2.  **Chunks** the text using three distinct strategies:
    * *Recursive Character Splitting* (Standard)
    * *Token-Based Splitting* (Fixed size)
    * *Semantic Splitting* (Embedding-based)
3.  **Generates** answers using a RAG pipeline (`gpt-4o-mini`).
4.  **Evaluates** the performance using **Ragas metrics**:
    * `Context Recall`: Did we retrieve the correct information?
    * `Context Precision`: Did we avoid retrieving noise?
    * `Faithfulness`: Did the LLM answer truthfully based on the context?

## 🛠️ Prerequisites

You will need an **OpenAI API Key** to run the generation and evaluation models.

### Installation

1.  Navigate to this folder:
    ```bash
    cd RAGAS-evaluation
    ```

2.  Install the required dependencies:
    ```bash
    pip install langchain langchain-openai langchain-community \
    langchain-experimental langchain-huggingface ragas faiss-cpu pandas openai
    ```

3.  Set your API key (Linux/Mac):
    ```bash
    export OPENAI_API_KEY="sk-..."
    ```
    *(Or set it directly in the python script if testing locally)*

## 🏃‍♂️ How to Run

Simply execute the main script:

```bash
python ragas_evaluation_main.py
```
### What happens next?

* The script will initialize the embedding models (**HuggingFace all-MiniLM-L6-v2**) locally.
* It will run the RAG pipeline for all defined scenarios.
* It will print a summary table of scores to the console.
* A detailed report will be saved to: `output/ragas_chunking_experiment_results.csv`.

## 📊 Sample Output

| Strategy | Context Recall | Context Precision | Faithfulness |
| :--- | :--- | :--- | :--- |
| **Recursive** | 1.00 | 0.83 | 1.00 |
| **Semantic** | 0.83 | 1.00 | 1.00 |
| **Token-Based** | 1.00 | 0.50 | 0.92 |

## 🔗 Article Link

Read the full breakdown of the methodology and results here:
* [Medium Article Link](Add your link here)
* [LinkedIn Post](Add your link here)

---
*Author: Prasad Chathuranga*