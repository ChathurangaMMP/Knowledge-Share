import os
import asyncio
import pandas as pd
from openai import AsyncOpenAI, OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas.metrics.collections import Faithfulness, ContextPrecision, ContextRecall
from ragas.llms import llm_factory
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

NEXUS_9_CONTENT = ""
with open("extracted_data/Nexus_9_Datasheet.txt", "r") as f:
    NEXUS_9_CONTENT = f.read()

# 1. Configuration
# We use AsyncOpenAI for the Ragas evaluator to speed up multiple 
# concurrent scores
os.environ["OPENAI_API_KEY"] = "sk-..." 
openai_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

# 2. Models
# Embeddings: Local & Free (HuggingFace)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Generator: The LLM that answers your user's questions
generator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Evaluator: The "Judge" LLM that scores the quality
evaluator_llm = llm_factory(model="gpt-4o-mini", client=openai_client)

# 3. Metrics
# We initialize the specific metrics we want to track
faithfulness_metric = Faithfulness(llm=evaluator_llm)
context_precision = ContextPrecision(llm=evaluator_llm)
context_recall = ContextRecall(llm=evaluator_llm)

def get_chunks(strategy_name: str, text: str) -> list[Document]:
    raw_doc = [Document(page_content=text)]
    
    if strategy_name == "Recursive":
        # The industry standard: splits by paragraphs/newlines first
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        return splitter.split_documents(raw_doc)
        
    elif strategy_name == "Token-Based":
        # The brute force method: strict token limits (risks cutting sentences)
        splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=20)
        return splitter.split_documents(raw_doc)
        
    elif strategy_name == "Semantic":
        # The AI method: breaks text when the "topic" changes
        splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
        return splitter.split_documents(raw_doc)
    
    else:
        raise ValueError("Unknown Strategy")
    

def build_retriever(chunks):
    # Create a transient vector store for this specific test run
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 2})

def generate_answer(retriever, question):
    # 1. Retrieve
    retrieved_docs = retriever.invoke(question)
    context_text = "\n\n".join([d.page_content for d in retrieved_docs])
    retrieved_contexts = [d.page_content for d in retrieved_docs]
    
    # 2. Generate
    template = """Answer based ONLY on the context:
    {context}
    Question: {question}"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | generator_llm | StrOutputParser()
    
    answer = chain.invoke({"context": context_text, "question": question})
    return answer, retrieved_contexts


EVAL_SCENARIOS = [
    {
        "question": "What is the critical threshold for coolant flow rate?",
        "ground_truth": "The critical threshold is less than 10.0 L/min.",
        # Metric Target: Context Recall 
        # Why: This info is buried in a table. Naive chunking often splits headers from data,
        # causing the retriever to miss the 'row' containing the answer.
    },
    {
        "question": "What happens specifically if the Core Temperature exceeds 30 mK?",
        "ground_truth": "The automated failsafe triggers the irreversible Liquid Helium Dump protocol.",
        # Metric Target: Faithfulness 
        # Why: The 'Warning' text is visually separated from the table. If the chunk cuts off 
        # before the warning, the LLM might hallucinate a generic safety response.
    },
    {
        "question": "Can I power down the Nexus-9 immediately after Stage 4 cooling begins?",
        "ground_truth": "No. Once Stage 4 (Hyper-Cooling) begins, the system cannot be powered down for 4 hours.",
        # Metric Target: Context Precision
        # Why: The document lists 4 different stages. A poor retriever might pull "Stage 1" 
        # and "Stage 2" because they look similar, diluting the correct answer with noise.
    }
]

async def run_experiment():
    results = []
    strategies = ["Recursive", "Token-Based", "Semantic"]
    
    for strategy in strategies:
        print(f"Testing Strategy: {strategy}...")
        
        # 1. Chunk & Index
        chunks = get_chunks(strategy, NEXUS_9_CONTENT)
        retriever = build_retriever(chunks)
        
        # 2. Ask & Score
        for scenario in EVAL_SCENARIOS:
            q = scenario["question"]
            gt = scenario["ground_truth"]
            
            response, contexts = generate_answer(retriever, q)
            
            # 3. Calculate Ragas Metrics
            # Note: We use .ascore() for async execution
            scores = await asyncio.gather(
                faithfulness_metric.ascore(user_input=q, response=response, retrieved_contexts=contexts),
                context_precision.ascore(user_input=q, retrieved_contexts=contexts, reference=gt),
                context_recall.ascore(user_input=q, retrieved_contexts=contexts, reference=gt)
            )
            
            results.append({
                "Strategy": strategy,
                "Question": q,
                "Faithfulness": scores[0],
                "Precision": scores[1],
                "Recall": scores[2]
            })
            
    return pd.DataFrame(results)

if __name__ == "__main__":
    df = asyncio.run(run_experiment())
    
    print("\n=== FINAL EVALUATION RESULTS ===")
    
    # Group by strategy to see which performed best
    summary = df.groupby("Strategy")[["Recall", "Precision", "Faithfulness"]].mean()
    print(summary)
    
    # Save to CSV for your article
    df.to_csv("output/rag_chunking_experiment_results.csv", index=False)
    print("\nDetailed results saved to 'output/rag_chunking_experiment_results.csv'")