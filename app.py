import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from metrics import calculate_bleu, get_closest_reference, calculate_cosine_similarity, calculate_rouge_n, calculate_rouge_l
from references import reference_answers


# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Function to split the extracted text into smaller chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks


# Function to save the vector store locally
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


# Function to get the conversational chain for question answering
def get_conversational_chain():
    prompt_template = """
    You are a highly knowledgeable and empathetic healthcare assistant trained to provide in-depth, structured, and evidence-based responses in a conversational manner.
    When answering medical or healthcare-related questions, provide information in the following format:
    1. Start with a friendly acknowledgment of the user's input (e.g., "Hello! I'm here to help. That sounds concerning, let's explore it further.")
    2. Describe the condition and its causes in detail.
    3. Explain common and rare symptoms comprehensively.
    4. Offer detailed diagnosis and treatment advice, including prevention and management options.
    5. Provide thorough treatment options, including medications, therapies, and self-care tips.
    6. Conclude with encouragement or follow-up guidance (e.g., "I hope this helps! Let me know if you have more questions.").
    Always use clear and empathetic language. If you don't know the answer to something, acknowledge it respectfully and offer suggestions for where to seek help (e.g., "That's a great question. It might be best to consult with a medical professional for personalized advice.").
    Never give blank response.
    If the user asks a random question that is unrelated to the medical dataset provided, politely decline to answer and respond with: "I'm sorry, I don't have relevant information on that topic. Please feel free to ask me healthcare-related questions!"

    Context: {context}
    Question: {question}
    Respond in a detailed, helpful, and conversational manner below:
    Detailed answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


# Function to preprocess user input
def preprocess_user_input(user_question):
    user_question = user_question.strip().lower()
    if not user_question.endswith("?"):
        user_question += "?"
    return user_question


# Function to handle user input and get a response
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question, k=5)
    
    # Handle case where no relevant documents are found
    if not docs:
        return "I'm sorry, I couldn't find information on that topic in the book. Could you rephrase your question or ask about something else?"
    
    chain = get_conversational_chain()
    # Generate the initial response
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True).get("output_text", "").strip()

    # If the response is empty, reformulate the question and retry
    if response == "":
        reformulated_question = f"What is {user_question.strip()}?"
        response = chain({"input_documents": docs, "question": reformulated_question}, return_only_outputs=True).get("output_text", "").strip()
    
    # Retrieve the closest reference using the get_closest_reference function
    closest_reference = get_closest_reference(user_question, reference_answers)
    
    
    if closest_reference:
        print(f"Closest reference found: {closest_reference}")  # Log closest reference
        
        # Calculate Cosine Similarity
        cosine_similarity_score = calculate_cosine_similarity(closest_reference, response)
        print(f"Cosine Similarity Score: {cosine_similarity_score:.2f}")  # Log Cosine Similarity
        
        # Calculate BLEU score
        bleu_score = calculate_bleu(closest_reference, response,2)
        print(f"BLEU Score: {bleu_score:.2f}")  # Log BLEU Score

        # Calculate ROUGE-N (Unigram)
        precision_n, recall_n, f1_n = calculate_rouge_n(closest_reference, response, 1)
        print(f"ROUGE-1 Precision: {precision_n:.4f}, Recall: {recall_n:.4f}, F1: {f1_n:.4f}")

        # Calculate ROUGE-L
        precision_l, recall_l, f1_l = calculate_rouge_l(closest_reference, response)
        print(f"ROUGE-L Precision: {precision_l:.4f}, Recall: {recall_l:.4f}, F1: {f1_l:.4f}")

        
        # Append scores to the response
        response += f"\n\n(Cosine Similarity Score: {cosine_similarity_score:.2f},\n BLEU Score: {bleu_score:.2f},\n ROUGE-1 Precision: {precision_n:.4f}, Recall: {recall_n:.4f}, F1: {f1_n:.4f},\n ROUGE-L Precision: {precision_l:.4f}, Recall: {recall_l:.4f}, F1: {f1_l:.4f}\n Closest Reference: {closest_reference})"
    else:
        print(f"No reference found for the question: {user_question}")  # Log when no reference is found

    return response



# HTML + CSS for the enhanced chatbot UI
def render_chat_ui():
    st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif;
    }
    .chat-window {
        max-height: 400px;
        overflow-y: auto;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #ddd;
        background: linear-gradient(135deg, #e9f7fe, #fdf7fc);
    }
    .chat-bubble {
        margin: 10px 0;
        padding: 15px;
        border-radius: 15px;
        max-width: 70%;
        font-size: 15px;
        word-wrap: break-word;
    }
    .user-bubble {
        background: linear-gradient(135deg, #87ceeb, #add8e6);
        color: black;
        align-self: flex-end;
    }
    .bot-bubble {
        background: linear-gradient(135deg, #f3e5f5, #ce93d8);
        color: black;
        align-self: flex-start;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
    }
    .question-input-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 20px;
    }
    input[type="text"] {
        width: 70%;
        padding: 12px 20px;
        font-size: 16px;
        border-radius: 20px;
        border: 1px solid #007bff;
        background-color: #f0f8ff;
        color: #000;
        box-shadow: none;
        margin: 0 auto;
    }
    input[type="text"]:focus {
        outline: none;
        border: 1px solid #007bff;
        background-color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)


# Function to display messages in chat bubble format
def display_message(message, is_user=True):
    bubble_class = "user-bubble" if is_user else "bot-bubble"
    st.markdown(f'<div class="chat-bubble {bubble_class}">{message}</div>', unsafe_allow_html=True)


# Main function to build the Streamlit app
def main():
    st.set_page_config("Chat PDF", layout="wide")
    st.title("💬 Chat with Your AI Assistant")


    # Render custom chatbot UI
    render_chat_ui()


    # Provide PDF file path directly in code
    pdf_docs = ["Medical_book.pdf"]  # Replace with your actual PDF path


    # Directly process the PDF and vector store if not already processed
    if not os.path.exists("faiss_index"):
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        st.success("PDF files processed and vector store created successfully!")
    else:
        st.success("Vector store already exists, ready to chat!")


    # Chat section
    st.subheader("Ask a Question with Chatbot")
    st.markdown("Type your question below and press **Enter** to chat.")
    conversation_container = st.container()
   
    user_question = st.text_input(
        "Your Question",
        key="user_input",
        placeholder="Type your question here...",
        on_change=lambda: st.session_state.submit_flag.append(True),
    )
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "submit_flag" not in st.session_state:
        st.session_state.submit_flag = []


    # Process user input and generate response
    if st.session_state.submit_flag:
        st.session_state.submit_flag = []  # Clear the flag
        if user_question:

            # Preprocess the user question
            processed_question = preprocess_user_input(user_question)
            # Add user's question to the conversation
            st.session_state.conversation.append(("user", user_question))
            display_message(user_question, is_user=True)


            # Get and add the bot's response
            response = user_input(processed_question)
            st.session_state.conversation.append(("bot", response))
            display_message(response, is_user=False)


    # Show the full conversation
    with conversation_container:
        st.markdown('<div class="chat-window chat-container">', unsafe_allow_html=True)
        for sender, message in st.session_state.conversation:
            display_message(message, is_user=(sender == "user"))
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
