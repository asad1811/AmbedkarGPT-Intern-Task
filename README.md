# AmbedkarGPT-Intern-Task

Project Description-A local Retrieval-Augmented Generation (RAG) system that answers questions solely based on Dr. B. R. Ambedkar's speech snippet as in the speech.txt file using LangChain, ChromaDB, HuggingFace embeddings, and Ollama (Mistral 7B). Runs 100% offline, no API keys required.

Folder Structure on your system-

Project/main.py

Project/speech.txt

Project/requirements.txt


Setup-

1.Install Ollama-

curl -fsSL https://ollama.com/install.sh | sh

Can be directly downloaded from the website on windows-https://ollama.com/download

2.Install Mistral 7b-

ollama pull mistral:7b

3.Install Dependencies

pip install -r requirements.txt

4.Run the program

python main.py





