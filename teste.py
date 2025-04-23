from langchain_community.llms import Ollama

# Inicializar o modelo
llm = Ollama(model="deepseek-r1:1.5b")

# Teste básico
try:
    response = llm.invoke("Olá, como você está?")
    print("Resposta do modelo:", response)
except Exception as e:
    print("Erro ao invocar o modelo:", e)