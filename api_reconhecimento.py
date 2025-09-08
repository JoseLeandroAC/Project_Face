from flask import Flask, request, jsonify
from deepface import DeepFace
import cv2
import numpy as np
import os

app = Flask(__name__)

# Define a pasta que contém as imagens das pessoas conhecidas
banco_de_dados = "imagens_conhecidas"

print("\nPreparando o banco de dados de rostos...")
# Este passo vai processar suas imagens e criar os embeddings
# Pode demorar um pouco na primeira vez que você rodar o código
try:
    DeepFace.find(
        img_path=r"C:\José\20250905_075120.jpg", 
        db_path=banco_de_dados,
        model_name="VGG-Face",
        silent=True
    )
    print("Banco de dados de rostos pronto. O servidor está inicializando.")
except Exception as e:
    print(f"Erro ao preparar o banco de dados: {e}")
    print("Certifique-se de que a pasta 'imagens_conhecidas' existe e contém subpastas com imagens de rostos.")

@app.route('/reconhecer', methods=['POST'])
def reconhecer_rosto():
    # Verifica se a requisição contém um arquivo de imagem
    if 'file' not in request.files:
        return jsonify({"status": "erro", "mensagem": "Nenhum arquivo de imagem foi enviado."}), 400

    file = request.files['file']
    
    # Se o nome do arquivo estiver vazio, retorne um erro
    if file.filename == '':
        return jsonify({"status": "erro", "mensagem": "Nome do arquivo inválido."}), 400

    try:
        # Lê a imagem e a converte para o formato que o OpenCV entende
        img_bytes = file.read()
        img_array = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Usa o DeepFace para encontrar o rosto no banco de dados
        # O resultado é um DataFrame com informações sobre o rosto encontrado
        resultados = DeepFace.find(
            img_path=frame, 
            db_path=banco_de_dados, 
            model_name="VGG-Face"
        )
        
        # Se um rosto foi encontrado e a lista de resultados não está vazia
        if resultados and not resultados[0].empty:
            # Pega o nome da pessoa a partir do caminho da imagem
            caminho_identidade = resultados[0]['identity'][0]
            nome_pessoa = caminho_identidade.split(os.path.sep)[-2]
            
            # Retorna o nome da pessoa em formato JSON
            return jsonify({"status": "sucesso", "nome": nome_pessoa}), 200
        else:
            # Se a IA não encontrou correspondência, retorne "Desconhecido"
            return jsonify({"status": "sucesso", "nome": "Desconhecido"}), 200

    except ValueError:
        # Este erro ocorre se a IA não conseguir detectar nenhum rosto na imagem
        return jsonify({"status": "sucesso", "nome": "Nenhum rosto detectado"}), 200
    except Exception as e:
        # Para qualquer outro erro inesperado, retorne uma mensagem genérica
        return jsonify({"status": "erro", "mensagem": str(e)}), 500

if __name__ == '__main__':
    # Roda a aplicação Flask na sua rede local
    app.run(host='0.0.0.0', port=5000)