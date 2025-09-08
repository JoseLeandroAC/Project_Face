from deepface import DeepFace
import cv2
import os

# Define a pasta que contém as imagens das pessoas conhecidas
banco_de_dados = "imagens_conhecidas"

# Inicia a captura de vídeo da webcam. O '0' se refere à sua webcam padrão.
webcam = cv2.VideoCapture(0)
print("\nWebcam iniciada. Pressione 'q' para sair.")

while True:
    # Lê um frame da webcam
    ret, frame = webcam.read()
    if not ret:
        break

    # Tenta detectar e reconhecer o rosto no frame atual
    try:
        # DeepFace.find faz todo o trabalho: detecta o rosto e o compara com o banco de dados
        # O modelo "VGG-Face" é um dos mais precisos
        resultados = DeepFace.find(
            img_path=frame, 
            db_path=banco_de_dados, 
            model_name="VGG-Face"
        )

        # O DeepFace retorna uma lista de DataFrames, um para cada rosto encontrado
        if resultados and not resultados[0].empty:

            # Pega o primeiro resultado, que corresponde ao rosto mais provável
            resultado_mais_proximo = resultados[0]

            # Extrai o nome da pessoa a partir do caminho da imagem no banco de dados
            caminho_identidade = resultado_mais_proximo['identity'][0]
            nome_pessoa = caminho_identidade.split(os.path.sep)[-2]

            # Pega as coordenadas do rosto para desenhar o retângulo
            x, y, w, h = resultado_mais_proximo['source_x'][0], resultado_mais_proximo['source_y'][0], resultado_mais_proximo['source_w'][0], resultado_mais_proximo['source_h'][0]

            # Desenha o retângulo e o nome na tela
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, nome_pessoa, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            # Se a lista de resultados estiver vazia, o rosto é "Desconhecido"
            cv2.putText(frame, 'Desconhecido', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    except ValueError:
        # Este erro acontece se nenhum rosto for detectado no frame, o que é normal
        cv2.putText(frame, 'Nenhum rosto detectado', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Exibe o frame na janela
    cv2.imshow('Reconhecimento Facial', frame)

    # Pressione 'q' para sair do programa
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a webcam e fecha todas as janelas
webcam.release()
cv2.destroyAllWindows()