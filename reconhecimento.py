import face_recognition
import cv2
import numpy as np
import os

# --- Parte 1: Carregar imagens e extrair características ---
#pip install face_recognition opencv-python

conhecidos_encodings = []
conhecidos_nomes = []
caminho_imagens = "imagens_conhecidas"

for nome_pasta in os.listdir(caminho_imagens):
    pasta = os.path.join(caminho_imagens, nome_pasta)
    if os.path.isdir(pasta):
        for nome_arquivo in os.listdir(pasta):
            caminho_arquivo = os.path.join(pasta, nome_arquivo)
            imagem = face_recognition.load_image_file(caminho_arquivo)
            
            # Tenta encontrar o rosto na imagem. Se não encontrar, pule para a próxima
            try:
                face_encoding = face_recognition.face_encodings(imagem)[0]
                conhecidos_encodings.append(face_encoding)
                conhecidos_nomes.append(nome_pasta)
                print(f"Rosto de '{nome_pasta}' carregado com sucesso.")
            except IndexError:
                print(f"Nenhum rosto encontrado em '{caminho_arquivo}'.")

# --- Parte 2: Acessar a webcam para reconhecimento em tempo real ---

webcam = cv2.VideoCapture(0)

print("\nWebcam iniciada. Pressione 'q' para sair.")

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    # Redimensiona o frame para processar mais rápido
    frame_pequeno = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
    # Converte de BGR (OpenCV) para RGB (face_recognition)
    rgb_frame_pequeno = cv2.cvtColor(frame_pequeno, cv2.COLOR_BGR2RGB)

    # Encontra todos os rostos e encodings no frame atual da webcam
    rostos_localizacoes = face_recognition.face_locations(rgb_frame_pequeno)
    rostos_encodings = face_recognition.face_encodings(rgb_frame_pequeno, rostos_localizacoes)

    # Itera sobre cada rosto detectado na webcam
    for (top, right, bottom, left), rosto_encoding in zip(rostos_localizacoes, rostos_encodings):
        # Compara o rosto da webcam com os rostos conhecidos
        matches = face_recognition.compare_faces(conhecidos_encodings, rosto_encoding)
        nome = "Desconhecido"

        # Usa a distância da face para encontrar a melhor correspondência
        distancias_faces = face_recognition.face_distance(conhecidos_encodings, rosto_encoding)
        melhor_correspondencia_indice = np.argmin(distancias_faces)

        if matches[melhor_correspondencia_indice]:
            nome = conhecidos_nomes[melhor_correspondencia_indice]

        # Ajusta as coordenadas para o frame original
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Desenha um retângulo e o nome na imagem
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, nome, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Exibe o frame
    cv2.imshow('Reconhecimento Facial', frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a webcam e fecha as janelas
webcam.release()
cv2.destroyAllWindows()