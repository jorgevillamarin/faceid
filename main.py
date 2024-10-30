import cv2
import face_recognition
import pickle
import os

#instalar dependencias
#pip install -r requirements.txt


# Archivo para almacenar codificaciones de rostros conocidos
KNOWN_FACES_FILE = "rostros_conocidos.pkl"

# Función para cargar los rostros conocidos desde el archivo
def cargar_rostros_conocidos():
    if os.path.exists(KNOWN_FACES_FILE) and os.path.getsize(KNOWN_FACES_FILE) > 0:
        with open(KNOWN_FACES_FILE, "rb") as f:
            return pickle.load(f)
    return [], []

# Función para guardar los rostros conocidos en el archivo
def guardar_rostros_conocidos(encodings, names):
    with open(KNOWN_FACES_FILE, "wb") as f:
        pickle.dump((encodings, names), f)

# Cargar codificaciones y nombres conocidos
known_face_encodings, known_face_names = cargar_rostros_conocidos()

# Inicializar la cámara
video_capture = cv2.VideoCapture(0)

while True:
    # Captura un solo frame de video
    ret, frame = video_capture.read()
    
    # Convierte la imagen de BGR (OpenCV) a RGB (face_recognition)
    rgb_frame = frame[:, :, ::1]
    
    # Encuentra todas las ubicaciones y codificaciones de rostros en el frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Recorre cada rostro detectado en el frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compara el rostro detectado con los rostros conocidos
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        
        name = "Desconocido"
        
        # Si hay coincidencia, obtiene el nombre
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        else:
            # Si no hay coincidencia, pedir el nombre del nuevo rostro
            name = input("Nuevo rostro detectado. Ingresa el nombre de esta persona: ")
            known_face_encodings.append(face_encoding)
            known_face_names.append(name)
            print(f"Rostro guardado como {name}")
            
            # Guardar los rostros actualizados en el archivo inmediatamente
            guardar_rostros_conocidos(known_face_encodings, known_face_names)

        # Dibujar el recuadro y nombre alrededor del rostro
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Muestra el resultado en una ventana
    cv2.imshow("Reconocimiento Facial", frame)

    # Salir del bucle al presionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera la cámara y cierra la ventana
video_capture.release()
cv2.destroyAllWindows()
