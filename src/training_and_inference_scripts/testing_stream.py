import cv2
from ultralytics import YOLO
import torch

# --- Konfiguration ---
# Die URL muss den Stream-Schlüssel einschließen, der in OBS verwendet wird.
rtmp_url = 'rtmp://localhost/live/teststream'
model_path = r"C:\Users\chris\foosball-statistics\weights\yolov11n_imgsz640_Topview.pt"

# --- Skript ---
# Check if a GPU is available and use it, otherwise fall back to CPU.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = YOLO(model_path).to(device)
cap = cv2.VideoCapture(rtmp_url)

if not cap.isOpened():
    print("Fehler: Stream konnte nicht geöffnet werden. Stellen Sie sicher, dass der Nginx-Server läuft und OBS aktiv streamt.")
    exit()

print("Stream erfolgreich geöffnet. Drücken Sie Ctrl+C zum Beenden.")

# Die predict-Funktion mit stream=False gibt eine Liste von Result-Objekten zurück.
while True:
    ret, frame = cap.read()
    if not ret:
        print("Ende des Streams erreicht.")
        break
    
    # Führen Sie die Vorhersage auf dem Frame aus.
    # Wir verwenden stream=False, da Sie annotated_frame = results[0].plot() verwenden wollen.
    results = model.predict(source=frame, stream=False, verbose=True)
    
    # Um zu überprüfen, ob die Vorhersage funktioniert, können Sie die Anzahl der erkannten Objekte ausgeben.
    # Diese Zeile ist optional, aber hilfreich zur Fehlersuche.
    # print(f"Detected {len(results[0].boxes)} objects.")

cap.release()
# cv2.destroyAllWindows()
