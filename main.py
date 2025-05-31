import cv2
import numpy as np

# --- Konfiguration ---
TARGET_WIDTH = 400
TARGET_HEIGHT = 400
# Erhöhe MIN_CONTOUR_AREA, wenn kleine, falsch erkannte Objekte ein Problem sind.
# Für tomatengroße Objekte könnte ein Wert zwischen 300 und 1000 sinnvoll sein,
# je nach Abstand zur Kamera und der tatsächlichen Größe im Bild.
MIN_CONTOUR_AREA = 200  # Mindestfläche, um als Objekt erkannt zu werden (Pixel^2)

# HSV Farbbereiche für Rot (möglicherweise weiter anpassen!)
# Um die Empfindlichkeit zu reduzieren und nur "starkes" Rot zu erkennen,
# erhöhen wir die S_min (Sättigung min) und V_min (Helligkeit/Value min) Werte.

# Bereich 1:
LOWER_RED1 = np.array([0, 130, 100])    # H_min, S_min, V_min
UPPER_RED1 = np.array([10, 255, 255])   # H_max, S_max, V_max
# Bereich 2:
LOWER_RED2 = np.array([160, 130, 100])  # H_min, S_min, V_min
UPPER_RED2 = np.array([180, 255, 255])  # H_max, S_max, V_max

# Hinweise zur Anpassung der HSV-Werte:
# H (Hue/Farbton):
#   - Der Bereich für Rot liegt typischerweise um 0-10 und 160-180 im OpenCV HSV-Raum (0-180).
# S (Saturation/Sättigung):
#   - Ein höherer S_min-Wert (z.B. 120-150) erfordert sattere Farben. Verwaschene Rottöne werden ignoriert.
#   - S_max ist meist 255.
# V (Value/Brightness/Helligkeit):
#   - Ein höherer V_min-Wert (z.B. 100-150) erfordert hellere Farben. Sehr dunkle Rottöne werden ignoriert.
#   - V_max ist meist 255.
#
# Wenn du immer noch Fehlalarme hast, versuche S_min und V_min in LOWER_RED1 und LOWER_RED2
# weiter schrittweise zu erhöhen (z.B. in 10er-Schritten).
# Wenn du echte rote Objekte nicht mehr erkennst, musst du die Werte wieder etwas senken.

# Morphologischer Kernel für das Schließen von Lücken
MORPH_KERNEL_SIZE = (7, 7)
kernel = np.ones(MORPH_KERNEL_SIZE, np.uint8)

# --- Webcam initialisieren ---
cap = cv2.VideoCapture(0) # 0 ist meist die Standard-Webcam
if not cap.isOpened():
    print("Fehler: Webcam konnte nicht geöffnet werden.")
    exit()

print("Webcam gestartet. Drücke 'q' zum Beenden.")
print("Koordinaten und Radien werden live in der Konsole ausgegeben.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fehler: Frame konnte nicht gelesen werden (Stream Ende?). Beende...")
        break

    # --- Schritt 1: Bild auf 1:1 Seitenverhältnis zuschneiden ---
    h_orig, w_orig = frame.shape[:2]
    if w_orig > h_orig:
        diff = (w_orig - h_orig) // 2
        cropped_frame = frame[:, diff:diff + h_orig]
    elif h_orig > w_orig:
        diff = (h_orig - w_orig) // 2
        cropped_frame = frame[diff:diff + w_orig, :]
    else:
        cropped_frame = frame

    # --- Schritt 1.1: Zugeschnittenes Bild auf Zielgröße (400x400) skalieren ---
    display_image = cv2.resize(cropped_frame, (TARGET_WIDTH, TARGET_HEIGHT))

    # --- Schritt 2: Rote Pixel isolieren ---
    hsv_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2HSV)
    mask_red1 = cv2.inRange(hsv_image, LOWER_RED1, UPPER_RED1)
    mask_red2 = cv2.inRange(hsv_image, LOWER_RED2, UPPER_RED2)
    red_mask = cv2.bitwise_or(mask_red1, mask_red2)

    # --- Schritt 3: Stellen zu Gruppen zusammenfassen (Lücken schließen & Rauschen entfernen) ---
    closed_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    processed_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # --- Schritt 4: Blob Detection und Koordinaten/Radius Ausgabe ---
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_visualization_frame = display_image.copy()

    # Nur wenn Objekte gefunden werden, um die Konsolenausgabe sauberer zu halten
    if contours:
        print("\n--- Detektierte Rote Objekte ---")

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > MIN_CONTOUR_AREA:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)

            # Zusätzliche Filterung (optional): Überprüfe, ob der Radius eine plausible Größe hat
            # if radius < 5 or radius > 100: # Beispiel: Ignoriere sehr kleine oder sehr große Kreise
            #     continue

            print(f"Objekt ID {i}: Mitte=({center[0]}, {center[1]}), Radius={radius}")

            # --- Schritt 5: Rote Stellen blau einkreisen ---
            cv2.circle(output_visualization_frame, center, radius, (255, 0, 0), 2)
            cv2.putText(output_visualization_frame, str(i), (center[0] - 10, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # --- Live-Video Fenster anzeigen ---
    cv2.imshow('Original (400x400) mit Detektion (Bild 4)', output_visualization_frame)
    cv2.imshow('Rote Maske (Bild 2)', red_mask)
    cv2.imshow('Prozessierte Maske (fuer Konturen)', processed_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Aufräumen ---
cap.release()
cv2.destroyAllWindows()
print("Programm beendet.")