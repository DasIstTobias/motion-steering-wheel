import cv2
import numpy as np

# --- Konfiguration ---
TARGET_WIDTH = 400
TARGET_HEIGHT = 400
MIN_CONTOUR_AREA = 150  # Mindestfläche, um als Objekt erkannt zu werden (Pixel^2)

# HSV Farbbereiche für Rot (möglicherweise anpassen!)
# Rot ist im HSV-Farbraum "geteilt" (um 0 und um 170-180)
# Bereich 1:
LOWER_RED1 = np.array([0, 100, 100])    # H_min, S_min, V_min
UPPER_RED1 = np.array([10, 255, 255])   # H_max, S_max, V_max
# Bereich 2:
LOWER_RED2 = np.array([165, 100, 100])  # H_min, S_min, V_min
UPPER_RED2 = np.array([180, 255, 255])  # H_max, S_max, V_max

# Morphologischer Kernel für das Schließen von Lücken
# Eine größere Kernelgröße schließt größere Lücken, kann aber auch Objekte verbinden
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
        # Breiter als hoch: horizontal zuschneiden
        diff = (w_orig - h_orig) // 2
        cropped_frame = frame[:, diff:diff + h_orig]
    elif h_orig > w_orig:
        # Höher als breit: vertikal zuschneiden
        diff = (h_orig - w_orig) // 2
        cropped_frame = frame[diff:diff + w_orig, :]
    else:
        # Bereits 1:1
        cropped_frame = frame

    # --- Schritt 1.1: Zugeschnittenes Bild auf Zielgröße (400x400) skalieren ---
    # Dies ist das Bild, auf dem wir später zeichnen und das für die Farberkennung verwendet wird
    display_image = cv2.resize(cropped_frame, (TARGET_WIDTH, TARGET_HEIGHT))

    # --- Schritt 2: Rote Pixel isolieren (alles andere schwarz, rot wird weiß) ---
    hsv_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2HSV)

    # Erstelle Masken für die beiden roten Bereiche
    mask_red1 = cv2.inRange(hsv_image, LOWER_RED1, UPPER_RED1)
    mask_red2 = cv2.inRange(hsv_image, LOWER_RED2, UPPER_RED2)

    # Kombiniere die Masken
    red_mask = cv2.bitwise_or(mask_red1, mask_red2)
    # Dies entspricht "Bild 2" (schwarz-weiß Maske)

    # --- Schritt 3: Stellen zu Gruppen zusammenfassen (Lücken schließen) ---
    # Morphologische Operationen: Dilatation gefolgt von Erosion (Closing)
    # um kleine Lücken innerhalb der roten Objekte zu schließen.
    closed_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Erosion gefolgt von Dilatation (Opening), um kleine Rauschpunkte zu entfernen.
    processed_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Für "Bild 3" (unterschiedliche Blautöne für verbundene Komponenten):
    # Dies ist optional, da wir Konturen für die Blob-Erkennung verwenden.
    # Es dient hier nur der Visualisierung dieses Zwischenschritts.
    # num_labels, labels_im = cv2.connectedComponents(processed_mask)
    # colored_components_img = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
    # for i in range(1, num_labels): # Label 0 ist der Hintergrund
    #     colored_components_img[labels_im == i] = [np.random.randint(100, 256), np.random.randint(50, 150), 0] # Zufällige Blautöne


    # --- Schritt 4: Blob Detection und Koordinaten/Radius Ausgabe ---
    # Finde Konturen in der prozessierten Maske
    # cv2.RETR_EXTERNAL findet nur die äußeren Konturen
    # cv2.CHAIN_APPROX_SIMPLE komprimiert horizontale, vertikale und diagonale Segmente
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_objects = []
    output_visualization_frame = display_image.copy() # Kopie des 400x400 Originalbildes zum Zeichnen

    print("\n--- Detektierte Rote Objekte ---")
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > MIN_CONTOUR_AREA:
            # Finde den kleinsten umschließenden Kreis für die Kontur
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y)) # Koordinaten der Mitte (im 400x400 Raster)
            radius = int(radius)      # Radius des Objekts

            detected_objects.append({
                "id": i,
                "center_x": center[0],
                "center_y": center[1],
                "radius": radius
            })

            # Ausgabe der Koordinaten und des Radius
            print(f"Objekt ID {i}: Mitte=({center[0]}, {center[1]}), Radius={radius}")

            # --- Schritt 5: Rote Stellen blau einkreisen auf dem Originalbild (400x400) ---
            cv2.circle(output_visualization_frame, center, radius, (255, 0, 0), 2) # (Blau, Grün, Rot), 2px Dicke
            cv2.putText(output_visualization_frame, str(i), (center[0] - 10, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


    # --- Live-Video Fenster anzeigen ---
    cv2.imshow('Original (400x400) mit Detektion (Bild 4)', output_visualization_frame)
    cv2.imshow('Rote Maske (Bild 2)', red_mask) # Zeigt die rohe Rot-Maske
    cv2.imshow('Prozessierte Maske (fuer Konturen)', processed_mask) # Maske nach Morphologie
    # if 'colored_components_img' in locals(): # Nur wenn Schritt 3 Visualisierung aktiv ist
    #     cv2.imshow('Verbundene Komponenten (Bild 3 approx.)', colored_components_img)


    # Auf Tastendruck 'q' warten, um die Schleife zu beenden
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Aufräumen ---
cap.release()
cv2.destroyAllWindows()
print("Programm beendet.")