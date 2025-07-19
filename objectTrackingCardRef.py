import cv2
import numpy as np

# Real-world size of a credit card (cm)
CARD_WIDTH_CM = 8.56
CARD_HEIGHT_CM = 5.4
CARD_ASPECT_RATIO = CARD_WIDTH_CM / CARD_HEIGHT_CM

# Initialize webcam
cap = cv2.VideoCapture(1)  # Change to 0 if 1 doesn't work
cap.set(3, 1920)
cap.set(4, 1080)

pixel_per_cm = None


def is_card_candidate(cnt):
    area = cv2.contourArea(cnt)
    if area < 10000 or area > 150000:
        return False, None

    rect = cv2.minAreaRect(cnt)
    (w, h) = rect[1]
    if w == 0 or h == 0:
        return False, None

    aspect = max(w, h) / min(w, h)
    if abs(aspect - CARD_ASPECT_RATIO) > 0.08:
        return False, None

    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Ensure box is valid
    if box is None or len(box) < 4:
        return False, None

    return True, (box, max(w, h))


while True:
    ret, frame = cap.read()
    if not ret:
        break

    output = frame.copy()
    card_box = None  # Reset every frame

    # Preprocess
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)

    # Adaptive threshold for light cards
    thresh = cv2.adaptiveThreshold(blur, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)

    # Canny edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Combine both
    mask = cv2.bitwise_or(thresh, edges)

    # Morphological close to seal gaps
    kernel = np.ones((5, 5), np.uint8)
    clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    biggest_card = None
    biggest_size = 0

    # Try to find the card reference
    for cnt in contours:
        is_card, result = is_card_candidate(cnt)
        if is_card:
            box, size = result
            if size > biggest_size:
                biggest_card = box
                biggest_size = size
                pixel_per_cm = size / CARD_WIDTH_CM

    # Draw the detected card safely
    if biggest_card is not None and len(biggest_card) >= 4:
        card_box = biggest_card
        cx, cy = np.mean(card_box[:, 0]), np.mean(card_box[:, 1])
        cv2.drawContours(output, [card_box], 0, (255, 0, 0), 2)
        cv2.putText(output, "Card Ref", (int(cx - 60), int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Measure other objects
    if pixel_per_cm:
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 3000:
                continue

            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            (x, y) = rect[0]

            # Skip if it's the card itself
            if card_box is not None and cv2.pointPolygonTest(card_box, (x, y), False) >= 0:
                continue

            w, h = rect[1]
            if w == 0 or h == 0:
                continue

            width_cm = w / pixel_per_cm
            height_cm = h / pixel_per_cm

            cv2.drawContours(output, [box], 0, (0, 255, 0), 2)
            cv2.putText(output, f"{width_cm:.1f} x {height_cm:.1f} cm",
                        (int(x - 30), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)

    cv2.imshow("Frame", output)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
