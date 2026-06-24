import time
import serial
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from picamera2 import Picamera2
from torchvision import transforms
import torch.nn.functional as F

# Serial settings ----------------------------------------------------------------------------------------
SERIAL_PORT = "/dev/ttyACM0"
BAUD = 115200

#  Model file -------------------------------------------------------------------------------------------
MODEL_PATH = Path("models/resnet18_best_ft.pt")

# Small timing delays -----------------------------------------------------------------------------------
STARTUP_DELAY = 2.0         # wait when starting the Pi program
AFTER_HOLD_DELAY = 1.0      # wait after Arduino says HOLD (arm is stable)

# Class order must match the training label order--------------------------------------------------------
CLASSES = ["color_pants", "color_shirts", "dresses", "jeans", "towel", "white"]


def load_model(device):
   # Load a trained PyTorch model from disk.
    model = torch.load(MODEL_PATH, map_location=device)
    model.eval()
    return model


def preprocess():
   # Image preprocessing:
    #- Resize to 224x224 (ResNet input size)
    #- Convert to tensor (numbers)
    #- Normalize (pixels to resnet range)

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])


def classify(model, device, tfm, rgb_array):
    #Classify a single camera frame:
    # Convert array -> PIL image
    # Apply transforms -> tensor
    # Pick the highest probability class
    
    img = Image.fromarray(rgb_array)
    x = tfm(img).unsqueeze(0).to(device)   

    with torch.no_grad():      # no gradients needed for inference - not in train right now
        logits = model(x)      # scores for each category
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]   # transfer into propabillity precents each category score

    idx = int(np.argmax(probs)) # return the biggest prop
    return CLASSES[idx], float(probs[idx]) #return the class index and the prop


def wait_for_line(ser, want, timeout=10.0):
    #Wait until we receive an exact line from Arduino, like "HOLD" or "DONE".
    #NEW LINE AFTER RECEVING '\n'.
    
    t0 = time.time()  #start time
    buf = b""

    while time.time() - t0 < timeout: # if the max time from start still not end
        chunk = ser.read(128) #read from serial

        if chunk:              # if theres information from serial in chunk
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)   #cut the info before the /n
                s = line.decode(errors="ignore").strip()   #transfer info from bytes to text
                if s == want:  #if the info match what we wanted to recieve
                    return True
        else:
            time.sleep(0.01)

    return False


def wait_for_prefix(ser, prefix, timeout=10.0):
    #Wait until Arduino sends a line that starts with a prefix -"ACK ".
    #Returns the full line or None.  #like the func wait_for_line but return the line

    t0 = time.time()
    buf = b""

    while time.time() - t0 < timeout:
        chunk = ser.read(128)

        if chunk:
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                s = line.decode(errors="ignore").strip()
                if s.startswith(prefix):
                    return s
        else:
            time.sleep(0.01)

    return None


def main():
    device = "cpu"

    # Load model + transforms
    model = load_model(device)
    tfm = preprocess()

    #  Camera init 
    # Use a fixed configuration so the arm pose matches what we trained on.
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()   # statring the cam
    time.sleep(1)  # let camera stabilize

    # Serial init 
    ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0.5) 

    # Small wait to let everything fully boot
    time.sleep(STARTUP_DELAY)
    print("Pi READY")

    busy = False
    last_full = 0.0

    while True:
        # Read one line from Arduino (status)
        line = ser.readline().decode(errors="ignore").strip()

        # Arduino sends FULL when sensor sees clothes and it is not busy
        if line == "FULL" and not busy:
            now = time.time()  #keeping the time (now)

            # ignore repeated "FULL" too quickly
            if now - last_full < 0.5:
                continue
            last_full = now

            # Tell Arduino to pick a cloth
            ser.write(b"PICK\n")
            busy = True

            #  Wait until Arduino says "HOLD" (= arm is stable for camera)
            ok = wait_for_line(ser, "HOLD", timeout=25.0)
            if not ok:
                busy = False
                continue  #going back to start and waiting

            # Wait a bit more, then capture ONE frame
            time.sleep(AFTER_HOLD_DELAY)
            frame = picam2.capture_array()   #taking photo

            # Run classification and choose the top class
            label, conf = classify(model, device, tfm, frame)
            print(f"Pred: {label} ({conf*100:.1f}%)") #accuracy of one num after the dot-prop

            # Send label to Arduino so it drops in the right basket
            ser.write(f"DROP {label}\n".encode())

            # Arduino replies "ACK <label>" when it receives the command
            ack = wait_for_prefix(ser, "ACK ", timeout=6.0)
            if ack:
                print("Arduino:", ack)

            # Wait until Arduino says DONE (= finished drop + return)
            done = wait_for_line(ser, "DONE", timeout=40.0)
            busy = False

        time.sleep(0.01)


if __name__ == "__main__":
    main()
