import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18

num_classes = 29  # 26 letters + 'del', 'nothing', 'space'
labels = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "del",
    "nothing",
    "space",
]

model = resnet18(weights=None)  # load resnet18 with no weights
model.fc = nn.Linear(
    model.fc.in_features, num_classes
)  # change the final layer to match the class
model.load_state_dict(
    torch.load("ResNet18.pth", map_location=torch.device("cpu"))
)  # load trained weights
model.eval()  # set to evaluation mode

# define transform similar to validation dataset
transform = transforms.Compose(
    [
        transforms.ToPILImage(),  # convert opencv image to PIL image
        transforms.Resize((224, 224)),  # resize to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

cap = cv2.VideoCapture(0)  # start video capturing from the webcam

while True:
    ret, frame = cap.read()  # read the frame from the webcam
    if not ret:  # check is frame was captured
        break
    frame = cv2.flip(frame, 1)  # mirror the frame
    h, w = frame.shape[:2]  # height and widht of the frame
    x1, y1 = 1300, 300  # top-left of the ROI
    x2, y2 = x1 + 500, y1 + 500  # bottom-right of the ROI, 500x500 px

    # make sure ROI is within the frame
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    roi = frame[y1:y2, x1:x2]  # get the roi from the frame
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)  # convert BGR to RGB

    input_tensor = transform(roi)  # apply transform
    input_tensor = input_tensor.unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        output = model(input_tensor)
        pred = output.argmax(dim=1).item()  # get prediction
        label = labels[pred]  # get letter from prediction

    # draw the roi rectangle on the frame
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # put the predicted letter on top of the ROI
    cv2.putText(
        frame,
        f"Letter: {label}",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 0),
        2,
    )
    cv2.imshow("Real-Time Classification", frame)  # show the frame with prediction

    if cv2.waitKey(1) & 0xFF == ord("q"):  # quit if q is pressed
        break

cap.release()  # release webcam
cv2.destroyAllWindows()  # close window
