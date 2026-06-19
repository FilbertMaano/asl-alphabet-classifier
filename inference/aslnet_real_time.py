import cv2
import torch
import torch.nn as nn
from torchvision import transforms


# redefine AslNet from the training
class AslNet(nn.Module):
    def __init__(self, num_classes):
        super(AslNet, self).__init__()
        # first conv block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 2nd conv block
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 3rd conv block
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # fully connected layer
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=128 * 8 * 8, out_features=512)
        self.relu_fc = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)  # first conv
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)  # second conv
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)  # third conv
        x = self.relu3(x)
        x = self.maxpool3(x)

        x = self.flatten(x)  # flatten for dense layer
        x = self.fc1(x)  # first dense layer
        x = self.relu_fc(x)
        x = self.dropout(x)
        x = self.fc2(x)  # output layer

        return x


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

model = AslNet(num_classes)  # initialize AslNet
model.load_state_dict(
    torch.load("weights/AslNet.pth", map_location=torch.device("cpu"))
)  # load trained weights
model.eval()  # set to evaluation mode

# define transform similar to validation dataset
transform = transforms.Compose(
    [
        transforms.ToPILImage(),  # convert opencv image to PIL image
        transforms.Resize((64, 64)),  # resize to 64x64
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
