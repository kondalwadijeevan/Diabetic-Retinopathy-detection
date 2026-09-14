# Diabetic Retinopathy Detection — Full Execution Guide

Project: **A Lesion-Based Diabetic Retinopathy Detection Through Hybrid Deep Learning Model**
Files: `Main.py` (GUI), `test.py` (script version), `test1.py` (data extraction), `run.bat`, `requirements.txt`, `Untitled.ipynb`

---

## STEP 1 — Create Python 3.7 environment

This project uses old-style Keras/TensorFlow (`keras.optimizers.SGD(lr=...)`, `keras.utils.np_utils`), which only work on TF 1.x / old Keras. Use Python 3.7.

```bash
conda create -n dr_env python=3.7 -y
conda activate dr_env
```

---

## STEP 2 — Install dependencies

`requirements.txt` is a list of pip commands, not a real requirements file. Run them in this order:

```bash
pip install numpy==1.19.2
pip install pandas==0.25.3
pip install matplotlib==3.1.1
pip install keras==2.3.1
pip install tensorflow==1.14.0
pip install h5py==2.10.0
pip install protobuf==3.16.0
pip install scikit-learn==0.22.2.post1
pip install shap==0.38.1
pip install jupyter==1.0.0
pip install jupyter-client==6.1.3
pip install jupyter-console==6.4.0
pip install jupyter-core==4.6.3
pip install jupyterlab-widgets==1.0.0
pip install seaborn==0.10.1
pip install ipython==7.9.0
pip install ipython-genutils==0.2.0
pip install ipykernel==6.5.0
pip install opencv-python==4.1.1.26
pip install opencv-contrib-python==4.3.0.36
```

> If any exact version fails to resolve on your OS, tell me your OS/architecture and I'll help find the closest compatible version.

---

## STEP 3 — Create the required folder structure

None of the scripts auto-create folders — you must make them manually in the project root (same folder as `Main.py`):

```bash
mkdir -p Dataset/images
mkdir -p SelectedImages/Normal
mkdir -p SelectedImages/DR
mkdir -p SelectedImages/MH
mkdir -p SelectedImages/ODC
mkdir -p model
```

Final folder layout should look like:

```
project/
├── Main.py
├── test.py
├── test1.py
├── run.bat
├── requirements.txt
├── Dataset/
│   ├── images/                  <- raw dataset images go here
│   └── RFMiD_Training_Labels.csv
├── SelectedImages/
│   ├── Normal/
│   ├── DR/
│   ├── MH/
│   └── ODC/
└── model/                       <- trained weights/history saved here
```

---

## STEP 4 — Download the dataset

Download from:
**https://riadd.grand-challenge.org/download-all-classes**

Place:
- All fundus images → `Dataset/images/*.png`
- Labels CSV → `Dataset/RFMiD_Training_Labels.csv`

The CSV must contain columns: `ID`, `Disease_Risk`, `DR`, `MH`, `ODC`.

---

## STEP 5 — Extract the 4 target classes (`test1.py`)

This script reads the CSV and sorts images into 4 class folders based on their label columns.

**Code (`test1.py`):**
```python
import pandas as pd
import numpy as np
import cv2

dataset = pd.read_csv("Dataset/RFMiD_Training_Labels.csv", usecols=['ID', 'Disease_Risk', 'DR', 'MH', 'ODC'])
names = dataset['ID'].ravel()
normal = dataset['Disease_Risk'].ravel()
dr = dataset['DR'].ravel()
mh = dataset['MH'].ravel()
odc = dataset['ODC'].ravel()

for i in range(len(normal)):
    if normal[i] == 0:
        img = cv2.imread("Dataset/images/"+str(names[i])+".png")
        cv2.imwrite("SelectedImages/Normal/"+str(names[i])+".png", img)
        print("normal")

for i in range(len(dr)):
    if dr[i] == 1:
        img = cv2.imread("Dataset/images/"+str(names[i])+".png")
        cv2.imwrite("SelectedImages/DR/"+str(names[i])+".png", img)
        print("dr")

for i in range(len(mh)):
    if mh[i] == 1:
        img = cv2.imread("Dataset/images/"+str(names[i])+".png")
        cv2.imwrite("SelectedImages/MH/"+str(names[i])+".png", img)
        print("mh")

for i in range(len(odc)):
    if odc[i] == 1:
        img = cv2.imread("Dataset/images/"+str(names[i])+".png")
        cv2.imwrite("SelectedImages/ODC/"+str(names[i])+".png", img)
        print("odc")
```

**Run it:**
```bash
python test1.py
```

**What it does:**
- `Disease_Risk == 0` → healthy eye → copied to `SelectedImages/Normal/`
- `DR == 1` → Diabetic Retinopathy → copied to `SelectedImages/DR/`
- `MH == 1` → Media Haze → copied to `SelectedImages/MH/`
- `ODC == 1` → Optic Disc Cupping → copied to `SelectedImages/ODC/`

After this finishes, verify each `SelectedImages/<class>/` folder actually has images in it before moving on.

---

## STEP 6 — Launch the GUI (`Main.py`)

```bash
python Main.py
```
or on Windows, double-click:
```bat
python Main.py
pause
```
(this is exactly what `run.bat` contains)

This opens a Tkinter window titled **"Diabetic Retinopathy"** with 8 buttons and an output text box.

---

## STEP 7 — Click "Upload Dataset"

**What runs (`UploadDataset()` in `Main.py`):**
```python
def UploadDataset():
    global filename, dataset, labels, X_train, Y_train, text, X, Y
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir='Dataset')
    text.insert(END, filename+" loaded\n\n")

    if os.path.exists('model/X.txt.npy'):
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
    else:
        X = []
        Y = []
        for root, dirs, directory in os.walk(path):
            for j in range(len(directory)):
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j])
                    img = cv2.resize(img, (32, 32))
                    X.append(img)
                    label = getLabel(name)
                    Y.append(label)
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save('model/X.txt', X)
        np.save('model/Y.txt', Y)

    text.insert(END, "Dataset Loading Completed\n")
    text.insert(END, "Total images found in dataset Before Augmentation : "+str(X.shape[0])+"\n")

    # bar chart of class counts before augmentation
    names, count = np.unique(Y, return_counts=True)
    plt.figure(figsize=(5, 3))
    plt.bar(np.arange(len(labels)), count)
    plt.xticks(np.arange(len(labels)), labels)
    plt.xlabel("Dataset Class Label Graph Before Augmentation")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.show()
```

**What to do:** In the folder dialog that pops up, select the `SelectedImages` folder (the one containing `Normal/`, `DR/`, `MH/`, `ODC/`).

**Result:** All images resized to 32×32, saved as `model/X.txt.npy` and `model/Y.txt.npy` for reuse, and a bar chart of class distribution is displayed.

---

## STEP 8 — Click "Preprocess Dataset"

**What runs (`PreprocessDataset()`):**
```python
def PreprocessDataset():
    global filename, dataset, labels, vectorizer
    global X, Y, X_train, X_test, y_train, y_test, X_train, X_val, y_train, y_val

    text.delete('1.0', END)
    X = X.astype('float32')
    X = X/255                                  # normalize pixels 0-1
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)                 # shuffle
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)                      # one-hot encode labels
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    if os.path.exists('model/aug_X.txt.npy'):
        X = np.load('model/aug_X.txt.npy')
        Y = np.load('model/aug_Y.txt.npy')
    else:
        aug = ImageDataGenerator(rotation_range=15, shear_range=0.8, horizontal_flip=True)
        data = aug.flow(X_train, y_train, 1)
        X = []
        Y = []
        for x, y in data:
            X.append(x[0])
            Y.append(y[0])
            if len(Y) > 30000:
                break
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save('model/aug_X.txt', X)
        np.save('model/aug_Y.txt', Y)

    text.insert(END, "Image Augmentation Completed")
    text.insert(END, "Total images found in dataset After Augmentation : "+str(X.shape[0])+"\n")

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

    text.insert(END, "Training Images Size   : "+str(X_train.shape[0])+"\n")
    text.insert(END, "Validation Images Size : "+str(X_val.shape[0])+"\n")
    text.insert(END, "Testing Images Size    : "+str(X_test.shape[0])+"\n")
    # bar chart of class counts after augmentation
    ...
```

**What to do:** Just click the button — no dialog appears.

**Result:**
- Pixel values normalized to [0,1]
- Data shuffled
- Labels one-hot encoded
- Augmentation applied (rotation ±15°, shear 0.8, horizontal flip) to fix class imbalance, generating up to ~30,000 images
- Final split: 80% train / 20% test, then train further split 80/20 into train/validation
- A second bar chart shows the balanced class distribution after augmentation

⚠️ First run is slow (augmentation loop runs until 30,000 images or generator exhausted). Cached to `model/aug_X.txt.npy` / `aug_Y.txt.npy` afterward.

---

## STEP 9 — Click "Train CNN" (baseline model, SGD optimizer)

**Architecture (`trainCNN()`):**
```python
eyenet_model = keras.models.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=(11,11), strides=(4,4), activation='relu',
                         input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3])),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(1,1), strides=(2,2)),
    keras.layers.Conv2D(filters=16, kernel_size=(9,9), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(1,1), strides=(2,2)),
    keras.layers.Conv2D(filters=8, kernel_size=(7,7), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(1,1), strides=(2,2)),
    keras.layers.Conv2D(filters=8, kernel_size=(6,6), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(1,1), strides=(2,2)),
    keras.layers.Conv2D(filters=8, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(1,1), strides=(2,2)),
    keras.layers.Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=8, kernel_size=(3,3), strides=(2,2), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(1,1), strides=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(y_train.shape[1], activation='softmax')
])

opt = SGD(lr=0.001)
eyenet_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

if os.path.exists("model/sgd_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/sgd_weights.hdf5', verbose=1, save_best_only=True)
    hist = eyenet_model.fit(X_train, y_train, epochs=40, validation_data=(X_test, y_test),
                             callbacks=[model_check_point], verbose=1)
    pickle.dump(hist.history, open('model/sgd_history.pckl', 'wb'))
else:
    eyenet_model.load_weights("model/sgd_weights.hdf5")

# evaluate
predict = eyenet_model.predict(X_test)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test, axis=1)
calculateMetrics("Existing CNN", predict, y_test1)
```

**What to do:** Click **"Train CNN"**.
**Result:** Trains for 40 epochs (first run only — later runs load cached `model/sgd_weights.hdf5`). Displays Accuracy, Precision, Recall, F-Score in the text box (~80% per the project docs) and a confusion matrix heatmap.

---

## STEP 10 — Click "Train Resnet50" (Adam optimizer)

**Code (`trainResnet50()`):**
```python
opt = Adam(lr=0.001)
eyenet_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

if os.path.exists("model/adam_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/adam_weights.hdf5', verbose=1, save_best_only=True)
    hist = eyenet_model.fit(X_train, y_train, epochs=40, validation_data=(X_test, y_test),
                             callbacks=[model_check_point], verbose=1)
    pickle.dump(hist.history, open('model/adam_history.pckl', 'wb'))
else:
    eyenet_model.load_weights("model/adam_weights.hdf5")

predict = eyenet_model.predict(X_test)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test, axis=1)
calculateMetrics("Propose Resnet50", predict, y_test1)
```

**What to do:** Click **"Train Resnet50"** (must click "Train CNN" first — this reuses `eyenet_model`).
**Result:** Same architecture, retrained with Adam optimizer (~84% accuracy per docs). Weights cached to `model/adam_weights.hdf5`.

---

## STEP 11 — Click "Train with Adam & Resnet50" (Extension model)

**Code (`trainHybrid()`):**
```python
extension_model = Sequential()
extension_model.add(InputLayer(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3])))
extension_model.add(Conv2D(25, (5,5), activation='relu', strides=(1,1), padding='same'))
extension_model.add(MaxPool2D(pool_size=(2,2), padding='same'))
extension_model.add(Conv2D(50, (5,5), activation='relu', strides=(2,2), padding='same'))
extension_model.add(MaxPool2D(pool_size=(2,2), padding='same'))
extension_model.add(BatchNormalization())
extension_model.add(Conv2D(70, (3,3), activation='relu', strides=(2,2), padding='same'))
extension_model.add(MaxPool2D(pool_size=(2,2), padding='valid'))   # <-- 'valid' padding = the key change
extension_model.add(BatchNormalization())
extension_model.add(Flatten())
extension_model.add(Dense(units=100, activation='relu'))
extension_model.add(Dense(units=100, activation='relu'))
extension_model.add(Dropout(0.25))
extension_model.add(Dense(units=y_train.shape[1], activation='softmax'))
extension_model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

if os.path.exists("model/extension_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/extension_weights.hdf5', verbose=1, save_best_only=True)
    hist = extension_model.fit(X_train, y_train, epochs=40, validation_data=(X_test, y_test),
                                callbacks=[model_check_point], verbose=1)
    pickle.dump(hist.history, open('model/extension_history.pckl', 'wb'))
else:
    extension_model.load_weights("model/extension_weights.hdf5")

predict = extension_model.predict(X_test)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test, axis=1)
calculateMetrics("Extension Adam & Resnet50", predict, y_test1)
```

**What to do:** Click **"Train with Adam & Resnet50"**.
**Result:** New, smaller architecture mixing `'same'` and `'valid'` padding to reduce parameters → best accuracy (~95% per docs). This is the model used for final predictions.

---

## STEP 12 — Click "Comparision Graph"

**Code (`graph()`):**
```python
def graph():
    algorithms = ['Existing CNN', 'Propose Resnet50', 'Extension Adam & Resnet50']
    data = []
    for i in range(len(accuracy)):
        data.append([algorithms[i], accuracy[i], precision[i], recall[i], fscore[i]])
    data = pd.DataFrame(data, columns=['Algorithm Name', 'Accuracy', 'Precision', 'Recall', 'FSCORE'])
    text.insert(END, str(data)+"\n")

    df = pd.DataFrame([...])   # long-form table of all metrics for all 3 models
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar', figsize=(8, 2))
    plt.title("All Algorithms Performance Graph")
    plt.show()
```

**Requires:** Steps 9, 10, and 11 to have been run first (metrics lists must be populated).
**Result:** Bar chart comparing Accuracy/Precision/Recall/F-Score across all 3 models, plus a printed table in the text box.

---

## STEP 13 — Click "Accuracy & Loss Graph"

**Code (`AccuracyGraph()`):**
```python
def AccuracyGraph():
    sgd_acc, sgd_loss = values("model/sgd_history.pckl", "accuracy", "loss")
    adam_acc, adam_loss = values("model/adam_history.pckl", "accuracy", "loss")
    extension_acc, extension_loss = values("model/extension_history.pckl", "accuracy", "loss")

    plt.figure(figsize=(6,4))
    plt.grid(True)
    plt.xlabel('EPOCH')
    plt.ylabel('Accuracy')
    plt.plot(sgd_acc, 'ro-', color='green')
    plt.plot(adam_acc, 'ro-', color='blue')
    plt.plot(extension_acc, 'ro-', color='black')
    plt.legend(['Existing CNN', 'Propose Resnet50', 'Extension Adam & Resnet50'], loc='lower right')
    plt.title('All Algorithm Training Accuracy Graph')
    plt.show()
```

**Result:** Line graph of training accuracy per epoch (first 20 epochs) for all 3 models overlaid — green = CNN/SGD, blue = Resnet50/Adam, black = Extension.

---

## STEP 14 — Click "Predict from Test"

**Code (`predict()`):**
```python
def predict():
    image_path = filedialog.askopenfilename(initialdir='.')
    image = cv2.imread(image_path)
    img = cv2.resize(image, (32,32))
    im2arr = np.array(img).reshape(1,32,32,3)
    img = im2arr.astype('float32')/255

    predict = extension_model.predict(img)     # uses the BEST (extension) model
    predict = np.argmax(predict)

    img = cv2.imread(image_path)
    img = cv2.resize(img, (400,300))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.putText(img, 'Predicted As : '+labels[predict], (10,25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    plt.figure(figsize=(4,3))
    plt.imshow(img)
    plt.show()
```

**What to do:** Click **"Predict from Test"**, then in the file picker select any single retinal fundus image (`.png`/`.jpg`).
**Requires:** Step 11 must have run first (`extension_model` must exist).
**Result:** Displays the selected image with the predicted class label (`Normal`, `DR`, `MH`, or `ODC`) overlaid in red text.

---

## Full Order of Operations (Quick Reference)

```text
1. conda create -n dr_env python=3.7 -y && conda activate dr_env
2. pip install <all packages from requirements.txt, one by one>
3. mkdir Dataset/images, SelectedImages/{Normal,DR,MH,ODC}, model
4. Download dataset from https://riadd.grand-challenge.org/download-all-classes
5. python test1.py                     # sorts images into SelectedImages/*
6. python Main.py   (or run.bat)       # opens GUI
7. Click "Upload Dataset"      → select SelectedImages folder
8. Click "Preprocess Dataset"  → normalize/shuffle/augment/split
9. Click "Train CNN"           → SGD, ~80% acc
10. Click "Train Resnet50"     → Adam, ~84% acc
11. Click "Train with Adam & Resnet50" → Extension, ~95% acc
12. Click "Comparision Graph"  → compare all 3 models
13. Click "Accuracy & Loss Graph" → epoch-wise training curves
14. Click "Predict from Test"  → classify a new image
```

## Common Pitfalls
- **Skipping step 3/5**: "Upload Dataset" finds 0 images if `SelectedImages/` subfolders don't exist or are empty.
- **`lr=` argument errors**: If you use a newer Keras (not 2.3.1), `SGD(lr=0.001)` / `Adam(lr=0.001)` will throw an error — Keras 2.4+ renamed `lr` to `learning_rate`. Stick to the pinned version.
- **Clicking buttons out of order**: `trainResnet50()` and `trainHybrid()` reuse `eyenet_model`/`X_train` set up by earlier steps — clicking them before "Upload Dataset" → "Preprocess Dataset" → "Train CNN" will raise `NameError`.
- **Re-running training**: Once `model/*.hdf5` exists, buttons just load cached weights instead of retraining — delete the corresponding `.hdf5` file if you want to force retraining.
