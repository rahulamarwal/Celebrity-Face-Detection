from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import svm
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import shutil
import joblib
import json
import pywt
import cv2
import os



def w2d(img,mode='haar',level=1):
    imArray =img
    imArray = cv2.cvtColor(imArray, cv2.COLOR_BGR2GRAY)
    imArray = np.float32(imArray)
    imArray /= 255
    coeffs = pywt.wavedec2(imArray,mode,level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0
    imArray_H = pywt.waverec2(coeffs_H,mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)

    return imArray_H


def cropped_img_if_2_eyes(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color
        else:
            return None

def append_img_path_to_list(path_to_data,):
    img_dirs = []
    for entry in os.scandir(path_to_data):
        if entry.is_dir():
            img_dirs.append(entry.path)
    return img_dirs

def create_cropped(path_to_cr_data):
    if os.path.exists(path_to_cr_data):
        shutil.rmtree(path_to_cr_data)
    os.mkdir(path_to_cr_data)

model_params = {
    'svm':{
        'model':svm.SVC(gamma='auto',probability=True),
        'params':{
            'svc__C':[1,10,100,1000],
            'svc__kernel':['rbf','linear']
        }
    },
    'random_forest':{
            'model':RandomForestClassifier(),
            'params':{
                'randomforestclassifier__n_estimators':[1,5,10]
        }
    },
    'logistic_regression':{
            'model':LogisticRegression(solver='liblinear',multi_class='auto'),
            'params':{
                'logisticregression__C':[1,5,10]
        }
    }
}



face_cascade = cv2.CascadeClassifier('./opencv/haarcascade/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./opencv/haarcascade/haarcascade_eye.xml')


path_to_data = './dataset'
path_to_cr_data = './dataset/cropped'

img_dirs = append_img_path_to_list(path_to_data)
create_cropped(path_to_cr_data)


cropped_img_dir=[]
celebrity_file_names_dict = {}

for img_dir in img_dirs:
    count =1
    celebrity_name = img_dir.split('\\')[-1]
    if celebrity_name=='cropped':
        continue

    celebrity_file_names_dict[celebrity_name] = []
    for entry in os.scandir(img_dir):
        # print('entry2')
        roi_color = cropped_img_if_2_eyes(entry.path)
        if roi_color is not None:
            cropped_folder = path_to_cr_data +'/'+ celebrity_name
            if not os.path.exists(cropped_folder):
                os.makedirs(cropped_folder)
                cropped_img_dir.append(cropped_folder)
                print("generate cropped images.....",cropped_folder)

            cropped_file_name = celebrity_name + str(count) + '.png'
            cropped_file_path = cropped_folder + '/' + cropped_file_name

            cv2.imwrite(cropped_file_path,roi_color)
            celebrity_file_names_dict[celebrity_name].append(cropped_file_path)
            count+=1

class_celb_dict= {}
count=0
for celebrity_name in celebrity_file_names_dict.keys():
    class_celb_dict[celebrity_name] = count
    count = count +1

X=[]
y=[]

for celebrity_name , training_files in celebrity_file_names_dict.items():
    for training_image in training_files:
        img = cv2.imread(training_image)
        if img is None:
            continue
        scalled_raw_img = cv2.resize(img, (32,32))
        img_har = w2d(img,'db1',5)
        scalled_har_img = cv2.resize(img_har, (32,32))
        combined_img = np.vstack((scalled_raw_img.reshape(32*32*3,1), scalled_har_img.reshape(32*32,1)))
        X.append(combined_img)
        y.append(class_celb_dict[celebrity_name])
X = np.array(X).reshape(len(X),4096).astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# pipe = Pipeline([('scaler',StandardScaler()),('svc',SVC(kernel='rbf',C=10))])
# pipe.fit(X_train,y_train)
scores = []
best_estimator ={}
for algo, mp in model_params.items():
    pipe = make_pipeline(StandardScaler(), mp['model'])
    clf = GridSearchCV(pipe, mp['params'], cv=5, return_train_score=False)
    clf.fit(X_train, y_train)
    scores.append({
        'model': algo,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    best_estimator[algo] = clf.best_estimator_
scores = pd.DataFrame(scores,columns=['model','best_score','best_params'])
print(scores)

best_clf = best_estimator['svm']
joblib.dump(best_clf,'saved_model.pkl')

with open("class_dictionary.json","w") as f:
    f.write(json.dumps(class_celb_dict))

