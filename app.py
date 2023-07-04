# from flask import Flask, request, Response, render_template
# from werkzeug.utils import secure_filename
# import cv2
# from db import db_init, db
# from models import Img

# app = Flask(__name__)
# # SQLAlchemy config. Read more: https://flask-sqlalchemy.palletsprojects.com/en/2.x/
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///img.db'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# db_init(app)


# @app.route('/')
# def home():
#     return render_template('index.html')


# @app.route('/index', methods=['POST','GET'])
# def index():
#     pic = request.files['pic']
#     if not pic:
#         return render_template('index.html',text="No pic uploaded!")
#         # return 'No pic uploaded!', 400

#     filename = secure_filename(pic.filename)
#     mimetype = pic.mimetype
#     if not filename or not mimetype:
#         # return 'Bad upload!', 400
#         return render_template('index.html',text="bad upload!")

#     img = Img(img=pic.read(), name=filename, mimetype=mimetype)
#     db.session.add(img)
#     db.session.commit()

#     # return 'Img Uploaded!', 200
#     return render_template('index.html',text="Img uploaded!")

# @app.route('/olduser', methods=['POST','GET'])
# def olduser():
#     return render_template('olduser.html')


# @app.route('/<int:id>')
# def get_img(id):
#     img = Img.query.filter_by(id=id).first()
#     if not img:
#         # return 'Img Not Found!', 404
#         return render_template('index.html',text="Not found!")

#     # image=Response(img.img, mimetype=img.mimetype)
#     # print(type(image))
#     # cv2.imshow(image)
#     # return render_template('index.html',img_url=image)
#     return Response(img.img, mimetype=img.mimetype)
#     # return render_template('newuser.html')

# @app.route('/newuser')
# def newuser():
#     return render_template('newuser.html')

# if __name__ == "__main__":
#     app.run(debug=True)





from flask import Flask, request, Response, render_template,flash
from werkzeug.utils import secure_filename
# from db import db_init, db
# from models import Img
from PIL import Image
import pyautogui
import os, sys
import cv2
import shutil
import numpy as np
import math
import pandas as pd
import warnings
import time
import face_recognition
from progressbar import ProgressBar
from pathlib import Path
import joblib
from sklearn.decomposition import PCA
from sklearn import svm, datasets
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier  
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from datetime import datetime
from flask import Flask, flash, request, redirect, url_for, render_template, Response
from werkzeug.utils import secure_filename
from sklearn.preprocessing import StandardScaler as sc
pbar = ProgressBar()
app = Flask(__name__)
# SQLAlchemy config. Read more: https://flask-sqlalchemy.palletsprojects.com/en/2.x/
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///img.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# db_init(app)

x_train = []
y_train = []
X = []
Y = []
name_grp = []
data_fr = []
G = []
pca_mean = 0
msg = ""
data_changed = False
reloaded = False
pred_name = ""
pred_prob = ""


def encode_faces():
    global name_grp,X,Y,data_fr,pca_mean          
    faces = {}
    name_grp = set()
    for image in pbar(os.listdir('faces_l')):
        person_name = image.split("_")[0]            
        # face_image = face_recognition.load_image_file(f"faces_l\{image}",mode="RGB")
        # face_image = face_image.flatten()
        # print(face_image.shape)
        # faces[image] = cv2.imdecode(face_image.flatten(), cv2.IMREAD_COLOR)
        faces[image] = cv2.imdecode(np.fromfile(f"faces_l\{image}", dtype=np.uint8), cv2.IMREAD_COLOR)
        X.append(faces[image].flatten())
        Y.append(person_name)
        name_grp.add(person_name)
    # print(len(X),len(X[0]))
    # X = np.array(X)
    pca = PCA(n_components=10).fit(X)
    X = pca.transform(X)
    joblib.dump(pca,'models_l\\PCA.pkl')
#     n_components = 128
#     eigenfaces = pca.components_[:n_components]
#     X = eigenfaces@(X - pca.mean_).T
#     X = X.T
    name_grp = list(name_grp) 
    joblib.dump(name_grp,'models_l\\names.pkl')
    print("Data fetched Successfully..")  
    data_fr = pd.DataFrame(X)
    temp = []
    for k in Y:
        temp.append(name_grp.index(k)+1)
    Y = temp
    data_fr['y'] = Y
    new_data = data_fr.copy()
    new_data = new_data.sample(frac = 1)
    filepath = Path('data_l1.csv')  
    # filepath.parent.mkdir(parents=True, exist_ok=True)  
    new_data.to_csv(filepath,index=False,header=False) 
    print("Data Saved..")
    # print(data_fr)
    # print(y_train)

def train():
    global data_fr,X,Y
    data_fr = pd.read_csv("data_l1.csv")
    Y = data_fr.iloc[:,-1]
    X = data_fr.iloc[:,:-1]
    print(X.shape)
    print(Y.shape)
    knn = KNeighborsClassifier()
    knn_params = dict(n_neighbors=list(range(1, 5)))
    grid_knn = GridSearchCV(knn, knn_params, cv=3, scoring='accuracy', return_train_score=False)
    grid_knn.fit(X, Y)
    joblib.dump(grid_knn,'models_l\\KNN.pkl')
    print("KNN trained")        

    clf = svm.SVC(probability=True)

    clf_params={
        "C":[0.001,0.01],
        "gamma":[0.001,0.01],
        "kernel":["rbf"]
    }
    grid_svc = GridSearchCV(clf,clf_params,refit=True,verbose=3)
    grid_svc.fit(X,Y)
#         SVC_params = grid.best_params_()
    joblib.dump(grid_svc,'models_l\\SVC.pkl')
    print("Support Vector Classifier Trained")


    
    RF_classifier= RandomForestClassifier()  
    RF_classifier.fit(X,Y)  
    RF_params = { 
        'n_estimators': [5, 20],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth' : [4,5,6,7,8],
        'criterion' :['gini', 'entropy']
    }
    grid_RF = GridSearchCV(RF_classifier, RF_params, cv= 3)
    joblib.dump(grid_RF,'models_l\\RF.pkl')
    print("Random Forest Classifier Trained")

#         classifier.

    
    nn = MLPClassifier(hidden_layer_sizes=(50,40,20),activation="relu",solver="adam",learning_rate="constant",learning_rate_init=0.001,max_iter=1000)
    nn.fit(X,Y)
#         MLP_params = nn.get_params()      
    joblib.dump(nn,'models_l\\MPL.pkl')
    print("MultiLayer Protocol Trained")

    

    ensbl = VotingClassifier(estimators = [('knn', knn), ('svc', grid_svc),('rf',RF_classifier),('ANN',nn)],voting='soft')
    ensbl.fit(X,Y)        
    joblib.dump(ensbl,'models_l\\ENSBL.pkl')
    print("Ensemble Trained")
    
    epoch = 5
    n_dimensions = data_fr.shape[1]-1
    Acc = []
    # n_datapoints = result.shape[0]

    for e in range(epoch):

        train_data = data_fr.sample(frac=0.80)

        train_data = train_data.values
        train_x = train_data[:,:-1]
        train_y = train_data[:,-1].ravel()

        test_data = data_fr.sample(frac=0.20)
        test_data = test_data.values
        test_x = test_data[:,:-1]
        test_y = test_data[:,-1].ravel()

        ensbl.fit(train_x,train_y)
        y_pred = ensbl.predict(test_x)
        Acc.append(accuracy_score(test_y,y_pred))
        print(Acc[-1])


    print("accuracy_score : {:.5f}".format(np.sum(Acc)/epoch))

@app.route('/')
def hello_world():
    if(data_changed):
        # encode_faces()
        train()
    # return render_template('index.html')
    return render_template('index.html')

@app.route('/olduser')
def olduser():
    return render_template('olduser.html',msg=msg)

@app.route('/olduser/test')
def olduser_test():
    global msg,page_no,reloaded,pred_name,pred_prob
    # print("Hello")
    if(not reloaded):
        time.sleep(2)
        reloaded = True
        return redirect(url_for('olduser_test'))
    else:
        reloaded = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        name_grp = joblib.load('models_l\\names.pkl')
        test_faces = []
        for img in os.listdir('temp_img'):
            if(img.startswith("Test")):
                test_faces.append(img)
        # print(len(test_faces))
        print(test_faces)
        if(len(test_faces)>0):
            if(len(test_faces)==1):
                test_img = test_faces[-1]
            else:
                test_img = test_faces[-2]
            print(test_img)
            x_test = cv2.imdecode(np.fromfile(f"temp_img/{test_img}", dtype=np.uint8), cv2.IMREAD_COLOR)
            print(x_test.shape,x_test.flatten().shape)
            x_test = [x_test.flatten()]
            pca = joblib.load('models_l\\PCA.pkl')            
            x_test = pca.transform(x_test)

            ensbl = joblib.load('models_l\\ENSBL.pkl')

            y_pred = ensbl.predict(x_test)
            prob = ensbl.predict_proba(x_test)[0]
            # print(y_pred)            
            known_persons = 0
            # msg += " ".join(prob) +  " ".join(name_grp)
            print(prob)
            print(name_grp)
            for loc in y_pred:
                if((prob[int(loc)-1])>0.30):
                    name=name_grp[int(loc)-1]
                    known_persons+=1
                    msg = name + str(prob[int(loc)-1])
                    pred_name = name
                    pred_prob= str(prob[int(loc)-1])
                    # print(name)
            if(known_persons==0):
                msg = "No Known person in the frame"
                                
            print(msg)
            return redirect(url_for('olduser'))
        else:
            print("No image for testing")
        
            
        
        

@app.route('/newuser/<string:person_name>')
def newuser_register(person_name):
    global data_changed
    print(person_name)
    # print("Hello")
    # new_data = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        files = os.listdir('temp_img')
        img_count=1
        for img in files:
            if(img.startswith('Sample')):
                # new_data.append(cv2.imdecode(np.fromfile(f'temp_img\{img}', dtype=np.uint8), cv2.IMREAD_COLOR.flatten()))
                shutil.move(f'temp_img\{img}',f'faces_l\{person_name}_{img_count}.jpg')
                data_changed = True
                img_count+=1
        print("Done..")        
        return redirect(url_for('hello_world'))
        # return render_template('olduser.html',msg=msg)


@app.route('/Redirecting', methods=['POST','GET'])
def Redirecting():
    return render_template('Redirect.html')

# @app.route('/upload', methods=['POST','GET'])
# def upload():
#     pic = request.files['pic']    
#     # print("Hello")
#     img = Image.open(pic.read())       


#     if not pic:
#         # render_template('test.html')
#         return render_template('newuser.html',text="No pic uploaded!")
#         # return 'No pic uploaded!', 400

#     filename = secure_filename(pic.filename)
#     mimetype = pic.mimetype
#     if not filename or not mimetype:
#         return render_template('Redirect.html',text="Bad upload!", redirectMessage="Redirecting to New user page...")
#         # return 'Bad upload!', 400

#     img = Img(img=pic.read(), name=filename, mimetype=mimetype)
#     db.session.add(img)
#     db.session.commit()

#     # return 'Img Uploaded!', 200
#     # return render_template('index.html',text="Image uploaded!")
#     return render_template('Redirect.html',text="Image uploaded successfully!", redirectMessage="Redirecting to Old user page...")


# @app.route('/<int:id>')
# def get_img(id):
#     img = Img.query.filter_by(id=id).first()
#     if not img:
#         return 'Img Not Found!', 404

#     return Response(img.img, mimetype=img.mimetype)

@app.route('/newuser', methods=['POST','GET'])
def newuser():
    return render_template('newuser.html')


if __name__ == "__main__":
    app.run(debug=False)
