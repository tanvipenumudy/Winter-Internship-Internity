# this one is used to recognize the 
# face after training the model with 
# our data stored using knn 
import cv2 
import numpy as np 
import pandas as pd 

from npwriter import f_name 

def euc_dist(x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))
class KNN:
    def __init__(self, K=3):
        self.K = K
    def fit(self, x_train, y_train):
        self.X_train = x_train
        self.Y_train = y_train
    def predict(self, X_test):
        predictions = []
        for i in range(len(X_test)):            
            dist = np.array([euc_dist(X_test[i], x_t) for x_t in self.X_train])
            dist_sorted = dist.argsort()[:self.K]
            neigh_count = {}
            for idx in dist_sorted:
                if self.Y_train[idx] in neigh_count:
                    neigh_count[self.Y_train[idx]] += 1
                else:
                    neigh_count[self.Y_train[idx]] = 1       
            sorted_neigh_count = sorted(neigh_count.items(), key=operator.itemgetter(1), reverse=True)
            predictions.append(sorted_neigh_count[0][0])
        return predictions

# reading the data 
data = pd.read_csv(f_name).values 

# data partition 
X, Y = data[:, 1:-1], data[:, -1] 

print(X, Y) 

# Knn function calling with k = 5 
model = KNN(K = 5)

# fdtraining of model 
model.fit(X, Y) 

cap = cv2.VideoCapture(0) 

classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 

f_list = [] 

while True: 

	ret, frame = cap.read() 

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

	faces = classifier.detectMultiScale(gray, 1.5, 5) 

	X_test = [] 

	# Testing data 
	for face in faces: 
		x, y, w, h = face 
		im_face = gray[y:y + h, x:x + w] 
		im_face = cv2.resize(im_face, (100, 100)) 
		X_test.append(im_face.reshape(-1)) 

	if len(faces)>0: 
		response = model.predict(np.array(X_test)) 
		# prediction of result using knn 

		for i, face in enumerate(faces): 
			x, y, w, h = face 

			# drawing a rectangle on the detected face 
			cv2.rectangle(frame, (x, y), (x + w, y + h), 
										(255, 0, 0), 3) 

			# adding detected/predicted name for the face 
			cv2.putText(frame, response[i], (x-50, y-50), 
							cv2.FONT_HERSHEY_DUPLEX, 2, 
										(0, 255, 0), 3) 
	
	cv2.imshow("full", frame) 

	key = cv2.waitKey(1) 

	if key & 0xFF == ord("q") : 
		break

cap.release() 
cv2.destroyAllWindows() 
