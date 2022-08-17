import cv2
import os
import mediapipe as mp
import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt

import pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import f1_score

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Reshape, Input
from keras.layers import LSTM, SpatialDropout1D
from keras.layers import SimpleRNN
from keras.layers import Dropout, Embedding
from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences

# pip install mediapipe opencv-python

# os.chdir('Documents\Python Scripts\pushups-projects')

#%% mp_drawing and mp_pose

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


#%% Data Path

# these are not needed if you set directory to pushups-project
PROJECT_PATH = os.path.join('pushups-project') 

DATA_PATH = os.path.join(PROJECT_PATH, 'data')

#%% mediapipe functions

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                              )

def extract_keypoints(results):
    return np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)

def extract_keypoints_no_vis(results):
    return np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    return angle if (angle <= 180.0) else (360 - angle)

#%% BASIC VIDEO FEED

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    frame = cv2.resize(frame, (1200, 900))
    cv2.imshow('Mediapipe Feed', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()

#%% Collect image data for pushup form classification

positions = np.array(['top'])

no_samples_per_class = 12

cap = cv2.VideoCapture(0)

# Set mediapipe model 
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:    

    # Loop through positions
    for position in positions:
        
        position_data = pd.DataFrame()
        
        # Loop through number of samples per position
        for sample in range(no_samples_per_class):            

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break            

            # Read feed
            ret, frame = cap.read()
            
            frame = cv2.resize(frame, (1200, 900))

            # Make detections
            image, results = mediapipe_detection(frame, pose)
            
            # Draw landmarks
            draw_landmarks(image, results)
            
            # Apply wait logic
            if sample == 0: 
                cv2.putText(image, position, (120,200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(position, sample), (15,12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                # Show to screen
                cv2.imshow('OpenCV Feed', image)
                cv2.waitKey(5000)
            if sample == no_samples_per_class//2: 
                cv2.putText(image, 'FLIP AROUND.', (120,200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(position, sample), (15,12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                # Show to screen
                cv2.imshow('OpenCV Feed', image)
                cv2.waitKey(5000)
            else: 
                cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(position, sample), (15,12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                # Show to screen
                cv2.imshow('OpenCV Feed', image)
                cv2.waitKey(1000)
            
            position_data = pd.concat([position_data, pd.Series(extract_keypoints(results))], ignore_index=True, axis=1)
            
        # Export keypoints
        position_data = position_data.transpose().reset_index(drop=True)
        position_data.columns = [f'lm_{ix}_{dim}' for ix in range(33) for dim in ['x', 'y', 'z', 'vis']]
        npy_path = os.path.join('data', f'{position}.csv')
        position_data.to_csv(npy_path)
                    
    cap.release()
    cv2.destroyAllWindows()

#%% # Data preprocessing for NN (pushup detector)

test_size = 0.2

# Load the data

top = pd.read_csv('data/top.csv', index_col=0)
middle = pd.read_csv('data/middle.csv', index_col=0)
bottom = pd.read_csv('data/bottom.csv', index_col=0)
no_issues = pd.read_csv('data/no_issues.csv', index_col=0)
bottom_high = pd.read_csv('data/bottom_high.csv', index_col=0)
bottom_low = pd.read_csv('data/bottom_low.csv', index_col=0)
head_low = pd.read_csv('data/head_low.csv', index_col=0)
in_position = pd.read_csv('data/in_position_combined.csv', index_col=0)

in_position = pd.concat([in_position, bottom_high, bottom_low, head_low, no_issues, bottom, middle, top],
                        axis=0, ignore_index=True)
in_position = in_position.drop(columns = [f'lm_{i}_vis' for i in range(int(in_position.shape[1]/4))])

not_in_position = pd.read_csv('data/not_in_position_combined2.csv', index_col=0)
not_in_position = not_in_position.drop(columns = [f'lm_{i}_vis' for i in range(int(not_in_position.shape[1]/4))])

X = pd.concat([in_position, not_in_position], axis=0, ignore_index=True)
X = X.drop(columns= 'label')
y = pd.DataFrame([1 for _ in range(in_position.shape[0])] + [0 for _ in range(not_in_position.shape[0])])

# Remove rows where the coords are all 0's
sums = X.sum(axis=1)
bad_rows = sums.index[sums==0]
X, y = X.drop(index = bad_rows), y.drop(index = bad_rows)

# Split between train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)


#%% # Data preprocessing for NN (form classifier)

test_size = 0.2

# Coords for butt classifier
# cols = ['lm_11_x', 'lm_11_y', 'lm_11_z', 
#         'lm_12_x', 'lm_12_y', 'lm_12_z', 
#         'lm_23_x', 'lm_23_y', 'lm_23_z', 
#         'lm_24_x', 'lm_24_y', 'lm_24_z', 
#         'lm_25_x', 'lm_25_y', 'lm_25_z', 
#         'lm_26_x', 'lm_26_y', 'lm_26_z']

# Coords for head classifier
cols = ['lm_2_x', 'lm_2_y', 'lm_2_z', 
        'lm_5_x', 'lm_5_y', 'lm_5_z', 
        'lm_9_x', 'lm_9_y', 'lm_9_z', 
        'lm_10_x', 'lm_10_y', 'lm_10_z', 
        'lm_11_x', 'lm_11_y', 'lm_11_z', 
        'lm_12_x', 'lm_12_y', 'lm_12_z'] 

# Load the data

no_issues = pd.read_csv('data/no_issues.csv', index_col=0).loc[:, cols]
# bottom_high = pd.read_csv('data/bottom_high.csv', index_col=0).loc[:, cols]
# bottom_low = pd.read_csv('data/bottom_low.csv', index_col=0).loc[:, cols]
head_low = pd.read_csv('data/head_low.csv', index_col=0).loc[:, cols]

# Remove the cols with the word 'vis' in them (not helpful for the NN, or is it?)

# no_issues = no_issues.drop(columns = [f'lm_{i}_vis' for i in range(int(no_issues.shape[1]/4))])
# bottom_high = bottom_high.drop(columns = [f'lm_{i}_vis' for i in range(int(bottom_high.shape[1]/4))])
# bottom_low = bottom_low.drop(columns = [f'lm_{i}_vis' for i in range(int(bottom_low.shape[1]/4))])
# head_low = head_low.drop(columns = [f'lm_{i}_vis' for i in range(int(head_low.shape[1]/4))])

y0 = pd.DataFrame(np.zeros(no_issues.shape[0]))
y1 = pd.DataFrame(np.zeros(head_low.shape[0]))
# y0 = pd.DataFrame(np.zeros((no_issues.shape[0], 2)))
# y1 = pd.DataFrame(np.zeros((no_issues.shape[0], 2)))
# y2 = pd.DataFrame(np.zeros((no_issues.shape[0], 2)))
# y3 = pd.DataFrame(np.zeros((no_issues.shape[0], 3)))
y1[0] = 1
# y2[1] = 1
# y3[2] = 1

X = pd.concat([no_issues, head_low], axis=0, ignore_index=True)
y = pd.concat([y0, y1], axis=0, ignore_index=True)
# y.columns = ['bottom_high', 'bottom_low']

# Remove rows where the coords are all 0's
sums = X.sum(axis=1)
bad_rows = sums.index[sums==0]
X, y = X.drop(index = bad_rows), y.drop(index = bad_rows)

# Split between train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)

#%% # Data preprocessing for NN (depth regressor)

test_size = 0.2

cols = ['lm_11_x', 'lm_11_y', 'lm_11_z', 
        'lm_12_x', 'lm_12_y', 'lm_12_z', 
        'lm_13_x', 'lm_13_y', 'lm_13_z', 
        'lm_14_x', 'lm_14_y', 'lm_14_z', 
        'lm_15_x', 'lm_15_y', 'lm_15_z', 
        'lm_16_x', 'lm_16_y', 'lm_16_z']

# Load the data

top = pd.read_csv('data/top.csv', index_col=0).loc[:, cols]
middle = pd.read_csv('data/middle.csv', index_col=0).loc[:, cols]
bottom = pd.read_csv('data/bottom.csv', index_col=0).loc[:, cols]

# Remove the cols with the word 'vis' in them (not helpful for the NN, or is it?)

# top = top.drop(columns = [f'lm_{i}_vis' for i in range(int(top.shape[1]/4))])
# middle = middle.drop(columns = [f'lm_{i}_vis' for i in range(int(middle.shape[1]/4))])
# bottom = bottom.drop(columns = [f'lm_{i}_vis' for i in range(int(bottom.shape[1]/4))])

y_top = pd.DataFrame([1 for _ in range(top.shape[0])])
# y_middle = pd.DataFrame([0.5 for _ in range(middle.shape[0])])
y_bottom = pd.DataFrame([0 for _ in range(bottom.shape[0])])

X = pd.concat([top, bottom], axis=0, ignore_index=True)
y = pd.concat([y_top, y_bottom], axis=0, ignore_index=True)

# Remove rows where the coords are all 0's
sums = X.sum(axis=1)
bad_rows = sums.index[sums==0]
X, y = X.drop(index = bad_rows), y.drop(index = bad_rows)

# Split between train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=2)

#%% Train Keras NN

epochs=200

# Code from some website sentiment analysis tutorial
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=99))
model.add(Dense(100, activation='relu', input_dim=99))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC()])

model.fit(X_train, y_train, epochs = epochs, batch_size = 50, verbose=2)

print(model.summary())

#%% Train SKLearn NN

epochs = 200

# sklearn_model_regressor = MLPRegressor(hidden_layer_sizes=[100,100], activation='relu',
#                   solver='adam', alpha=0.0001, max_iter=epochs,
#                   random_state=1).fit(X_train, y_train)

sklearn_model_classifier = MLPClassifier(hidden_layer_sizes=[100,100], activation='relu',
                  solver='adam', alpha=0.0001, max_iter=epochs,
                  random_state=1).fit(X_train, y_train)

#%% Test regressor, calculate RMSE

def rmse(y, y_hat):
  return np.sqrt(np.mean((y - y_hat) ** 2))

depth_preds = sklearn_model_regressor.predict(X_test)

error = rmse(depth_preds, y_test[0])

print('\nRMSE: ', error)

# pickle.dump(sklearn_model_regressor, open('sklearn_depth_regressor_2', 'wb'))

#%% Test classifier, calculate F1 score

threshold = 0.5

# keras_preds = model.predict(X_test)
# keras_01 = (keras_preds > threshold) * 1
# keras_f1 = f1_score(y_test, keras_01, average='weighted')

# -- Pick regressor or classifier model for classification task --
# sklearn_preds = sklearn_model_regressor.predict(X_test)
sklearn_preds = sklearn_model_classifier.predict(X_test)

sklearn_01 = (sklearn_preds > threshold) * 1
sklearn_f1 = f1_score(y_test, sklearn_01, average='weighted')

# print("Keras F1: ", keras_f1, "\nSKLearn F1: ", sklearn_f1)
print('\nSKLearn F1: ', sklearn_f1)

# pickle.dump(sklearn_model_classifier, open('sklearn_form_classifier_1', 'wb'))

#%% Access webcam video and make detections    

pushup_detector = pickle.load(open('sklearn_pushup_detector_v2', 'rb'))
butt_classifier = pickle.load(open('sklearn_butt_classifier_3', 'rb'))
head_classifier = pickle.load(open('sklearn_head_classifier_1', 'rb'))
depth_regressor = pickle.load(open('sklearn_depth_regressor_9', 'rb'))

form_cols = ['lm_11_x', 'lm_11_y', 'lm_11_z', 
             'lm_12_x', 'lm_12_y', 'lm_12_z', 
             'lm_23_x', 'lm_23_y', 'lm_23_z', 
             'lm_24_x', 'lm_24_y', 'lm_24_z', 
             'lm_25_x', 'lm_25_y', 'lm_25_z', 
             'lm_26_x', 'lm_26_y', 'lm_26_z']

depth_cols = ['lm_11_x', 'lm_11_y', 'lm_11_z', 
              'lm_12_x', 'lm_12_y', 'lm_12_z', 
              'lm_13_x', 'lm_13_y', 'lm_13_z', 
              'lm_14_x', 'lm_14_y', 'lm_14_z', 
              'lm_15_x', 'lm_15_y', 'lm_15_z', 
              'lm_16_x', 'lm_16_y', 'lm_16_z']

head_cols = ['lm_2_x', 'lm_2_y', 'lm_2_z', 
             'lm_5_x', 'lm_5_y', 'lm_5_z', 
             'lm_9_x', 'lm_9_y', 'lm_9_z', 
             'lm_10_x', 'lm_10_y', 'lm_10_z', 
             'lm_11_x', 'lm_11_y', 'lm_11_z', 
             'lm_12_x', 'lm_12_y', 'lm_12_z'] 

detection_threshold = 0.5
down_threshold = 0.25 # Must have depth below this to make it a complete rep
top_threshold = 0.65
resistance = 5 # It takes 'resistance' # of frames in a row of the same prediction to change pushup detector
n_skip = 0

skip_frame_ix = n_skip
detect_streak = 0
current_position = 'up' # 1: up. 0: down
depth_pred = 1 # this is needed to calculate the very first depth pred

num_images = 0
deep_pushups = 0
partial_pushups = 0
made_it_down = False

# os.chdir('images')

# cap = cv2.VideoCapture('demo.mp4')
cap = cv2.VideoCapture('vid.mp4')
# cap = cv2.VideoCapture(0)

# out = cv2.VideoWriter('analysis.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 10, (1200, 900))

# Set mediapipe model 
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:    

    while cap.isOpened():
    
        # Read feed
        success, frame = cap.read()
        
        if not success:
            break
        
        # DETECT POSE
        if skip_frame_ix == n_skip:

            skip_frame_ix = 0
            
            # Enlarge the image
            frame = cv2.resize(frame, (1200, 900))
    
            # Make detections
            image, results = mediapipe_detection(frame, pose)
            # print(results)
            
            # Draw landmarks
            draw_landmarks(image, results)
    
            # Make prediction
            current_pose = pd.DataFrame(extract_keypoints_no_vis(results)).transpose()
            current_pose.columns = [f'lm_{ix}_{dim}' for ix in range(33) for dim in ['x', 'y', 'z']]
            
            if current_pose.sum(axis=1)[0] != 0:
                
                pushup_pred = pushup_detector.predict(current_pose)
                
                form_preds = list(butt_classifier.predict(current_pose.loc[:, form_cols])[0])
                
                head_pred = head_classifier.predict(current_pose.loc[:, head_cols])[0]
                
                depth_pred = round((depth_regressor
                                    .predict(current_pose.loc[:, depth_cols])[0] + depth_pred) / 2, 3)
                
                prev_position = current_position
                
                # If pushup position is detected
                if pushup_pred > detection_threshold:
                    
                    # Make sure it's not a false positive
                    if detect_streak == resistance:
                    
                        # Track the count of good and bad pushups
                        if depth_pred > top_threshold:
                            current_position = 'up'
                            if current_position != prev_position:
                                if made_it_down == True:
                                    deep_pushups += 1
                                else:
                                    partial_pushups += 1
                                made_it_down = False
                        elif depth_pred > down_threshold:
                            current_position = 'middle'
                        else:
                            current_position = 'down'
                            made_it_down = True    
                    
                        # cv2.imwrite(f'in_position_{num_images}.jpg', image)
                        # num_images += 1
                        # cv2.putText(image, 'YOURE IN POSITION!', (120,200), 
                                    # cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255, 0), 4, cv2.LINE_AA)
                                    
                        # cv2.putText(image, f'Form: {form_preds + [head_pred]}', (600,50), 
                        #             cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255, 0), 4, cv2.LINE_AA)
                        # cv2.putText(image, f'Depth: {depth_pred}', (600,150), 
                        #             cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255, 0), 4, cv2.LINE_AA)
                        # cv2.putText(image, f'Complete Reps: {deep_pushups}', (600,250), 
                        #             cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255, 0), 4, cv2.LINE_AA)
                        # cv2.putText(image, f'Partial Reps: {partial_pushups}', (600,350), 
                        #             cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255, 0), 4, cv2.LINE_AA)
                        
                        cv2.putText(image, f'Form: {form_preds + [head_pred]}', (50,200), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, f'Depth: {depth_pred}', (50,300), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, f'Complete Reps: {deep_pushups}', (50,400), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, f'Partial Reps: {partial_pushups}', (50,500), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255, 0), 4, cv2.LINE_AA)
                        # cv2.waitKey(4000)
                    else:
                        detect_streak += 1
                else:
                    detect_streak = 0
            
            cv2.imshow('Mediapipe Feed', image)
            # out.write(image)
            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
        else:
            skip_frame_ix += 1
        
    cap.release()
    # out.release()
    cv2.destroyAllWindows()
