import numpy as np
import pandas as pd
from flask import Flask, request, render_template, Response
from sklearn import preprocessing
import cv2
import mediapipe as mp
import pickle
from threading import Thread
import time

from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired

app = Flask(__name__)

#%% Import ML models and define features to be used for each model

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

# HYPER PARAMETERS
n_skip = 2 # Skip this many frames of the video between each prediction
detection_threshold = 0.5 # Minimum confidence needed to classify the user as in position
down_threshold = 0.2 # If your depth is below this, you went "all the way down"
top_threshold = 0.7 # If your depth is above this, you're at the "top"
resistance = 4 # Number of frames in a row needed to confirm the user is in push-up position

#%% Mediapipe functions

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

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

#%% Analyze Frames        

def analyze_frames(vid):

    # INITIALIZE VARIABLES USED TO TRACK PUSH-UPS
    skip_frame_ix = n_skip
    detect_streak = 0 # Tracks whether we've reached 'resistance' # of frames
    current_position = 'up'
    depth_pred = 1 # this is needed to calculate the very first depth pred
    deep_pushups = 0
    partial_pushups = 0
    made_it_down = False
    
    pushup_preds = []    
    
    # READ FRAME
    while vid.isOpened():
        
        success, frame = vid.read()
        
        if not success:
            break
        
        # DETECT POSE
        if skip_frame_ix == n_skip:

            skip_frame_ix = 0
        
            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:    
                
                frame, results = mediapipe_detection(frame, pose)
                
                # EXTRACT POSE COORDINATES
                current_pose = pd.DataFrame(extract_keypoints_no_vis(results)).transpose()
                current_pose.columns = [f'lm_{ix}_{dim}' for ix in range(33) for dim in ['x', 'y', 'z']]
                
                if current_pose.sum(axis=1)[0] != 0:
                    
                    # DETECT PUSH-UP
                    pushup_pred = pushup_detector.predict(current_pose)
                    
                    form_preds = list(butt_classifier.predict(current_pose.loc[:, form_cols])[0])
                    
                    head_pred = round(head_classifier.predict(current_pose.loc[:, head_cols])[0], 0)
                    
                    depth_pred = round((depth_regressor.predict(current_pose.loc[:, depth_cols])[0] + depth_pred) / 2, 3)
                    
                    prev_position = current_position
                    
                    # If pushup position is detected
                    if pushup_pred > detection_threshold:
                        
                        # Make sure it's not a false positive
                        if detect_streak == resistance:
                        
                            # Track the count of good and bad pushups
                            if depth_pred > top_threshold:
                                current_position = 'up'
                                # Check if a push-up has been completed
                                if current_position != prev_position:
                                    if made_it_down:
                                        deep_pushups += 1
                                    else:
                                        partial_pushups += 1
                                    made_it_down = False
                                    
                            elif depth_pred > down_threshold:
                                current_position = 'middle'
                            
                            else:
                                current_position = 'down'
                                made_it_down = True    
                            
                            # Organize all predictions
                            current_preds = [deep_pushups,
                                             partial_pushups,
                                             depth_pred > top_threshold,
                                             form_preds[0],
                                             form_preds[1],
                                             head_pred]
                            pushup_preds.append(current_preds)
                            
                        else:
                            detect_streak += 1
                    else:
                        detect_streak = 0
        else:
            skip_frame_ix += 1
        
    pred_cols = ['count_deep_pushups', 'count_partial_pushups', 'at_top', 'bottom_high', 'bottom_low', 'head_low']
    pred_df = pd.DataFrame(pushup_preds, columns=pred_cols)
    
    return pred_df

def generate_report(preds):
    
    if len(preds) == 0:
        return ['No push-ups detected in video.']
    
    report = []
    
    count_partial_pushups = preds['count_partial_pushups'].max()
    total_pushups = preds['count_deep_pushups'].max() + count_partial_pushups
    
    # doing_pushup = preds[preds['at_top'] == False]
    form_cols = ['bottom_high', 'bottom_low', 'head_low']
    
    count_bad_form = (preds
                      .groupby(['count_deep_pushups', 'count_partial_pushups'])
                      [form_cols]
                      .max()
                      .sum()
                      .astype(int)
                      )
    
    count_bottom_high = count_bad_form['bottom_high']
    count_bottom_low = count_bad_form['bottom_low']
    count_head_low = count_bad_form['head_low']
    
    report.append(f'You completed {total_pushups} push-ups.')
    
    if count_partial_pushups > 0:
        report.append(f'You didn\'t go deep enough in {count_partial_pushups} of them.')
    else:
        report.append('You went deep enough in every one of them.')
        
    if count_bottom_high > 0:
        report.append(f'Your bottom was too high during {count_bottom_high} of them.')
    else:
        report.append('Your bottom was never too high.')
        
    if count_bottom_low > 0:
        report.append(f'Your bottom was too low during {count_bottom_low} of them.')
    else:
        report.append('Your bottom was never too low.')
        
    if count_head_low > 0:
        report.append(f'Your head was too low during {count_head_low} of them.')
    else:
        report.append('Your head was never too low.')
        
    report.append('Good workout!')
        
    return report

#%% Run webpage

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/video_choice/', methods=['POST'])
def video_choice():
    
    if request.form.get('demo') == 'Use Demo Video':
        vid = cv2.VideoCapture('vid.mp4')
    else:
        file = request.files["file"]
        filename = secure_filename(file.filename)
        file.save(os.path.join('upload_folder', filename))
        vid = cv2.VideoCapture(os.path.join('upload_folder', filename))
        
    pushup_preds = analyze_frames(vid)

    report = generate_report(pushup_preds)
    
    return render_template('report.html', status_message='Analysis complete.', report=report)

@app.route('/how_i_work/')
def how_i_work():
    return render_template('how_i_work.html')

if __name__ == '__main__':
    app.run(debug=True)