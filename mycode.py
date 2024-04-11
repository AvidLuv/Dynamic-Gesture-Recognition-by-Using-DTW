import os
import mediapipe as mp
import cv2
import numpy as np
import time
import scipy.spatial.distance as euclidean
from fastdtw import fastdtw
import glob

class HandKeypoints:
    """
    get the hand keypoints using mediapipe
    """
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.6, max_num_hands=1)
        self.last_keypoints = None 

    def getFramePose(self, image):
        start_time = time.time()
        img_cvt = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_cvt)

        if not results.multi_hand_landmarks:
            
            if self.last_keypoints is None:
                return [], 0
           
            return self.last_keypoints, time.time() - start_time

        keypoints = []
        for hand_landmarks in results.multi_hand_landmarks:
            keypoints = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
            keypoints = [keypoints[i] for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]
            self.last_keypoints = keypoints  
            break 

        time_consumed = time.time() - start_time

        return keypoints, time_consumed
    
    def getVectorsAngle(self, v1, v2):
        if np.array_equal(v1, v2):
            return 1
        dot_product = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        #if norm == 0:
           # return 0
        return np.arccos(dot_product/norm)

    def getFrameFeat(self, frame):
        """
        get the frame features
        """
        mediapipe_time = 0

        mediapipe_start = time.time()
        keypoints_with_scores, _= self.getFramePose(frame)
        mediapipe_consumed = time.time() - mediapipe_start

        if not keypoints_with_scores:
            return[]
        keypoints_list = [[landmark[0], landmark[1]] for landmark in keypoints_with_scores]
        #construct the connections between keypoints

    
        conns = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
                 (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
                 #21 keypoints vectors

        
        start_time = time.time()  #start recording time
        #transform to numpy 
        keypoints_list = np.asarray(keypoints_list)

        vector_list = list(map(lambda conn: keypoints_list[conn[1]] - keypoints_list[conn[0]], conns))

        #calculatet the angles between vectors
        
        angle_list = []
        for vector_a in vector_list:
            for vector_b in vector_list:
                angle = self.getVectorsAngle(vector_a, vector_b)
                angle_list.append(angle)

        time_consumed = time.time() - start_time

        return angle_list, time_consumed, mediapipe_consumed
    
    def getVideoFeat(self, videoFile):
        feat_time = 0
        mediapipe_time = 0
        cap = cv2.VideoCapture(videoFile)
        video_feat = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_feat, feat_duration, mediapipe_duration = self.getFrameFeat(frame)
            feat_time += feat_duration
            mediapipe_time += mediapipe_duration
            video_feat.append(frame_feat)
        cap.release()
        return video_feat, feat_time, mediapipe_time
    
    def getTrainingFeats(self):
        """
        calculate the featrues of the dataset
        """
        saveFileName = 'C:/Users/Hank Yue/Desktop/data/trainingData.npz'

        if os.path.exists(saveFileName):
            with open(saveFileName, 'rb') as f:
                return np.load(f, allow_pickle='TRUE' )

        filename = 'C:/Users/Hank Yue/Desktop/data/action_train/*/*.mp4'
        file_list = glob.glob(filename)

        training_feat = []

        for file in file_list:
            unified_file_path = file.replace('\\', '/')
            print('starts', unified_file_path)
            video_feat, _, _ = self.getVideoFeat(unified_file_path)
            print(len(video_feat))
            action_name = unified_file_path.split('/')[6]
            print(action_name)
            training_feat.append([action_name, video_feat])

        training_feat = np.array(training_feat, dtype=object)
        
        with open(saveFileName, 'wb') as f:
            np.save(f, training_feat)

        return training_feat
 
class Pose_recgnition:

    def __init__(self):
        print('start')
        self.cap = cv2.VideoCapture(0)
        # set the resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        # confirm the resolution
        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Resolution: {self.frame_w}x{self.frame_h}")  
        self.video_feat = HandKeypoints()

        self.training_feat = self.video_feat.getTrainingFeats()

        #set the threshold range
        self.batch_size = 6
        self.threshold = 0.5
    
    def calSimularity(self, sequen_feats):
        """
        calculate the dtw distance and return back the recognize gesture
        """
        #compare the live features with the dataset's feature
        start_time = time.time()
        dist_list =[]
        for action_name, video_feat in self.training_feat:
            distance, path = fastdtw(sequen_feats, video_feat)
            dist_list.append([action_name, distance])

        #sort the output from the shortest distance
        dist_list = np.array(dist_list, dtype=object)
        dist_list = dist_list[dist_list[:,1].argsort()][:self.batch_size]

        #get the first gesture's name
        first_key = dist_list[0][0]

        max_num = np.count_nonzero(dist_list[:, 0] == first_key)

        time_consumed = time.time() - start_time

        #check the threshold state
        if max_num / self.batch_size >= self.threshold:
            return first_key, time_consumed
        else:
            return 'unknown', time_consumed

    def recognize(self):
        cap = self.cap

        programe_time = 0

        # if record
        record_status = False
        # record the frames
        frame_count = 0
        # live features
        sequen_feat = []

        action_name = ''

        # trigger time
        triger_time = time.time()
        
        
        last_frame_time = time.time()

        
        action_start_time = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # calculate the FPS
            current_time = time.time()
            time_diff = current_time - last_frame_time
            fps = 1 / time_diff if time_diff > 0 else 0
            last_frame_time = current_time
            keypoints, _= self.video_feat.getFramePose(frame)
            keypoints_mark = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

            #draw the keypoints
            for i, kp in enumerate(keypoints):
                if i in keypoints_mark:
                    x, y = int(kp[0] * self.frame_w), int(kp[1] * self.frame_h)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            if record_status:
                if time.time() - triger_time > 2:
                    cv2.circle(frame, (40, 40), 5, (0, 255, 0), -1)

                    if frame_count == 0:
                        mediapipe_time = 0
                        fastdtw_time = 0
                        feat_time = 0
                        # start gesture recognition
                        action_start_time = time.time()

                    if frame_count < 35:
                        fake, mediapipe_duration = self.video_feat.getFramePose(frame)
                        mediapipe_time += mediapipe_duration
                        angle_list, feat_duration, _ = self.video_feat.getFrameFeat(frame)
                        feat_time += feat_duration
                        sequen_feat.append(angle_list)
                        frame_count += 1
                    else:
                        # finish gesture recognition
                        print('Start Recognizing')
                        action_name, fastdtw_duration = self.calSimularity(sequen_feat)
                        fastdtw_time += fastdtw_duration
                        print(action_name)

                        # calculate the average FPS
                        action_end_time = time.time()
                        action_duration = action_end_time - action_start_time
                        average_fps = frame_count / action_duration
                        print(f"Average FPS: {average_fps:.2f}")

                        if mediapipe_time + fastdtw_time + feat_time> 0:
                            mediapipe_percentage = (mediapipe_time / action_duration) * 100
                            fastdtw_percentage = (fastdtw_time / action_duration) * 100
                            feat_percentage = (feat_time / action_duration) * 100
                            print(f"MediaPipe time percentage: {mediapipe_percentage:.2f}%")
                            print(f"fastdtw time percentage: {fastdtw_percentage:.2f}%")
                            print(f"feat time percentage: {feat_percentage:.2f}%")
                            print(action_duration)
                            print(fastdtw_time)
                            print(mediapipe_time)
                            print(feat_time)


                        # reset the state
                        frame_count = 0
                        sequen_feat = []
                        triger_time = time.time()
                        record_status = False
                else:
                    cv2.circle(frame, (40, 40), 5, (0, 255, 255), -1)
            else:
                cv2.circle(frame, (40, 40), 5, (0, 0, 255), -1)

            # print the gesture recognized
            cv2.putText(frame, f'Action: {action_name}', (50, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # print the FPS
            cv2.putText(frame, f'FPS: {fps:.1f}', (frame.shape[1] - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.imshow('demo', frame)

            pressKey = cv2.waitKey(1) & 0xFF
            if pressKey == ord('q'):
                break
            elif pressKey == ord('r'):
                record_status = True
                triger_time = time.time()
                print('Start Recording')
        
    def test_video(self, root_folder):
        correct_predictions = 0
        total_videos = 0
        video_feat = []
        video_files = []

        mediapipe_time = 0
        fastdtw_time = 0
        feat_time = 0
        total_recognize_time = 0

        for root, dirs, files in os.walk(root_folder):
            for file in files:
                if file.endswith('.mp4'):
                    video_files.append(os.path.join(root, file))

        total_videos_count = len(video_files)
        for i, video_path in enumerate(video_files):

            start_time = time.time()

            # print the current process
            print(f"Processing video {i+1}/{total_videos_count}")

            video_feat, feat_duration, mediapipe_duration = self.video_feat.getVideoFeat(video_path)

            
            feat_time += feat_duration

            
            mediapipe_time += mediapipe_duration

            recognize_name, fastdtw_duration = self.calSimularity(video_feat)
            
            fastdtw_time += fastdtw_duration

            action_name = os.path.basename(os.path.dirname(video_path))

            recognize_time = time.time() - start_time

            total_recognize_time += recognize_time

            print(f"Recognized: {recognize_name}, Actual: {action_name}")

            if action_name == recognize_name:
                correct_predictions += 1
            total_videos += 1

        if total_videos > 0:
            accuracy = correct_predictions / total_videos
            print(f"Accuracy: {accuracy * 100:.2f}%")

            # calculate the percentage of each part
            mediapipe_percentage = (mediapipe_time / total_recognize_time) * 100
            fastdtw_percentage = (fastdtw_time / total_recognize_time) * 100
            feat_percentage = (feat_time / total_recognize_time) * 100

            print(f"MediaPipe time percentage: {mediapipe_percentage:.2f}%")
            print(f"fastdtw time percentage: {fastdtw_percentage:.2f}%")
            print(f"Feature extraction time percentage: {feat_percentage:.2f}%")
            print(f"Total processing time: {total_recognize_time:.2f} seconds")
        else:
            print("No videos found.")


def adjust_video_resolution(source_folder, target_folder, width=960, height=720):

    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith('.mp4'):
                source_path = os.path.join(root, file)
                
                
                relative_path = os.path.relpath(root, source_folder)
                target_dir = os.path.join(target_folder, relative_path)
                os.makedirs(target_dir, exist_ok=True)
                target_path = os.path.join(target_dir, file)
                
                
                cap = cv2.VideoCapture(source_path)
                original_fps = cap.get(cv2.CAP_PROP_FPS)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(target_path, fourcc, original_fps, (width, height))
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    resized_frame = cv2.resize(frame, (width, height))
                    out.write(resized_frame)
                
                cap.release()
                out.release()
                print(f'Processed and saved {target_path}')

pose = Pose_recgnition()
pose.recognize()
video_path = 'C:/Users/Hank Yue/Desktop/data/640x480'
output_path = 'C:/Users/Hank Yue/Desktop/data/960x720'
#adjust_video_resolution(video_path, output_path)
#pose.test_video(video_path)
