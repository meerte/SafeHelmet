import tkinter as tk
from tkinter import filedialog
import cv2 as cv
import os
import time
import numpy as np
import math
from natsort import natsorted, ns
from ultralytics import YOLO
import math
from datetime import date
from datetime import datetime

# This is the python file for creating exe for out project. However, there are some local machine path problems. So, currently it does not work properly.
# pyinstaller ad.py
class_names = {0: 'Helmet', 1: 'Motorcycle'}
model_path = r'C:\Users\mserl\Desktop\.VS Code Docs\HD2_20.12.23\weights\best.pt'
model = YOLO(model_path) # './weights/best.pt'
def h_and_m_counter(array):
    counter = []
    for element in array:
        helmet = (element == 0).sum()
        motor = (element == 1).sum()
        counter.append([helmet, motor])
    return np.array(counter)

def flag_counter(array):
    flags = []
    for element in array:
        if element[1] > element[0]:
            flags.append(False)
        else:
            flags.append(True)
    return flags

def is_violated(array):
    violation_counter = 0
    for element in array:
        if element == False:
            violation_counter += 1
    if violation_counter / len(array) * 100 > 50:
        return False
    return True

def add_frame_number(video_path, output_path, arr, flag_arr):
    cap = cv.VideoCapture(video_path)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS)

    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(output_path, fourcc, fps, (width, height))
    helmet = [row[0] for row in arr]
    motor = [row[1] for row in arr]

    counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if counter != arr.size:
            issue = True 
            if helmet[counter] == motor[counter] or helmet[counter] > motor[counter]:
                issue = False
            cv.putText(frame, f'Helmet: {helmet[counter]} | Motor: {motor[counter]}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv.putText(frame, f'Violation: {issue}', (10, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            if flag_arr[counter]:
                cv.circle(frame, (width - 50, 50), 20, (0, 0, 255), -1)  # Kırmızı yuvarlak
            else:
                cv.circle(frame, (width - 50, 50), 20, (0, 255, 0), -1)  # Yeşil yuvarlak
            counter += 1
        out.write(frame)
        cv.imshow('Orijinal Frame', frame)
        if cv.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv.destroyAllWindows()
   

def detection():
    base_path = r'C:\Users\mserl\Desktop\.VS Code Docs\HD2_20.12.23\dist\ad\runs' #'./test_video/inputs_from_camera/'
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    base_path = r'C:\Users\mserl\Desktop\.VS Code Docs\HD2_20.12.23\dist\ad\runs\detect'
    if not os.path.exists(base_path):
        os.mkdir(base_path)    
    cam = cv.VideoCapture(0)
    if not cam.isOpened():
        print("error opening camera")
        exit()   
    for i in range(2):
        today = datetime.now()
        str_time = str(today)
        str_time = str_time.replace(':','.')
        str_time = str_time.replace('-','_')
        today = str_time
        detected_array = []
        cam = cv.VideoCapture(0)
        cc = cv.VideoWriter_fourcc(*'XVID')
        #tmp = f'{today}_output{i}.mp4'
        filename = f'{base_path}\{today}_output{i}.mp4'#base_path +  tmp
        file = cv.VideoWriter(filename, cc, 10.0, (640, 480))
        starting_time = time.time()   
        while True:
            ret, frame = cam.read()
            if not ret:
                print("error in retrieving frame")
                break
            cv.imshow('frame', frame)
            file.write(frame)
            elapsed_time = time.time() - starting_time
            if math.ceil(elapsed_time) >= 5:
                break
            if cv.waitKey(1) == ord('q'):
                break    
        cam.release()
        file.release()      
        cv.destroyAllWindows() 
        results = model(source=filename, stream=True, save= True, conf=0.4)  # generator of Results objects
        for r in results:
            boxes = r.boxes  # Boxes object for bbox outputs
            masks = r.masks  # Masks object for segment masks outputs
            probs = r.probs  # Class probabilities for classification outputs
            tmp = np.array(r.boxes.cls.cpu())
            detected_array.append(tmp)
        helmet_and_motor_counter = h_and_m_counter(detected_array)
        flag_for_violation = flag_counter(helmet_and_motor_counter)
        violation_result = is_violated(flag_for_violation)
        detected_path = r'C:\Users\mserl\Desktop\.VS Code Docs\HD2_20.12.23\dist\ad\runs\detect' #'./runs/detect/'
        dir_list = os.listdir(detected_path)
        dir_list = natsorted(dir_list, alg=ns.PATH)
        if len(dir_list) == 1:
            chosen_dir =f'{detected_path}\{dir_list[-1]}'
        else:
            chosen_dir =f'{detected_path}\{dir_list[-2]}'     
        save_dir = f'{chosen_dir}'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        today = datetime.now()
        str_time = str(today)
        str_time = str_time.replace(':','.')
        str_time = str_time.replace('-','_')
        today = str_time
        save_dir_vid =  f'{detected_path}\{today}_output.mp4'    #f'{save_dir}/{today}_output.mp4'    
        add_frame_number(save_dir, save_dir_vid, helmet_and_motor_counter, flag_for_violation) # f'./runs/detect/output{i}.mp4'
        print(f'Video {i}, Result: {violation_result}')
        print(f'sleeping {i}')
        time.sleep(2)

def detection_w_root(base_path):

    detected_array = []
    split_path = base_path.split('\\')
    video_name = split_path[-1]
    x = video_name.split('.')
    video_name = x[0] +'.avi'
    results = model(source=base_path, stream=True, save= True, conf=0.4)  # generator of Results objects
    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs
        probs = r.probs  # Class probabilities for classification outputs
        tmp = np.array(r.boxes.cls.cpu())
        detected_array.append(tmp)
    helmet_and_motor_counter = h_and_m_counter(detected_array)
    flag_for_violation = flag_counter(helmet_and_motor_counter)
    detected_path = r'C:\Users\mserl\Desktop\.VS Code Docs\HD2_20.12.23\dist\ad\runs\detect' #'./runs/detect/'
    if not os.path.exists(r'C:\Users\mserl\Desktop\.VS Code Docs\HD2_20.12.23\dist\ad\runs'):
        os.mkdir(r'C:\Users\mserl\Desktop\.VS Code Docs\HD2_20.12.23\dist\ad\runs')
    if not os.path.exists(detected_path):
        os.mkdir(detected_path)    
    dir_list = os.listdir(detected_path)
    dir_list = natsorted(dir_list, alg=ns.PATH)
    if len(dir_list) == 1:
        chosen_dir =f'{detected_path}\{dir_list[-1]}'
    else:
        chosen_dir =f'{detected_path}\{dir_list[-2]}'    
    chosen_dir_vids = os.listdir(chosen_dir)
    chosen_dir_vids = natsorted(chosen_dir_vids, alg=ns.PATH)
    chosen_vid = f'{chosen_dir}/{video_name}'
    #if len(chosen_dir_vids) == 1:
    #    chosen_vid = f'{chosen_dir}/{chosen_dir_vids[-1]}'
    #else:
    #    chosen_vid = f'{chosen_dir}/{chosen_dir_vids[-2]}'    
    save_dir = f'{chosen_dir}/result'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    today = datetime.now()
    str_time = str(today)
    str_time = str_time.replace(':','.')
    str_time = str_time.replace('-','_')
    today = str_time
    save_dir_vid = f'{save_dir}/{today}_output.mp4'    
    add_frame_number(chosen_vid, save_dir_vid, helmet_and_motor_counter, flag_for_violation) # f'./runs/detect/output{i}.mp4'

  
def access_camera():
    detection()

def upload_video():

    root.filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                               filetypes=(("mp4 files", "*.mp4"), ("all files", "*.*")))
    print("Selected:", root.filename)
    detection_w_root(root.filename)

def real_time_detection():
    #base_path = r'C:\Users\mserl\Desktop\.VS Code Docs\HD2_20.12.23\test_video\inputs_from_camera' #'./test_video/inputs_from_camera/'
    #if not os.path.exists(base_path):
    #    os.mkdir(base_path)  
    cam = cv.VideoCapture(0)

    #cc = cv.VideoWriter_fourcc(*'XVID')
    #filename = base_path +  'output.mp4'
    #file = cv.VideoWriter(filename, cc, 5.0, (640, 480))
    if not cam.isOpened():
        print("error opening camera")
        exit()
   
    class_names = ['Helmet', 'motorcycle']

    #model = YOLO('./weights/best.pt')
    total = 0
    motor_counter = 0
    helmet_counter = 0
    while True:

        ret, frame = cam.read()

        if not ret:
            print("error in retrieving frame")
            break

        results = model(source=frame, stream=True, conf=0.4)
        for r in results:
            boxes = r.boxes
        
            for box in boxes:
            # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
                cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
                confidence = math.ceil((box.conf[0]*100))/100
                print("Confidence --->",confidence)

            # class name
                cls = int(box.cls[0])   
                print("Class name -->", class_names[cls])
                if class_names[cls] == 'Helmet':
                    helmet_counter +=1
                if class_names[cls] == 'Motorcycle':
                    motor_counter += 1
        
                   
            # object details
                org = [x1, y1]
                font = cv.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                cv.putText(frame, class_names[cls], org, font, fontScale, color, thickness)
        total += 1
        cv.imshow('frame', frame)
        #file.write(frame)
        if cv.waitKey(1) == ord('q'):
            break

    cam.release()
    #file.release()
    cv.destroyAllWindows()
    print(f'Total: {total}')
    print(f'HELMET: {helmet_counter}') 
    print(f'MOTOR: {motor_counter}') 


root = tk.Tk()

root.title("Smart Detection App")


welcome_label = tk.Label(root, text="Welcome to Smart Detection App", font=("Helvetica", 16))
welcome_label.pack()

camera_button = tk.Button(root, text="Access Camera", command=access_camera)
camera_button.pack()

upload_button = tk.Button(root, text="Upload Video", command=upload_video)
upload_button.pack()
upload_button = tk.Button(root, text="Real Time Detection", command=real_time_detection)
upload_button.pack()

root.mainloop()