import pyttsx3
#import speech_recognition as sr

engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
volume = engine.getProperty('volume')
engine.setProperty('volume', 10.0)
rate = engine.getProperty('rate')

engine.setProperty('rate', rate - 25)

def face_data():
    
    import cv2
    import os
    
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video width
    cam.set(4, 480) # set video height
    
    face_detector = cv2.CascadeClassifier(r'C:\Users\dell\Documents\OpenCV-Face-Recognition-master\OpenCV-Face-Recognition-master\FacialRecognition\haarcascade_frontalface_default.xml')
    
    # For each person, enter one numeric face id
    face_id = input('\n enter user id end press <return> ==>  ')
    
    print("\n [INFO] Initializing face capture. Look the camera and wait ...")
    # Initialize individual sampling face count
    count = 0
    
    while(True):
    
        ret, img = cam.read()
        #img = cv2.flip(img, -1) # flip video image vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
    
        for (x,y,w,h) in faces:
    
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
            count += 1
    
            # Save the captured image into the datasets folder
            cv2.imwrite(r"C:\Users\dell\Documents\OpenCV-Face-Recognition-master\OpenCV-Face-Recognition-master\FacialRecognition\dataset\User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
    
            cv2.imshow('image', img)
    
        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= 100: # Take 30 face sample and stop video
             break
    
    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()
    
    
def face_train():
    
    import cv2
    import numpy as np
    from PIL import Image
    import os
    
    # Path for face image database
    path = 'dataset'
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(r"C:\Users\dell\Documents\OpenCV-Face-Recognition-master\OpenCV-Face-Recognition-master\FacialRecognition\haarcascade_frontalface_default.xml");
    
    # function to get the images and label data
    def getImagesAndLabels(path):
    
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
        faceSamples=[]
        ids = []
    
        for imagePath in imagePaths:
    
            PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
            img_numpy = np.array(PIL_img,'uint8')
    
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)
    
            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)
    
        return faceSamples,ids
    
    print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces,ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))
    
    # Save the model into trainer/trainer.yml
    recognizer.write(r'C:\Users\dell\Documents\OpenCV-Face-Recognition-master\OpenCV-Face-Recognition-master\FacialRecognition/trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi
    
    # Print the numer of faces trained and end program
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
    
    
def face_recognize():
    
    import cv2
    import numpy as np
    import os 
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(r'C:\Users\dell\Documents\OpenCV-Face-Recognition-master\OpenCV-Face-Recognition-master\FacialRecognition\trainer\trainer.yml')
    cascadePath = r"C:\Users\dell\Documents\OpenCV-Face-Recognition-master\OpenCV-Face-Recognition-master\FacialRecognition\haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath);
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    #iniciate id counter
    id = 0
    c=0
    # names related to ids: example ==> Marcelo: id=1,  etc
    names = ['None', 'Pranav', 'Usha', 'Lavanya', 'Z', 'W'] 
 
    
    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video widht
    cam.set(4, 480) # set video height
    
    # Define min window size to be recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)
    
    while True:
    
        ret, img =cam.read()
        #img = cv2.flip(img, -1) # Flip vertically
    
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
        faces = faceCascade.detectMultiScale( 
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
           )
    
        for(x,y,w,h) in faces:
    
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
    
            # Check if confidence is less them 100 ==> "0" is perfect match 
            if (confidence < 100):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
                engine.say(id+'is infront of you')
                engine.runAndWait()
                cv2.imwrite(r"C:\Users\dell\Documents\OpenCV-Face-Recognition-master\OpenCV-Face-Recognition-master\RecognizedFaces\dataset\User." + str(id) +str(c) + '.'  + ".jpg", gray[y:y+h,x:x+w])
                
                c+=1
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
            
            cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
        
        cv2.imshow('camera',img) 
    
        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
    
    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()
    
    
    
def objects():
    
    
    import numpy as np
    import os
    import six.moves.urllib as urllib
    import sys
    import tarfile
    import tensorflow as tf
    import zipfile
    from gtts import gTTS
    import pyttsx3
    
    from collections import defaultdict
    from io import StringIO
    from matplotlib import pyplot as plt
    from PIL import Image
    
    
    from utils import label_map_util
    
    from utils import visualization_utils as vis_util
    
    
    # # Model preparation 
    # Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
    # By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
    
    # What model to download.
    MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
    
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
    
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
    
    NUM_CLASSES = 90
    
    
    # ## Download Model
    
    if not os.path.exists(MODEL_NAME + '/frozen_inference_graph.pb'):
    	print ('Downloading the model')
    	opener = urllib.request.URLopener()
    	opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    	tar_file = tarfile.open(MODEL_FILE)
    	for file in tar_file.getmembers():
    	  file_name = os.path.basename(file.name)
    	  if 'frozen_inference_graph.pb' in file_name:
    	    tar_file.extract(file, os.getcwd())
    	print ('Download complete')
    else:
    	print ('Model already exists')
    
    # ## Load a (frozen) Tensorflow model into memory.
    
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    
    
    # ## Loading label map
    # Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
    
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    
    #intializing the web camera device
    
    import cv2
    cap = cv2.VideoCapture(0)
    
    # Running the tensorflow session
    with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:
       ret = True
       while (ret):
          ret,image_np = cap.read()
          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          image_np_expanded = np.expand_dims(image_np, axis=0)
          image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
          # Each box represents a part of the image where a particular object was detected.
          boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
          # Each score represent how level of confidence for each of the objects.
          # Score is shown on the result image, together with the class label.
          scores = detection_graph.get_tensor_by_name('detection_scores:0')
          classes = detection_graph.get_tensor_by_name('detection_classes:0')
          num_detections = detection_graph.get_tensor_by_name('num_detections:0')
          # Actual detection.
          (boxes, scores, classes, num_detections) = sess.run(
              [boxes, scores, classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
          # Visualization of the results of a detection.
          vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=8)
    #      plt.figure(figsize=IMAGE_SIZE)
    #      plt.imshow(image_np)
          a = ([category_index.get(value) for index,value in enumerate(classes[0]) if scores[0,index] > 0.5])
          #a=([category_index.get('id') for index,value in enumerate(classes[0]) if scores[0,index] > 0.5])
          #x = category_index['name']
          #a=str()
          #b = a[]
          #print(b)
          engine = pyttsx3.init()
          engine.say(a)
          engine.say("is in front of you")
          
          engine.runAndWait()
          #tts = gTTS(a,lang='en',slow=False)
          #tts.save("abc.mp3")
          #os.system("abc.mp3")
          
          
          #print(scores)
          cv2.imshow('image',cv2.resize(image_np,(1280,960)))
          if cv2.waitKey(25) & 0xFF == ord('q'):
              cv2.destroyAllWindows()
              cap.release()
              break
    



    
    

def speech():
    print("Improper internet connectivity, please try manual input")
    engine.say('Improper internet connectivity, please try manual input')
    engine.runAndWait()
    print('Press 1 for learning your friend')
    print('Press 2 to train the facial data')
    print('press 3 for face recognition')
    print('press 4 for object recognition')
    engine.say('Press 1 for learning your friend,press 2 to train facial data,press 3 for face recognition, press 4 for object recognition')
    engine.runAndWait()
    n=int(input('Press Now'))
    
    if(n==1):
        engine.say('Collecting face data')
        engine.runAndWait()
        face_data()
        engine.say('Please train the newly fed data')
        engine.runAndWait()        
        
    elif(n==2):
        engine.say('Training all the faces')
        engine.runAndWait()
        face_train()
    elif(n==3):
        engine.say('Recognizing the face')
        print("Recognizing the face")
        engine.runAndWait()
        face_recognize()
    elif(n==4):
        print("Trying to recognize objects infront of you")
        engine.say('Trying to recognize objects infront of you')
        engine.runAndWait()
        objects()
    else:
        print("Sorry could not understand your command, Please repeat")
        engine.say('Sorry could not understand your command, Please repeat')
        engine.runAndWait()
        speech()
while True:
    speech()