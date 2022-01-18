import tkinter as tk
from tkinter import *
import tkinter.font as tkFont
import numpy as np
import cv2
from PIL import Image
from PIL import ImageTk
import threading
from datetime import datetime
import subprocess
import os
import time

import tensorflow as tf
from object_detection.utils import label_map_util

# ------------------------------------------------------------------------------------------
# ------------------------------------------- detector class -------------------------------
class DetectorTF2:

	def __init__(self, path_to_checkpoint, path_to_labelmap, class_id=None, threshold=0.5):
		# class_id is list of ids for desired classes, or None for all classes in the labelmap
		self.class_id = class_id
		self.Threshold = threshold
		# Loading label map
		label_map = label_map_util.load_labelmap(path_to_labelmap)
		categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
		self.category_index = label_map_util.create_category_index(categories)

		tf.keras.backend.clear_session()
		self.detect_fn = tf.saved_model.load(path_to_checkpoint)


	def DetectFromImage(self, img):
		im_height, im_width, _ = img.shape
		# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
		input_tensor = np.expand_dims(img, 0)
		detections = self.detect_fn(input_tensor)

		bboxes = detections['detection_boxes'][0].numpy()
		bclasses = detections['detection_classes'][0].numpy().astype(np.int32)
		bscores = detections['detection_scores'][0].numpy()
		det_boxes = self.ExtractBBoxes(bboxes, bclasses, bscores, im_width, im_height)

		return det_boxes


	def ExtractBBoxes(self, bboxes, bclasses, bscores, im_width, im_height):
		bbox = []
		for idx in range(len(bboxes)):
			if self.class_id is None or bclasses[idx] in self.class_id:
				if bscores[idx] >= self.Threshold:
					y_min = int(bboxes[idx][0] * im_height)
					x_min = int(bboxes[idx][1] * im_width)
					y_max = int(bboxes[idx][2] * im_height)
					x_max = int(bboxes[idx][3] * im_width)
					class_label = self.category_index[int(bclasses[idx])]['name']
					bbox.append([x_min, y_min, x_max, y_max, class_label, float(bscores[idx])])
		return bbox


	def DisplayDetections(self, image, boxes_list, det_time=None):
		if not boxes_list: return image  # input list is empty
		img = image.copy()
		for idx in range(len(boxes_list)):
			x_min = boxes_list[idx][0]
			y_min = boxes_list[idx][1]
			x_max = boxes_list[idx][2]
			y_max = boxes_list[idx][3]
			cls =  str(boxes_list[idx][4])
			score = str(np.round(boxes_list[idx][-1], 2))

			text = cls + ": " + score
			cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
			cv2.rectangle(img, (x_min, y_min - 20), (x_min, y_min), (255, 255, 255), -1)
			# cv2.putText(img, text, (x_min + 5, y_min - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

		if det_time != None:
			fps = round(1000. / det_time, 1)
			fps_txt = str(fps) + " FPS"
			cv2.putText(img, fps_txt, (25, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

		return img

# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
# ------------------------------------------- tensor program -------------------------------
# For preprocessing image
def make_image_square(filename):
    img = cv2.imread(filename)
    # Size of the image
    # s = max(img.shape[0:2])
    s = 640

    # Creating a dark square with NUMPY
    f = np.zeros((s, s, 3), np.uint8)

    # Getting the centering position
    ax, ay = (s - img.shape[1])//2, (s - img.shape[0])//2

    # Pasting the 'image' in a centering position
    f[ay:img.shape[0]+ay, ax:ax+img.shape[1]] = img
    cv2.imwrite(filename, f)

# For preprocessing image
def crop_image():
    for image_index in range(2):
        folder_name = 'data/'
        image_name = folder_name + str(image_index + 1) + '.jpg'
        img = cv2.imread(image_name)

        h, w, c = img.shape

        w_constant = w/3
        h_constant = h/2

        image_part_index = 0

        for index_w in range(3):
            for index_h in range(2):
                start_width = int(w_constant * index_w)
                end_width = int(w_constant * (index_w + 1))

                start_height = int(h_constant * index_h)
                end_height = int(h_constant * (index_h + 1))

                current_index = image_part_index

                # For training image set
                # section_name = 'PL_8_' + str(image_index+1) + '_'
                # file_name = section_name + str(image_index+1) + '_' + str(image_part_index) + '.jpg'

                # For testing image set
                section_name = str(image_index+1) + '/'
                file_name = folder_name + section_name + \
                    str(image_part_index+1) + '.jpg'

                crop_img = img[start_height:end_height, start_width:end_width]

                image_part_index = image_part_index + 1
                cv2.imwrite(file_name, crop_img)

                make_image_square(file_name)

# For detection
def WriteFile(output_dir, file_name, content):
    file_output = os.path.join(output_dir, file_name)
    f = open(file_output, 'a+')
    f.write(content)
    f.close()

# For detection
def DetectImagesFromFolder(detector, images_dir, save_output=False, output_dir='output/'):
	total_detected = 0
	timestamp2 = time.time()

	for file in os.scandir(images_dir):
		if file.is_file() and file.name.endswith(('.jpg', '.jpeg', '.png')) :
			image_path = os.path.join(images_dir, file.name)
			img = cv2.imread(image_path)
			timestamp1 = time.time()
			det_boxes = detector.DetectFromImage(img)
			elapsed_time = round((time.time() - timestamp1) * 1000) #ms
			img = detector.DisplayDetections(img, det_boxes)

			total_detected = total_detected + len(det_boxes)
			text_to_save = str(file.name) + ':\t' + str(len(det_boxes)) + ' benur detected' + '\t' + '[' + str(elapsed_time/1000) + ' s] \t\n'

			if save_output:
				img_out = os.path.join(output_dir, file.name)
				cv2.imwrite(img_out, img)
				WriteFile(output_dir, 'ResultLog.txt', text_to_save)

	elapsed_time2 = round((time.time() - timestamp2) * 1000) #ms
	final_text_to_save = str(total_detected) + 'benur detected\t' + '[' + str(elapsed_time2/1000) + ' s]'
	if save_output:
		WriteFile(output_dir, 'Final.txt', final_text_to_save)
	return total_detected

# For detection
def execute_tf(model_path, threshold, output_directory, labelmap_path, images_dir, id_list_data = None):
    id_list = id_list_data
    if id_list_data is not None:
        id_list = [int(item) for item in id_list_data.split(',')]

    save_output = True
    if save_output:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

    # instance of the class DetectorTF2
    detector = DetectorTF2(model_path, labelmap_path,
                            class_id=id_list, threshold=threshold)

    result = DetectImagesFromFolder(
        detector, images_dir, save_output=True, output_dir=output_directory)

    return result


models = ['faster-rcnn-resnet101-15600']
threshold_setup = [0.3]
test_images_folders = ['1', '2']

# For detection
def detect_images():
    detected_total = []
    for threshold in threshold_setup:
        # Generate string for threshold output folder
        threshold_str = str(threshold)
        threshold_str = threshold_str.replace('.', '_')

        for folder in test_images_folders:

            # Generate string for output folder
            folder_subname = folder.replace('/', '_')

            for model in models:
                # Generate output directory
                output_directory = 'output_' + folder_subname + '_' + threshold_str

                detection_model_path = 'models/' + model
                detection_labelmap_path = 'models/Shrimp-seed-object_label_map.pbtxt'
                detection_images_dir = 'data/' + folder
                detection_output_dir = 'data/' + output_directory + '/' + model

                detection_result = execute_tf(detection_model_path, threshold, detection_output_dir, detection_labelmap_path, detection_images_dir)

                detected_total.append(int(detection_result))

    if(detected_total[0] > detected_total[1]):
        detected_result = detected_total[0]
    else:
        detected_result = detected_total[1]
    return float(detected_result)


def calculate():

    #=========== Detection Program here ========#
    crop_image()
    #=========== Should return the result ======#
    detected_result = detect_images()

    return detected_result

# ------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
# --------------------------------------------- GUI Section --------------------------------

white       = "#ffffff"
BlackSolid  = "#000000"
font        = "Constantia"
fontButtons = (font, 12)
maxWidth    = 1920
maxHeight   = 1080
scale = 2

class buttonL:

    def __init__(self, obj, size, position, text,font, fontSize, hoverColor,command=None):
        self.obj= obj
        self.size= size
        self.position= position
        self.font= font
        self.fontSize= fontSize
        self.hoverColor= hoverColor
        self.text= text
        self.command = command
        self.state = True
        self.Button_ = None

    def myfunc(self):
        print("Hello size :" , self.size)
        print("Hello position :" , self.position)
        print("Hello font :" , self.font)
        print("Hello fontSize :" , self.fontSize)
        print("Hello hoverState :" , self.hoverColor)
  
    def changeOnHover(self, obj,colorOnHover, colorOnLeave):
         obj.bind("<Enter>", func=lambda e: obj.config(
             background=colorOnHover))

         obj.bind("<Leave>", func=lambda e: obj.config(
             background=colorOnLeave))
            
    def buttonShow(self):
        fontStyle = tkFont.Font(family= self.font, size=self.fontSize,weight="bold")
        self.Button_ = Button(self.obj,text = self.text, font=fontStyle, width = self.size[0], height = self.size[1],  bg =   self.hoverColor[1] if isinstance(self.hoverColor, list)  == True else  self.hoverColor, compound=TOP,command=self.command)         
        self.Button_.place(x=self.position[0],y=self.position[1])

        if isinstance(self.hoverColor, list) == True:
            self.changeOnHover(self.Button_, self.hoverColor[0], self.hoverColor[1])
        else:
            self.changeOnHover(self.Button_, self.hoverColor, self.hoverColor)
    
    def stateButton(self,st):
        self.st=st
        if not self.Button_ == None:
            self.Button_["state"]=self.st

class framecontroller(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        
         #Graphics window
        self.mainWindow = self
        self.mainWindow.configure(bg=BlackSolid)
        self.mainWindow.geometry('%dx%d+%d+%d' % (maxWidth,maxHeight,0,0))
        self.mainWindow.resizable(0,0)
        self.mainWindow.title("SHRICO")
        self.mainWindow.attributes("-fullscreen", True)
        
        # # creating a container
        container = tk.Frame(self.mainWindow) 
        container.configure(bg=BlackSolid)
        container.pack(side = "top", fill = "both", expand = True)
  
        container.grid_rowconfigure(0, weight = 1)
        container.grid_columnconfigure(0, weight = 1)
        
        self.frames = {} 

        for F in (StartPage,Page1):
  
            frame = F(container, self.mainWindow)

            self.frames[F] = frame
  
            frame.grid(row = 0, column = 0, sticky ="nsew")
  
        self.show_frame(StartPage)
    
    def show_frame(self, cont):
        
        frame = self.frames[cont]
        frame.tkraise()


class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.ratarata=0

        self.configure(bg=BlackSolid)

        fontStyleLabel= tkFont.Font(family="Arial", size=40*scale)
        self.label1 = Label( self, text="Jumlah Benih", bg='#000', fg='#fff', font=fontStyleLabel)
        self.label1.pack()
        self.label1.place(x=80*scale,y=100*scale)

        fontStyleLabel= tkFont.Font(family="Arial", size=100*scale)
        self.label2 = Label(self, text=0, bg='#000', fg='#fff', font=fontStyleLabel)
        self.label2.pack()
        self.label2.place(x=80*scale,y=200*scale)

        fontStyle = tkFont.Font(family= "Arial", size=15*scale,weight="bold")
        self.button1 = buttonL(self,[15*scale,2*scale],[580*scale,40*scale],"Kalibrasi",fontStyle,15,["yellow",white],lambda : [controller.show_frame(Page1)])
        self.button1.buttonShow()

        self.button2 = buttonL(self,[15*scale,2*scale],[580*scale,280*scale],"Hitung Benih",fontStyle,15,["yellow",white],self.Waitcalculate)
        self.button2.buttonShow()

        self.button3 = buttonL(self,[15*scale,2*scale],[580*scale,380*scale],"Matikan Alat",fontStyle,15,["yellow",white],lambda : self.close())
        self.button3.buttonShow()

        

    def Waitcalculate(self):
        fontStyleLabel= tkFont.Font(family="Arial", size=20*scale)
        self.label3 = Label( self, text="Proses Deteksi Sedang Berlangsung", bg='#000', fg='#fff', font=fontStyleLabel)
        self.label3.pack()
        self.label3.place(x=80*scale,y=50*scale)

        self.label2.configure(text="~")

        fontStyleLabel= tkFont.Font(family="Arial", size=15*scale)
        self.now = datetime.now()
        self.dt_string = self.now.strftime("%B %d, %Y %H:%M:%S")
        self.label4 = Label( self, bg='#000', fg='#fff', font=fontStyleLabel)
        self.label4.configure(text="Waktu:\n"+self.dt_string,justify="left")
        self.label4.pack()
        self.label4.place(x=80*scale,y=400*scale)

        self.button1.stateButton("disabled")
        self.button2.stateButton("disabled")
        self.button3.stateButton("disabled")

        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.tensorflow)
        self.thread.start()

    def tensorflow(self):
        #================ Process ===================#
        value=calculate()
        self.ratarata = value
        #============================================#
        self.stopEvent.set()
        self.Resultcalculate(self.ratarata)

    def Resultcalculate(self,ratarata):
        
        self.label3.configure(text="Proses Deteksi Selesai")
        self.label2.configure(text=ratarata)

        self.button1.stateButton("active")
        self.button1.buttonShow()
        self.button2.stateButton("active")
        self.button2.buttonShow()
        self.button3.stateButton("active")
       
    def close(self):
        subprocess.run('sudo shutdown -h now', shell=True)



class Page1(tk.Frame):
     
    def __init__(self, parent, controller):
         
        tk.Frame.__init__(self, parent)
        self.videoObj = None

        self.configure(bg=BlackSolid)

        fontStyleLabel= tkFont.Font(family="Arial", size=14)
        label1 = Label( self, text="Pastikan Wadah Benih\nUdang Terlihat Jelas\nMelalui Kamera", bg='#000', fg='#fff', font=fontStyleLabel)
        label1.pack()
        label1.place(x=600*scale,y=50*scale)

        fontStyle = tkFont.Font(family= "Arial", size=15,weight="bold")
        button1 = buttonL(self,[12*scale,1*scale],[620*scale,400*scale],"Selesai",fontStyle,15,["yellow",white],lambda : [controller.show_frame(StartPage), videoStream.onClose(self.videoObj)])
        button1.buttonShow()

        button2 = buttonL(self,[12*scale,1*scale],[620*scale,300*scale],"Camera On",fontStyle,15,["yellow",white],lambda : [ videoStream.onStart(self.videoObj)])
        button2.buttonShow()

        self.videoObj = videoStream()



class videoStream(tk.Frame):
    def __init__(self):
    
        self.ret = None
        self.frame = None

        self.thread = None
        self.stopEvent = None 
        self.capWebcam = None

        self.panel = None
    
    def onStart(self):

        self.capWebcam = cv2.VideoCapture(0) 
        if not self.capWebcam.isOpened():
            raise Exception("Could not open video device")
        self.capWebcam.set(cv2.CAP_PROP_FRAME_WIDTH, 600*scale)
        self.capWebcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480*scale)
  
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop)
        self.thread.start()

    def onClose(self):
        print("[INFO] closing...")
        if not self.panel == None:
            self.panel.destroy()
            self.stopEvent.set()
            self.capWebcam.release()
        

    def videoLoop(self):
        try:
            # keep looping over frames until we are instructed to stop
            while not self.stopEvent.is_set():
                
                self.ret,self.frame = self.capWebcam.read()
           
                if(self.ret==True):
                    image = cv2.flip(self.frame, 1)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(image)
                    image = ImageTk.PhotoImage(image)
                
                    # if the panel is not None, we need to initialize it
                    if self.panel is None:
                       
                        self.panel = Label(image=image,width=600*scale,height=480*scale)
                        self.panel.image = image
                        self.panel.place(x=0,y=0)
            
                    # otherwise, simply update the panel
                    else:
                     
                        if(not self.panel == None):
                            self.panel.configure(image=image)
                            self.panel.image = image
                else:
               
                    self.panel.destroy()
                    self.panel = None


        except RuntimeError:
            print("[INFO] caught a RuntimeError")

app = framecontroller()
app.mainloop()


