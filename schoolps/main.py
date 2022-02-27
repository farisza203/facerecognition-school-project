import tkinter as tk
from collections import deque
from tkinter.constants import BUTT, END, GROOVE, NW, RAISED,  RIDGE,  S, SUNKEN
import numpy as np
import cv2
from PIL import Image,ImageTk
import os
import face_recognition
import time
window =tk.Tk()
window.option_add("*Font","Helvetica 14")
window.geometry("300x350+100+20")
window.title("Face Recognition System")
window.resizable(False, False)
window.configure(bg="#7FACD6")
facerecog_Mainbutton= ImageTk.PhotoImage((Image.open('./Asset/savemain.png')).resize((250,100), Image.ANTIALIAS))
facerecog_Mainbutton_change= ImageTk.PhotoImage((Image.open('./Asset/savemain_change.png')).resize((250,100), Image.ANTIALIAS))
facedetect_Mainbutton= ImageTk.PhotoImage((Image.open('./Asset/facerecognitionmain.png')).resize((250,100), Image.ANTIALIAS))
facedetect_Mainbutton_change=ImageTk.PhotoImage((Image.open('./Asset/facerecognitionmain_change.png')).resize((250,100), Image.ANTIALIAS))
video_capture=cv2.VideoCapture(0+cv2.CAP_DSHOW)
def recog_enter(a):
    facerecog.configure(image=facerecog_Mainbutton_change)   
def recog_leave(a):
    facerecog.configure(image=facerecog_Mainbutton)
def detect_enter(a):
    facedetect.configure(image=facedetect_Mainbutton_change)
def detect_leave(a):
    facedetect.configure(image=facedetect_Mainbutton)
def save_file():

        Name=str(save_Entry.get())
        print("Your name is "+Name)
        newpath = f'./Known/{Name}'
        ret,capture=video_capture.read()
        if not os.path.exists(newpath):
            os.makedirs(newpath)
            print("make new floder")
            time.sleep(1)
            print("processing.......")
            time.sleep(2)
            filename=Name+"1.jpg"
            cv2.imwrite(f'./Known/{Name}/{filename}',capture)
            Path, dirs, files_now = next(os.walk("./Known/{}".format(Name)))
            file_count_now = len(files_now)
            print("now you have {} picture in floder".format(file_count_now))
            print(os.listdir(f'./Known/{Name}'))
        else :
            Path, dirs, files = next(os.walk("./Known/{}".format(Name)))
            file_count_beta = len(files)
            print("Before : you have {} picture in floder ".format(file_count_beta))
            print(os.listdir(f'./Known/{Name}'))
            time.sleep(1)
            print("processing.......")
            time.sleep(2)
            filename=Name+str(file_count_beta+1)+".jpg"
            cv2.imwrite('./Known/{}/{}'.format(Name,filename),capture)
            Path, dirs, files_now = next(os.walk("./Known/{}".format(Name)))
            file_count_now = len(files_now)
            print("After : you have {} picture in floder".format(file_count_now))
            print(os.listdir(f'./Known/{Name}'))
def facerecognition():
    def reset():
        fps_label.pack_forget()
        image_label.place_forget()
    def Entry_Callback(event):
        save_Entry.selection_range(0, END)
    reset()
    new_tab.title("จดจำใบหน้า")
    window.geometry("300x490+100+20")
    image_label.place(x = 0, y = 0)
    save_Label.configure(text='ลงชื่อผู้ใช้ระบบ')
    save_Label.pack(pady=20)
    save_Entry.pack()
    save_Entry.bind("<FocusIn>",Entry_Callback)
    save_Button.pack(pady=10)
    fps_label._frame_times = deque([0]*5)  
    fps_label.pack()
    cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
    faceCascade = cv2.CascadeClassifier(cascPathface)
    video_capture= cv2.VideoCapture(0+cv2.CAP_DSHOW)    
    video_capture.set(3, 640)
    video_capture.set(4, 480) 
    def all_update(new_tab, image_label, video_capture,fps_label):
        show_frames(image_label, video_capture)
        update_fps(fps_label)
        new_tab.after(0, func=lambda: all_update(new_tab, image_label, video_capture,fps_label))
    def update_fps(fps_label):
        frame_times = fps_label._frame_times
        frame_times.rotate()
        frame_times[0] = time.time()
        sum_of_deltas = frame_times[0] - frame_times[-1]
        count_of_deltas = len(frame_times) - 1
        try:
            fps = int(float(count_of_deltas) / sum_of_deltas)
        except ZeroDivisionError:
            fps = 0
        fps_label.configure(text=("FPS: {}".format(fps)))
    def show_frames(image_label, video_capture):
        ret,frame = video_capture.read(0)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        Face = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(60, 60),flags=cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in Face:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.waitKey(0)
        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(cv2.resize(frame,(0,0),fx=1.35,fy=1.35), cv2.COLOR_BGR2RGB)))
        image_label.configure(image=img)
        image_label._image_cache = img 
        (new_tab).update()
    new_tab.after(0,func=lambda:all_update(new_tab, image_label, video_capture,fps_label))
    new_tab.mainloop()
def facedetect():
    def reset():
        fps_label.pack_forget()
        image_label.place_forget()
        save_Label.pack_forget()
        save_Entry.pack_forget()
        save_Button.pack_forget()
    reset()
    new_tab.title("ตรวจสอบใบหน้า")
    window.geometry("300x370+100+20")
    image_label.place(x = 0, y = 0)  
    fps_label._frame_times = deque([0]*5)  
    fps_label.pack()
    video_capture = cv2.VideoCapture(0+cv2.CAP_DSHOW)
    video_capture.set(3, 640)
    video_capture.set(4, 480)
    known_faces = []
    known_names = []
    try:
        for name in os.listdir('Known'):
            for filename in os.listdir(f'Known/{name}'):
                image = face_recognition.load_image_file(f'Known/{name}/{filename}')
                test_encoding = face_recognition.face_encodings(image)
                if len(test_encoding) > 0 :
                    encoding = test_encoding[0]
                else:
                    continue
                known_faces.append(encoding)
                known_names.append(name)
    except:
        known_faces.append(None)
        known_names.append("Unknown")
    
        
        
    def all_update(new_tab, image_label, video_capture,fps_label):
        show_frames(image_label, video_capture)
        update_fps(fps_label)
        new_tab.after(0, func=lambda: all_update(new_tab, image_label, video_capture,fps_label))
    def update_fps(fps_label):
        frame_times = fps_label._frame_times
        frame_times.rotate()
        frame_times[0] = time.time()
        sum_of_deltas = frame_times[0] - frame_times[-1]
        count_of_deltas = len(frame_times) - 1
        try:
            fps = int(float(count_of_deltas) / sum_of_deltas)
        except ZeroDivisionError:
            fps = 0
        fps_label.configure(text=("FPS: {}".format(fps)))
    def show_frames(image_label, video_capture):
        face_locations = []
        face_encodings = []
        face_name = []
        face_percent=[]
        ret,frame = video_capture.read(0)
        if not ret:
            new_tab.destroy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small_frame = cv2.resize(rgb, (0, 0), fx=1/4, fy=1/4)
        try:
            face_locations = face_recognition.face_locations(small_frame)
            face_encodings = face_recognition.face_encodings(small_frame,face_locations)
            for encoding in face_encodings:
                face_distance=face_recognition.face_distance(known_faces,encoding)
                best_match = np.argmin(face_distance)
                value_percent = 1-face_distance[best_match]
                if value_percent >=0.5:
                    name=known_names[best_match]
                    percent=round(value_percent*100,0)
                    face_percent.append(int(percent))
                else:
                    name='Unknown'
                    face_percent.append(0)
                face_name.append(name)
            for ( (TOP,RIGHT,BOTTOM,LEFT), name,percent) in zip( face_locations, face_name,face_percent):
                if name == 'Unknown':
                    color_rectangle= [46,2,209]
                    color_Match=[0, 0, 255]
                else:
                    color_rectangle=[255,102,51]
                    color_Match=[0,255,0]
                cv2.rectangle(frame, (LEFT*4, TOP*4), (RIGHT*4,BOTTOM*4), color_rectangle, 2)
                cv2.putText(frame, name, (LEFT*4, TOP*4), cv2.FONT_HERSHEY_COMPLEX,1, (255, 255, 255), 2)
                cv2.putText(frame,'Matches '+str(percent)+' %',(LEFT*4,(BOTTOM*4)+16),cv2.FONT_HERSHEY_COMPLEX,0.5, color_Match, 1)
        except:
            face_locations = face_recognition.face_locations(small_frame)
            for (TOP,RIGHT,BOTTOM,LEFT) in face_locations:
                name = 'Unknown'
                color_rectangle= [46,2,209]
                color_Match=[0, 0, 255]

                cv2.rectangle(frame, (LEFT*4, TOP*4), (RIGHT*4,BOTTOM*4), color_rectangle, 2)
                cv2.putText(frame, name, (LEFT*4, TOP*4), cv2.FONT_HERSHEY_COMPLEX,1, (255, 255, 255), 2)
                cv2.putText(frame,'Matches '+'0'+' %',(LEFT*4,(BOTTOM*4)+16),cv2.FONT_HERSHEY_COMPLEX,0.5, color_Match, 1)
        cv2.waitKey(0)
        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(cv2.resize(frame,(0,0),fx=1.35,fy=1.35), cv2.COLOR_BGR2RGB)))
        image_label.configure(image=img)
        image_label._image_cache = img 
        (new_tab).update()
    new_tab.after(0,func=lambda:all_update(new_tab, image_label, video_capture,fps_label))
    new_tab.mainloop()
label_main = (tk.Label(master=window,text="Face Recognition System",bg="#7FACD6")).pack(pady=30)
facerecog=tk.Button(master=window,image=facerecog_Mainbutton,command=facerecognition,background="#7FACD6",activebackground="#7FACD6",borderwidth=0)
facerecog.pack()
facerecog.bind("<Enter>", recog_enter)
facerecog.bind("<Leave>", recog_leave)
facedetect=tk.Button(master=window,image=facedetect_Mainbutton,command=facedetect,background="#7FACD6",activebackground="#7FACD6",borderwidth=0)
facedetect.pack(pady=10)
facedetect.bind("<Enter>", detect_enter)
facedetect.bind("<Leave>", detect_leave)
new_tab=tk.Toplevel() 
new_tab.option_add("*Font","Helvetica 14")
new_tab.configure(bg='black')
new_tab.title("สวัสดีครับ")
new_tab.geometry("%dx%d+%d+%d" % (864,648, 300+100, 20))
new_tab.resizable(False, False)
image_label=tk.Label(master=new_tab,borderwidth=0)
save_Label =tk.Label(master=window,bg="#7FACD6",borderwidth=0)
save_Entry=tk.Entry(master=window,borderwidth=1,width=16,relief=SUNKEN,fg="#6E3CBC")
save_Button=tk.Button(master=window,command=save_file,bg="#B8E4F0",activebackground="#B8E4F0",text='บันทึกรูปภาพ',relief=RAISED,borderwidth=1)
fps_label = tk.Label(master=window,background="#7FACD6",foreground="#225140",borderwidth=0)  
new_tab.mainloop()
window.mainloop()