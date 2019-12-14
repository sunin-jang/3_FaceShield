#!/usr/bin/env python
# coding: utf-8

# In[5]:


from flask import Flask, render_template, request, send_file
from werkzeug import secure_filename
import tensorflow as tf
import cv2
import time
import argparse
import numpy as np
import os
import posenet
import face_recognition
import tensorflow as tf

def blur():
   # file = 비디오파일 이름(아래와 같이 적으려면 파이썬파일과 비디오파일을 같은 위치에 두세요.)

    upload_dir = 'uploads/'
    file_list = os.listdir(upload_dir)
    file=upload_dir+file_list[0] # file_list의 첫번째 파일

    if file is not None:
        cap = cv2.VideoCapture(file)
    height = int(cap.get(4))
    width = int(cap.get(3)) 
    model = 101
    cam_id = 1
    cam_width = 1280
    cam_height = 720
    scale_factor = 0.7125

    def set_blur_range(
            img, instance_scores, keypoint_scores, keypoint_coords,
            min_pose_score=0.5, min_part_score=0.5):
        out_img = img
        adjacent_keypoints = []
        cv_keypoints = []

        people_count = []
        list_start_blur_y = []
        list_end_blur_y = []
        list_start_blur_x = []
        list_end_blur_x = []

        for ii, score in enumerate(instance_scores):
            if score < min_pose_score:
                continue

            new_keypoints = posenet.utils.get_adjacent_keypoints(
                keypoint_scores[ii, :], keypoint_coords[ii, :, :], min_part_score)
            adjacent_keypoints.extend(new_keypoints)

            people_count.append(ii)

            kc2 = []
            #print(ii, "번째 객체")

            for ks, kc in zip(keypoint_scores[ii, :5], keypoint_coords[ii, :5, :]):
                if ks < min_part_score:
                    continue
                cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))
                if kc[1] < 0:
                    kc2.append([kc[0],1])
                elif kc[1] > width:
                    kc2.append([kc[0],width-3])
                elif kc[0] < 0:
                    kc2.append([1, kc[1]])
                elif kc[0] > height:
                    kc2.append([height-3, kc[1]])
                elif kc[0] < 0 and kc[1] < 0:
                    kc2.append([1,1])
                elif kc[0] > height and kc[1] > width:
                    kc2.append([height-3,width-3])
                else:
                    kc2.append(kc)


            if len(kc2) >= 3:
            # 코
                nose_y = int(kc2[0][0]) # y축
                nose_x = int(kc2[0][1]) # x축
            # 오른쪽눈
                eye_right_y = int(kc2[1][0])
                eye_right_x = int(kc2[1][1])
            # 왼쪽눈
                eye_left_y = int(kc2[2][0])
                eye_left_x = int(kc2[2][1])            

                #print("> 1.Nose(y:",nose_y,", x:",nose_x,")","2.EyeRight(y:",eye_right_y,", x:",eye_right_x,")",
                #      "3.EyeLeft(y:",eye_left_y,", x:",eye_left_x,")") 

            # y값 최솟값, 최댓값, 중앙값 구하기
                value_y = [nose_y, eye_right_y, eye_left_y]
                min_value_y = min(value_y)
                max_value_y = max(value_y)
                mean_value_y = int(sum(value_y)/len(value_y))
            # x값 최솟값, 최댓값, 중앙값 구하기
                value_x = [nose_x, eye_right_x, eye_left_x]
                min_value_x = min(value_x)
                max_value_x = max(value_x)
                mean_value_x = int(sum(value_x)/len(value_x))

            # 블러 크기를 키우기 위한 변수 만들기
                value_y_gap = abs(max_value_y-min_value_y)
                value_x_gap = abs(max_value_x-min_value_x)

            # 블러 씌울 부분 변수 선언
                start_blur_y = abs(mean_value_y-int(value_y_gap*1.7))
                end_blur_y = mean_value_y+int(value_y_gap*2)
                start_blur_x = abs(min_value_x-value_y_gap)
                end_blur_x = max_value_x+value_y_gap
                if start_blur_x==end_blur_x:
                    end_blur_x += 3
                if start_blur_y==end_blur_y:
                    end_blur_y += 3
            # 근만씨 동영상 오류 해결중
                if abs(eye_right_x-eye_left_x) > 20:
                    start_blur_y = abs(min_value_y-value_y_gap)
                    end_blur_y = nose_y+abs(eye_right_x-eye_left_x)+value_y_gap
                    start_blur_x = abs(min_value_x-value_x_gap)
                    end_blur_x = max_value_x +value_x_gap
                #print("[",start_blur_y,":",end_blur_y,", ",start_blur_x,":",end_blur_x,"]")                                 

            elif len(kc2) >= 2:
            # 코
                nose_y = int(kc2[0][0]) # y축
                nose_x = int(kc2[0][1]) # x축
            # 오른쪽눈
                eye_right_y = int(kc2[1][0])
                eye_right_x = int(kc2[1][1])      

                #print("> 1.Nose(y:",nose_y,", x:",nose_x,")","2.EyeRight(y:",eye_right_y,", x:",eye_right_x,")") 

            # y값 최솟값, 최댓값, 중앙값 구하기
                value_y = [nose_y, eye_right_y]
                min_value_y = min(value_y)
                max_value_y = max(value_y)
                mean_value_y = int(sum(value_y)/len(value_y))
            # x값 최솟값, 최댓값, 중앙값 구하기
                value_x = [nose_x, eye_right_x]
                min_value_x = min(value_x)
                max_value_x = max(value_x)
                mean_value_x = int(sum(value_x)/len(value_x))            
            # 블러 크기를 키우기 위한 변수 만들기
                value_y_gap = abs(max_value_y-int(sum(value_y)/len(value_y)))
                value_x_gap = abs(max_value_x-int(sum(value_x)/len(value_x)))

            # 블러 씌울 부분 변수 선언
                start_blur_y = abs(mean_value_y-int(value_y_gap*1.7))
                end_blur_y = mean_value_y+int(value_y_gap*2)
                start_blur_x = abs(mean_value_x-value_y_gap)
                end_blur_x = mean_value_x+value_y_gap
                if start_blur_x==end_blur_x:
                    end_blur_x += 3
                if start_blur_y==end_blur_y:
                    end_blur_y += 3
                #print("[",start_blur_y,":",end_blur_y,", ",start_blur_x,":",end_blur_x,"]")

            else:
            # 블러 씌울 부분 변수 선언
                start_blur_y = 0
                end_blur_y = 0
                start_blur_x = 0
                end_blur_x = 0

            list_start_blur_y.append(start_blur_y)
            list_end_blur_y.append(end_blur_y)
            list_start_blur_x.append(start_blur_x)
            list_end_blur_x.append(end_blur_x)

        return list_start_blur_y,list_end_blur_y,list_start_blur_x,list_end_blur_x, people_count

    def main():
        with tf.Session() as sess:
            model_cfg, model_outputs = posenet.load_model(model, sess)
            output_stride = model_cfg['output_stride']

            if file is not None:
                cap = cv2.VideoCapture(file)
            else:
                cap = cv2.VideoCapture(cam_id)
            cap.set(3, cam_width)
            cap.set(4, cam_height)

            start = time.time()
            frame_count = 0

            list_count = 0
            start_blur_y_list = []
            end_blur_y_list = []
            start_blur_x_list = []
            end_blur_x_list = []
            blur_y_gap_list = []
            blur_x_gap_list = []  

            # 녹화 기능 설정
            fps = 29.92        # 초당 프레임
            # 녹화할 이미지 크기
            width = int(cap.get(3))   
            height = int(cap.get(4))
            # 코덱 종류 설정
            fcc = cv2.VideoWriter_fourcc(*'u263')
            # 설정 된 값으로 동영상 녹화
            out = cv2.VideoWriter("static/downloads/blur.mp4", fcc , fps,(width, height))
            while True:
                res, img = cap.read()
      
                if not res:
                    cap.release()
                    cv2.destroyAllWindows()
                    print("webcam failure")   
                    break

                input_image, display_image, output_scale = posenet.utils._process_input(
                    img, scale_factor=scale_factor, output_stride=output_stride)

                heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                    model_outputs,
                    feed_dict={'image:0': input_image}
                )

                pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                    heatmaps_result.squeeze(axis=0),
                    offsets_result.squeeze(axis=0),
                    displacement_fwd_result.squeeze(axis=0),
                    displacement_bwd_result.squeeze(axis=0),
                    output_stride=output_stride,
                    max_pose_detections=10,
                    min_pose_score=0.15)

                keypoint_coords *= output_scale

            # 블러처리한 이미지 범위
                list_start_blur_y,list_end_blur_y,list_start_blur_x,list_end_blur_x, people_count = set_blur_range(
                    display_image, pose_scores, keypoint_scores, keypoint_coords,
                    min_pose_score=0.15, min_part_score=0.1)
                # print(">"*10,"main 함수","<"*10)
                # print("총 인원수: ",len(people_count))
                # print('main3')
                for start_blur_y,end_blur_y,start_blur_x,end_blur_x,i in zip(list_start_blur_y,list_end_blur_y,list_start_blur_x,list_end_blur_x,people_count):

                    if start_blur_y!=0 and end_blur_y!=0 and start_blur_x!=0 and end_blur_x!=0:
                        start_blur_y_list.append(start_blur_y)
                        end_blur_y_list.append(end_blur_y)
                        start_blur_x_list.append(start_blur_x)
                        end_blur_x_list.append(end_blur_x)

                        blur_y_gap = abs(end_blur_y-start_blur_y)
                        blur_x_gap = abs(end_blur_x-start_blur_x)
                        blur_y_gap_list.append(blur_y_gap)
                        blur_x_gap_list.append(blur_x_gap)

                    #블러 씌우는 부분
                        img_blur = display_image[start_blur_y:end_blur_y, start_blur_x:end_blur_x]
                        img_blur = cv2.blur(img_blur, (20,20), anchor=(-1, -1), borderType=cv2.BORDER_DEFAULT)
                        display_image[start_blur_y:end_blur_y, start_blur_x:end_blur_x] = img_blur

                    elif start_blur_y==0 and end_blur_y==0 and start_blur_x==0 and end_blur_x==0:

                        display_image = img

                #print("-"*60,'frame ',frame_count,"-"*60)
                #cv2.imshow('posenet', display_image)
                # 동영상 녹화
                out.write(display_image)

                frame_count += 1

                # 숫자 1 을 누르면 창이 닫혀요.
                if cv2.waitKey(1) == 49:
                    cap.release()
                    cv2.destroyAllWindows()
                    break

            #print('Average FPS: ', frame_count / (time.time() - start))


    if __name__ == "__main__":
        start_time = time.time()
        main()
        end_time = time.time()
        model_time = (end_time-start_time)/60
        print("sec : ", end_time-start_time, "min : ", model_time)

def premium_blur():
   # file = 비디오파일 이름(아래와 같이 적으려면 파이썬파일과 비디오파일을 같은 위치에 두세요.)

    upload_dir = 'uploads_premium/'
    file_list = os.listdir(upload_dir)
    file=upload_dir+file_list[0] # file_list의 첫번째 파일
    print(file)

    if file is not None:
        cap = cv2.VideoCapture(file)
    print(cap.isOpened())
    height = int(cap.get(4))
    width = int(cap.get(3)) 
    model = 101
    cam_id = 1
    cam_width = 1280
    cam_height = 720
    scale_factor = 0.7125

    def set_blur_range(
            img, instance_scores, keypoint_scores, keypoint_coords,
            min_pose_score=0.5, min_part_score=0.5):
        out_img = img
        adjacent_keypoints = []
        cv_keypoints = []

        people_count = []
        list_start_blur_y = []
        list_end_blur_y = []
        list_start_blur_x = []
        list_end_blur_x = []

        for ii, score in enumerate(instance_scores):
            if score < min_pose_score:
                continue

            new_keypoints = posenet.utils.get_adjacent_keypoints(
                keypoint_scores[ii, :], keypoint_coords[ii, :, :], min_part_score)
            adjacent_keypoints.extend(new_keypoints)

            people_count.append(ii)

            kc2 = []
            #print(ii, "번째 객체")

            for ks, kc in zip(keypoint_scores[ii, :5], keypoint_coords[ii, :5, :]):
                if ks < min_part_score:
                    continue
                cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))
                if kc[1] < 0:
                    kc2.append([kc[0],1])
                elif kc[1] > width:
                    kc2.append([kc[0],width-3])
                elif kc[0] < 0:
                    kc2.append([1, kc[1]])
                elif kc[0] > height:
                    kc2.append([height-3, kc[1]])
                elif kc[0] < 0 and kc[1] < 0:
                    kc2.append([1,1])
                elif kc[0] > height and kc[1] > width:
                    kc2.append([height-3,width-3])
                else:
                    kc2.append(kc)

            if len(kc2) >= 3:
            # 코
                nose_y = int(kc2[0][0]) # y축
                nose_x = int(kc2[0][1]) # x축
            # 오른쪽눈
                eye_right_y = int(kc2[1][0])
                eye_right_x = int(kc2[1][1])
            # 왼쪽눈
                eye_left_y = int(kc2[2][0])
                eye_left_x = int(kc2[2][1])            

            # y값 최솟값, 최댓값, 중앙값 구하기
                value_y = [nose_y, eye_right_y, eye_left_y]
                min_value_y = min(value_y)
                max_value_y = max(value_y)
                mean_value_y = int(sum(value_y)/len(value_y))
            # x값 최솟값, 최댓값, 중앙값 구하기
                value_x = [nose_x, eye_right_x, eye_left_x]
                min_value_x = min(value_x)
                max_value_x = max(value_x)
                mean_value_x = int(sum(value_x)/len(value_x))

            # 블러 크기를 키우기 위한 변수 만들기
                value_y_gap = abs(max_value_y-min_value_y)
                value_x_gap = abs(max_value_x-min_value_x)

            # 블러 씌울 부분 변수 선언
                start_blur_y = abs(mean_value_y-int(value_y_gap*1.7))
                end_blur_y = mean_value_y+int(value_y_gap*2)
                start_blur_x = abs(min_value_x-value_y_gap)
                end_blur_x = max_value_x+value_y_gap
                if start_blur_x==end_blur_x:
                    end_blur_x += 3
                if start_blur_y==end_blur_y:
                    end_blur_y += 3
            # 근만씨 동영상 오류 해결중
                if abs(eye_right_x-eye_left_x) > 20:
                    start_blur_y = abs(min_value_y-value_y_gap)
                    end_blur_y = nose_y+abs(eye_right_x-eye_left_x)+value_y_gap
                    start_blur_x = abs(min_value_x-value_x_gap)
                    end_blur_x = max_value_x +value_x_gap

            elif len(kc2) >= 2:
            # 코
                nose_y = int(kc2[0][0]) # y축
                nose_x = int(kc2[0][1]) # x축
            # 오른쪽눈
                eye_right_y = int(kc2[1][0])
                eye_right_x = int(kc2[1][1])      

            # y값 최솟값, 최댓값, 중앙값 구하기
                value_y = [nose_y, eye_right_y]
                min_value_y = min(value_y)
                max_value_y = max(value_y)
                mean_value_y = int(sum(value_y)/len(value_y))
            # x값 최솟값, 최댓값, 중앙값 구하기
                value_x = [nose_x, eye_right_x]
                min_value_x = min(value_x)
                max_value_x = max(value_x)
                mean_value_x = int(sum(value_x)/len(value_x))            
            # 블러 크기를 키우기 위한 변수 만들기
                value_y_gap = abs(max_value_y-int(sum(value_y)/len(value_y)))
                value_x_gap = abs(max_value_x-int(sum(value_x)/len(value_x)))

            # 블러 씌울 부분 변수 선언
                start_blur_y = abs(mean_value_y-int(value_y_gap*1.7))
                end_blur_y = mean_value_y+int(value_y_gap*2)
                start_blur_x = abs(mean_value_x-value_y_gap)
                end_blur_x = mean_value_x+value_y_gap
                if start_blur_x==end_blur_x:
                    end_blur_x += 3
                if start_blur_y==end_blur_y:
                    end_blur_y += 3

            else:
            # 블러 씌울 부분 변수 선언
                start_blur_y = 0
                end_blur_y = 0
                start_blur_x = 0
                end_blur_x = 0

            list_start_blur_y.append(start_blur_y)
            list_end_blur_y.append(end_blur_y)
            list_start_blur_x.append(start_blur_x)
            list_end_blur_x.append(end_blur_x)

        return list_start_blur_y,list_end_blur_y,list_start_blur_x,list_end_blur_x, people_count
    def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.4):
        """
        Compare a list of face encodings against a candidate encoding to see if they match.

        :param known_face_encodings: A list of known face encodings
        :param face_encoding_to_check: A single face encoding to compare against the list
        :param tolerance: How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
        :return: A list of True/False values indicating which known_face_encodings match the face encoding to check
        """
        return list(face_recognition.face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)

    def main():
        with tf.Session() as sess:
            model_cfg, model_outputs = posenet.load_model(model, sess)
            output_stride = model_cfg['output_stride']

            if file is not None:
                cap = cv2.VideoCapture(file)
            else:
                cap = cv2.VideoCapture(cam_id)
            cap.set(3, cam_width)
            cap.set(4, cam_height)

            start = time.time()
            frame_count = 0

            list_count = 0
            start_blur_y_list = []
            end_blur_y_list = []
            start_blur_x_list = []
            end_blur_x_list = []
            blur_y_gap_list = []
            blur_x_gap_list = []  

            # Load a sample picture and learn how to recognize it.    
            upload_image_dir = 'uploads_image/'
            image_file_list = os.listdir(upload_image_dir)
            image_file=upload_image_dir+image_file_list[0] # file_list의 첫번째 파일
            face_image = face_recognition.load_image_file(image_file) # 이미지 로드
            face_image_encoding = face_recognition.face_encodings(face_image)[0]  # 로드된 이미지로부터 encoding 정보 취득
            # 등록된 사람에 대한 encoding 정보 list
            known_face_encodings = [
                face_image_encoding
            ]
            # 등록된 사람에 대한 label 정보
            known_face_names = [
                "Target"
            ]            
            
            # 녹화 기능 설정
            fps = 29.92        # 초당 프레임
            # 녹화할 이미지 크기
            width = int(cap.get(3))   
            height = int(cap.get(4))
            # 코덱 종류 설정
            fcc = cv2.VideoWriter_fourcc(*'u263')
            # 설정 된 값으로 동영상 녹화
            out = cv2.VideoWriter("static/downloads_premium/blur.mp4", fcc , fps,(width, height))
            while True:
                res, img = cap.read()
      
                if not res:
                    cap.release()
                    cv2.destroyAllWindows()
                    print("webcam failure")   
                    break
                rgb_frame = img[:, :, ::-1]  # X, Y, channel(R,G,B)

                input_image, display_image, output_scale = posenet.utils._process_input(
                    img, scale_factor=scale_factor, output_stride=output_stride)

                heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                    model_outputs,
                    feed_dict={'image:0': input_image}
                )

                pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                    heatmaps_result.squeeze(axis=0),
                    offsets_result.squeeze(axis=0),
                    displacement_fwd_result.squeeze(axis=0),
                    displacement_bwd_result.squeeze(axis=0),
                    output_stride=output_stride,
                    max_pose_detections=10,
                    min_pose_score=0.15)

                keypoint_coords *= output_scale

            # 블러처리한 이미지 범위
                list_start_blur_y,list_end_blur_y,list_start_blur_x,list_end_blur_x, people_count = set_blur_range(
                    display_image, pose_scores, keypoint_scores, keypoint_coords,
                    min_pose_score=0.15, min_part_score=0.1)

                face_range_list = []

                # 포즈넷으로 인식한 얼굴 부분
                for start_blur_y,end_blur_x,end_blur_y,start_blur_x in zip(list_start_blur_y,list_end_blur_x,list_end_blur_y,list_start_blur_x):
                    face_range = [(start_blur_y,end_blur_x,end_blur_y,start_blur_x)]
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_range)

                    # 얼굴 특징 매칭
                    for (top, right, bottom, left), face_encoding in zip(face_range, face_encodings):
                        matches = compare_faces(known_face_encodings, face_encoding)
                        name = "Unknown"

                        # 탐색된 얼굴의 특징과 등록된 얼굴의 특징을 비교 후 매치가 있을 경우!
                        if True in matches:
                            first_match_index = matches.index(True)
                            name = known_face_names[first_match_index]

                        if name!="Target" and top!=0 and right!=0 and bottom!=0 and left!=0 :
                            img_blur = img[top:bottom, left:right]
                            img_blur = cv2.blur(img_blur, (20,20), anchor=(-1, -1), borderType=cv2.BORDER_DEFAULT)
                            img[top:bottom, left:right] = img_blur

                    display_image = img

                # 동영상 녹화
                out.write(display_image)

                frame_count += 1

                # 숫자 1 을 누르면 창이 닫혀요.
                if cv2.waitKey(1) == 49:
                    cap.release()
                    cv2.destroyAllWindows()
                    break


    if __name__ == "__main__":
        start_time = time.time()
        main()
        end_time = time.time()
        print("sec : ", end_time-start_time)

app = Flask(__name__)

@app.route('/')
def index():
   return render_template('index.html')

@app.route('/about')
def about():
   return render_template('about.html')
   
#업로드 HTML 렌더링
@app.route('/upload')
def upload():
   return render_template('upload.html')

@app.route('/upload_premium')
def upload_premium():
   return render_template('upload_premium.html')

#비디오 업로드 처리
@app.route('/fileUpload', methods = ['GET', 'POST'])
def fileUpload():
   if request.method == 'POST':
      f = request.files['videofile']
      #저장할 경로 + 파일명
      f.save('uploads/' + secure_filename(f.filename))
      blur()
      return render_template('download.html')

@app.route('/imageUpload', methods = ['GET', 'POST'])
def imageupload():
   if request.method == 'POST':
      f = request.files['imgfile']
      #저장할 경로 + 파일명
      f.save('uploads_image/' + secure_filename(f.filename))
      return render_template('upload_premium.html')

@app.route('/fileUpload_premium', methods = ['GET', 'POST'])
def fileUpload_premium():
   if request.method == 'POST':
      f = request.files['videofile']
      #저장할 경로 + 파일명
      f.save('uploads_premium/' + secure_filename(f.filename))
      premium_blur()
      return render_template('download_premium.html')

@app.route('/board')
def board():
   return render_template('board.html')

@app.route('/download')
def download():
   return render_template('download.html')

@app.route('/download_premium')
def download_premium():
   return render_template('download_premium.html')

@app.route('/download_video')
def download_video():
   return send_file(file_name,
                    attachment_filename='blur.mp4',
                    as_attachment=True)

@app.route('/download_premiumvideo')
def download_premiumvideo():
   file_name = f"static\\downloads_premium\\blur.mp4"
   return send_file(file_name,
                    attachment_filename='blur.mp4',
                    as_attachment=True)

if __name__ == '__main__':
    #서버 실행
   app.run(host='192.168.219.106', port=5000)