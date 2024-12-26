# -*- coding: utf-8 -*-
import os
import openai
import pyttsx3
import speech_recognition as sr
import cv2
import numpy as np

# Initialize OpenAI API
openai.api_key = "OpenAI Your Key"

# Initialize Text-to-Speech Engine
tts_engine = pyttsx3.init()

# Initialize Speech Recognition
recognizer = sr.Recognizer()

# Tool 1: Play Sound
def play_sound():
    tts_engine.say("This is a test sound")
    tts_engine.runAndWait()

# Tool 2: Speech-to-Text (STT)
def speech_to_text():
    with sr.Microphone() as source:
        print("Listening for input...")
        audio_data = recognizer.listen(source)
        try:
            # Convert speech to text
            text = recognizer.recognize_google(audio_data, language="zh-TW")  # 支援中文
            print(f"Recognized Text: {text}")
            # Save audio and text to files
            with open("output.txt", "w", encoding="utf-8") as text_file:
                text_file.write(text)
            with open("output.wav", "wb") as audio_file:
                audio_file.write(audio_data.get_wav_data())
            return text
        except Exception as e:
            print(f"Error: {e}")

# Tool 3: Count People in Camera Frame
def count_people():
    cap = cv2.VideoCapture(0)  # Open the default camera
    print("Accessing camera...")
    if not cap.isOpened():
        print("Error: Camera not accessible")
        return

    # Load pre-trained model for detecting people
    net = cv2.dnn.readNetFromCaffe(
        "./models/deploy.prototxt",  # Path to the model configuration file
        "./models/res10_300x300_ssd_iter_140000.caffemodel"  # Path to the model weights
    )

    ret, frame = cap.read()
    if ret:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        count = 0
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Threshold for confidence
                count += 1
        print(f"Number of people detected: {count}")
    else:
        print("Error: Unable to read from camera")

    cap.release()
    cv2.destroyAllWindows()

# Main AI Agent Function
def ai_agent(command):
    if command in ["播放嗽叭", "播放聲音", "讓喇叭發聲"]:
        play_sound()
    elif command in ["STT", "語音轉文字"]:
        speech_to_text()
    elif command in ["現場人數", "鏡頭中人數"]:
        count_people()
    else:
        print("Sorry, I didn't understand the command.")


# Example Usage
if __name__ == "__main__":
    while True:
        print("\n請選擇輸入方式：\n1. 打字輸入\n2. 語音輸入")
        choice = input("輸入選擇 (1 或 2)：")
        
        if choice == "1":
          
            user_input = input("請輸入指令: ")
        elif choice == "2":
            
            try:
                with sr.Microphone() as source:
                    recognizer.adjust_for_ambient_noise(source)
                    print("Listening for input...")
                    audio = recognizer.listen(source)
                    user_input = recognizer.recognize_google(audio, language="zh-TW")
                    print(f"您說的是：{user_input}")

            except sr.UnknownValueError:
                print("抱歉，我無法理解您的語音指令，請再試一次。")
                continue
            except sr.RequestError as e:
                print(f"語音辨識服務出現錯誤：{e}")
                continue
        else:
            print("無效的選擇，請重新輸入。")
            continue

        if user_input in ["退出", "結束"]:
            print("結束程式。")
            break

        ai_agent(user_input)
