import pyttsx3
import speech_recognition as sr
import smtplib

engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
volume = engine.getProperty('volume')
engine.setProperty('volume', 10.0)
rate = engine.getProperty('rate')

engine.setProperty('rate', rate - 25)


#import os
#import re
#import webbrowser
#import smtplib
#import requests
#from weather import Weather





def speech():
        
    mail=['hi','mail','email','e-mail','mail mail','meal','meal meal','mel mel','mel mel mel']
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please give a command")
        
        engine.say('Please give a command')
        engine.runAndWait()
        r.pause_threshold = 1
        r.adjust_for_ambient_noise(source, duration=1)
        audio = r.listen(source)
        print("Processing your command......Please Wait")
        engine.say('Processing your command please wait')
        engine.runAndWait()
        try:
            text = r.recognize_google(audio)
            print("You said : {}".format(text))
            text=str(text)
            text=text.lower()
            engine.say('You said'+text)
            engine.runAndWait()
                
        except:
            print("Sorry could not recognize what you said, Please repeat")
            engine.say('Sorry could not recognize what you said, Please repeat')
            engine.runAndWait()
            speech()

#content = "Hi, How are you"
    if (text in mail) :
        mail1()
    else:
        speech()
def mail1():
    contacts={'pranav':'pranavdeep1997@gmail.com'}
    
    print("Whom should I ping?")
    engine.say('Whom should I ping?')
    engine.runAndWait()
    #content="Hi, How are you doing?"
    
    r = sr.Recognizer()
    with sr.Microphone() as source:
        #print("Give the mail contents:")
        #engine.say('Give the mail contents')
        #engine.runAndWait()
        
        audio = r.listen(source)
        print("Processing your command......Please Wait")
        engine.say('Processing your command')
        engine.runAndWait()
        try:
            text = r.recognize_google(audio)
            print("You said : {}".format(text))
            text=str(text)
            text=text.lower()
            engine.say('You said'+text)
            engine.runAndWait()
                
        except:
            print("Sorry could not recognize what you said, Please repeat")
            engine.say('Sorry could not recognize what you said, Please repeat')
            engine.runAndWait()
            speech()
            
    if text in contacts:
        text=contacts[text]               
        print("Please give the content of the mail")
        engine.say('Please give the content of the mail')
        engine.runAndWait()
        r = sr.Recognizer()
        with sr.Microphone() as source:
            #print("Give the mail contents:")
            #engine.say('Give the mail contents')
            #engine.runAndWait()
            
            audio = r.listen(source)
            print("Processing your command......Please Wait")
            engine.say('Processing your command')
            engine.runAndWait()
            try:
                content = r.recognize_google(audio)
                print("You said : {}".format(content))
                content=str(content)
                content=content.lower()
                engine.say('You said'+content)
                engine.runAndWait()
                    
            except:
                print("Sorry could not recognize what you said, Please repeat")
                engine.say('Sorry could not recognize what you said, Please repeat')
                engine.runAndWait()
                speech()
        
                
        mail = smtplib.SMTP('smtp.gmail.com', 587)
        mail.ehlo()
        #encrypt session
        mail.starttls()
        #login
        mail.login('pranavdeep1997@gmail.com', '98665076612PD')
        #send message
        mail.sendmail('Pranav', text, content)
        #end mail connection
        mail.close()
        print("Message sent")
        engine.say('Message sent')
        engine.runAndWait()
        
    
speech()
    
