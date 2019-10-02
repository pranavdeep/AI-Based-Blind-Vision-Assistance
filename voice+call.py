from twilio.rest import Client
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


def speech():
        
    call=['hi','call','calls','call call','phone','phone call','phones']
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
    if (text in call) :
        call1()
    else:
        speech()

def call1():
    contacts={'pranav':'+917793963233'}
    print("Whom should I call?")
    engine.say('Whom should I call?')
    engine.runAndWait() 
    r = sr.Recognizer()
    with sr.Microphone() as source:
        
            r.pause_threshold = 1
            r.adjust_for_ambient_noise(source, duration=1)
            audio = r.listen(source)
            print("Processing your command......Please Wait")
            engine.say('Processing your command please wait')
            engine.runAndWait()
            try:
                number = r.recognize_google(audio)
                print("You said : {}".format(text))
                number=str(number)
                number=number.lower()
                engine.say('You said'+number)
                engine.runAndWait()
                    
            except:
                print("Sorry could not recognize what you said, Please repeat")
                engine.say('Sorry could not recognize what you said, Please repeat')
                engine.runAndWait()
                call1()
    
    
    
    # Twilio phone number goes here. Grab one at https://twilio.com/try-twilio
    # and use the E.164 format, for example: "+12025551234"
    TWILIO_PHONE_NUMBER = "+14087132918"
    sid="ACf0607f9182d93f50332ae4c23c3b136e"
    token="f1501059df0fac53b4ec2e20a519c392"
    
    # list of one or more phone numbers to dial, in "+19732644210" format
    #DIAL_NUMBERS = ["+917793963233"]
    
    # URL location of TwiML instructions for how to handle the phone call
    TWIML_INSTRUCTIONS_URL = \
      "http://static.fullstackpython.com/phone-calls-python.xml"
    
    # replace the placeholder values with your Account SID and Auth Token
    # found on the Twilio Console: https://www.twilio.com/console
    #client = TwilioRestClient("ACxxxxxxxxxx", "yyyyyyyyyy")
    tClient= Client(sid, token)
    number=contacts[number]
    tClient.calls.create(to=number, from_=TWILIO_PHONE_NUMBER,
                             url=TWIML_INSTRUCTIONS_URL, method="GET")
    #dial_numbers(contacts[pranav])  
speech()