from time import sleep
import pyttsx3
from ftplib import FTP 

thisdict =	{
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
engine = pyttsx3.init()
#engine.say(a)
a = str(getFTPConfig('thisdict')+"whatever")
#engine.say(thisdict+"is in front of you"    )
#engine.runAndWait()

#sleep(1)
#engine.say("is in front of you")
#engine.runAndWait()
