import smtplib
text='pranavdeep1997@gmail.com'
content="Hi, What's up?"
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
#engine.say('Sorry could not recognize what you said, Please repeat')
#engine.runAndWait()