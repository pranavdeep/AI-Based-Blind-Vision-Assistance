from twilio.rest import Client


# Twilio phone number goes here. Grab one at https://twilio.com/try-twilio
# and use the E.164 format, for example: "+12025551234"
TWILIO_PHONE_NUMBER = "+14087132918"
sid="ACf0607f9182d93f50332ae4c23c3b136e"
token="f1501059df0fac53b4ec2e20a519c392"

# list of one or more phone numbers to dial, in "+19732644210" format
DIAL_NUMBERS = {'pranav':'+917793963233'}
#DIAL_NUMBERS = list(DIAL_NUMBERS)

# URL location of TwiML instructions for how to handle the phone call
TWIML_INSTRUCTIONS_URL = \
  "http://static.fullstackpython.com/phone-calls-python.xml"

# replace the placeholder values with your Account SID and Auth Token
# found on the Twilio Console: https://www.twilio.com/console
#client = TwilioRestClient("ACxxxxxxxxxx", "yyyyyyyyyy")
tClient= Client(sid, token)

def dial_numbers(number):
    tClient.calls.create(to=number, from_=TWILIO_PHONE_NUMBER,
                            url=TWIML_INSTRUCTIONS_URL, method="GET")
    
    """Dials one or more phone numbers from a Twilio phone number.
    for number in numbers_list:
        print("Dialing " + number)
        # set the method to "GET" from default POST because Amazon S3 only
        # serves GET requests on files. Typically POST would be used for apps
        tClient.calls.create(to=number, from_=TWILIO_PHONE_NUMBER,
                            url=TWIML_INSTRUCTIONS_URL, method="GET")"""
dial_numbers(DIAL_NUMBERS['pranav'])  