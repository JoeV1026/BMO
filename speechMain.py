from speechListen import listen
from speechAPI import brain
from speechOutput import speak

while True:
    text = listen()

    if text:
        print("User:", text)

        try:
            response = brain(text)
        except Exception as e:
            response = "error"

        print("Robot:", response)
        speak(response)