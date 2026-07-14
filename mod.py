from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

# IBM Watson credentials
IBM_API_KEY = "suaR-uXPVkFosQLiQYUsT1ecnpayauFtY46T7SPK3WB0"
IBM_URL = "https://api.au-syd.text-to-speech.watson.cloud.ibm.com/instances/793b8d0c-c413-442e-aca9-a2c219c13dcb"

# Setup IBM Watson Text-to-Speech service
authenticator = IAMAuthenticator(IBM_API_KEY)
tts = TextToSpeechV1(authenticator=authenticator)
tts.set_service_url(IBM_URL)

# Text to convert to speech
text = "Hello, this is a test using IBM Watson Text-to-Speech."

# Generate speech
with open("output.mp3", "wb") as audio_file:
    response = tts.synthesize(
        text,
        voice="en-US_AllisonV3Voice",
        accept="audio/mp3"
    ).get_result()
    audio_file.write(response.content)

print("Audio file generated: output.mp3")