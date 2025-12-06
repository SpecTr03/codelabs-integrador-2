import sounddevice as sd
from scipy.io.wavfile import write
import speech_recognition as sr
import tempfile, os
import webbrowser
from datetime import datetime
from googletrans import Translator
import pyjokes

SRATE = 16000     # tasa de muestreo
DUR = 5           # segundos

print("Grabando... habla ahora!")
audio = sd.rec(int(DUR*SRATE), samplerate=SRATE, channels=1, dtype='int16')
sd.wait()
print("Listo, procesando...")

# guarda a WAV temporal
tmp_wav = tempfile.mktemp(suffix=".wav")
write(tmp_wav, SRATE, audio)

# reconoce con SpeechRecognition
r = sr.Recognizer()
with sr.AudioFile(tmp_wav) as source:
    data = r.record(source)

try:
    texto = r.recognize_google(data, language="es-ES")  # español
    print("Dijiste:", texto)

    # --- integración de comandos ---
    cmd = texto.lower()

    if "hola" in cmd:
        print("¡Hola, bienvenido al curso!")

    elif "abrir google" in cmd:
        webbrowser.open("https://www.google.com")

    elif "hora" in cmd:
        print("Hora actual:", datetime.now().strftime("%H:%M"))

    # --- nuevos comandos ---
    elif "traducir" in cmd:
        frase = cmd.replace("traducir", "").strip()
        if frase:
            translator = Translator()
            traduccion = translator.translate(frase, dest="en")
            print(f"Traducción: {traduccion.text}")
        else:
            print("Debes decir: traducir + frase.")

    elif "abrir youtube" in cmd:
        cancion = cmd.replace("abrir youtube", "").strip()
        if cancion:
            url = f"https://www.youtube.com/results?search_query={cancion}"
            webbrowser.open(url)
            print(f"Buscando en YouTube: {cancion}")
        else:
            webbrowser.open("https://www.youtube.com")
            print("Abriendo YouTube.")

    elif "chiste" in cmd:
        joke = pyjokes.get_joke(language="es")
        print("Aquí tienes un chiste:", joke)

    else:
        print("Comando no reconocido.")

except sr.UnknownValueError:
    print("No se entendió el audio.")
except sr.RequestError as e:
    print("Error:", e)
finally:
    if os.path.exists(tmp_wav):
        os.remove(tmp_wav)