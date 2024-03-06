from faster_whisper import WhisperModel


mp3_path = r"G:\Drive partagés\rfp-s\kbc-project\meetings\Outsourcing belgary.mp3"
output_path = './transcription.txt'

model_size = "large-v3"

# Choisissez la configuration en fonction de votre matériel. Exemple ici pour CPU avec INT8.
# or run on CPU with INT8
model = WhisperModel(model_size, device="cpu", compute_type="int8")
#model = WhisperModel(model_size, device="cuda", compute_type="float16")  # Exemple pour GPU avec FP16
# Remplacez "cuda" par "cpu" pour CPU et ajustez "compute_type" comme vous le souhaitez
# or run on GPU with INT8
#model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")



# Transcription de l'audio
segments, info = model.transcribe(mp3_path, beam_size=5)

# Affichage de la langue détectée
print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

# Création ou ouverture du fichier de transcription
with open(output_path, 'w', encoding='utf-8') as f:
    # Écriture de chaque segment transcrit dans le fichier
    for segment in segments:
        segment_text = "[%.2fs -> %.2fs] %s\n" % (segment.start, segment.end, segment.text)
        print(segment_text, end='')  # Affichage à l'écran
        f.write(segment_text)  # Écriture dans le fichier
