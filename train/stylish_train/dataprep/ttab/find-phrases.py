from pydub import AudioSegment, silence

# import whisper
import random, os, sys, re, json

# whisper_model = whisper.load_model("turbo")
skipped_time = 0
skipped_lang = 0
skipped_empty = 0


def get_transcript(filename):
    global whisper_model, skipped_lang, skipped_empty
    result = None
    audio = whisper.load_audio(filename)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio, n_mels=whisper_model.dims.n_mels).to(
        whisper_model.device
    )
    _, probs = whisper_model.detect_language(mel)
    if "en" in probs and probs["en"] > 0.8:
        transcription = whisper_model.transcribe(audio)
        text = transcription["text"]
        if len(text) > 0:
            result = text
        else:
            skipped_empty += 1
            sys.stderr.write("E")
    else:
        skipped_lang += 1
        sys.stderr.write("L")
    return result


def save_segment(chapter, phrase):
    global whisper_model, skipped_time
    audio = chapter[phrase[0] : phrase[1]]
    skipped = True
    if len(audio) < 30000:
        audio.export("temp.wav", format="wav", parameters=["-ar", "16000"])
        text = get_transcript("temp.wav")
        if text is not None:
            text = re.sub(r"\s+", r" ", text)
            print("phrase|%d|%d|%s" % (phrase[0], phrase[1], text))
            sys.stdout.flush()
            sys.stderr.write(".")
            skipped = False
    else:
        skipped_time += 1
        sys.stderr.write("T")
    sys.stderr.flush()
    if skipped:
        print("skipped")


badchapters = {}
with open("worst.txt", "r") as f:
    for line in f:
        fields = line.split("\t")
        badchapters[fields[1].strip()] = True

with open("chapters.txt", "r") as chapterfile:
    lines = chapterfile.readlines()
    chapter_number = 0
    for line in lines:
        name = line.strip()
        if name.split("/")[-1] not in badchapters:
            sys.stderr.write(
                "(%d/%d) Processing %s\n" % (chapter_number, len(lines), name)
            )
            chapter = AudioSegment.from_mp3(name)
            chapter_length = len(chapter)
            phrases = silence.detect_nonsilent(chapter, 200, -50)
            if len(phrases) < 10:
                sys.stderr.write("Skipping " + name + " for too few phrases\n")
            else:
                print(name + "|" + json.dumps(phrases))
                # for phrase in phrases:
                #    #save_segment(chapter, phrase)
        else:
            sys.stderr.write(
                "(%d/%d) Skipping %s\n" % (chapter_number, len(lines), name)
            )
        sys.stderr.write(
            "\nTime skips: %d, Lang skips: %d, Empty skips: %d\n"
            % (skipped_time, skipped_lang, skipped_empty)
        )
        sys.stderr.flush()
        chapter_number += 1
