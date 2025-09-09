from pydub import AudioSegment
import sys, re, json

skipped_time = 0
skipped_lang = 0
skipped_empty = 0


class WhisperModel:
    def __init__(self):
        import whisper

        self.model = whisper.load_model("turbo")

    def get_transcript(self, filename):
        import whisper

        global skipped_lang, skipped_empty
        result = None
        audio = whisper.load_audio(filename)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=self.model.dims.n_mels).to(
            self.model.device
        )
        _, probs = self.model.detect_language(mel)
        if "en" in probs and probs["en"] > 0.8:
            transcription = self.model.transcribe(audio)
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


class SpeechBrainModel:
    def __init__(self):
        from speechbrain.inference.ASR import EncoderDecoderASR
        from speechbrain.inference.ASR import EncoderASR

        # self.model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir="pretrained_models/asr-crdnn-rnnlm-librispeech", run_opts={"device":"cuda"})
        self.model = EncoderASR.from_hparams(
            source="speechbrain/asr-wav2vec2-librispeech",
            savedir="pretrained_models/asr-wav2vec2-librispeech",
            run_opts={"device": "cuda"},
        )

    def get_transcript(self, filename):
        global skipped_empty
        result = None
        text = self.model.transcribe_file(filename)
        if len(text) > 0 and re.search("[a-zA-Z]", text) is not None:
            result = text
        else:
            skipped_empty += 1
            sys.stderr.write("E")
        return result


def save_segment(model, chapter, phrase):
    global skipped_time
    audio = chapter[phrase[0] : phrase[1]]
    skipped = True
    if len(audio) < 30000:
        audio.export("temp.wav", format="wav", parameters=["-ar", "16000"])
        text = model.get_transcript("temp.wav")
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
        print("skipped|%d|%d" % (phrase[0], phrase[1]))


# model = SpeechBrainModel()
model = WhisperModel()

lines = sys.stdin.readlines()
chapter_number = 0
for line in lines:
    fields = line.split("|")
    name = fields[0].strip()
    sys.stderr.write("(%d/%d) Processing %s\n" % (chapter_number, len(lines), name))
    chapter = AudioSegment.from_mp3(name)
    chapter_length = len(chapter)
    phrases = json.loads(fields[1])
    if len(phrases) < 10:
        sys.stderr.write("Skipping " + name + " for too few phrases\n")
    else:
        print("chapter|" + name)
        for phrase in phrases:
            save_segment(model, chapter, phrase)
    sys.stderr.write(
        "\nTime skips: %d, Lang skips: %d, Empty skips: %d\n"
        % (skipped_time, skipped_lang, skipped_empty)
    )
    sys.stderr.flush()
    chapter_number += 1
