import torch, torchaudio
import numpy as np
import soundfile as sf
import yaml
import importlib.resources
import os, random, sys
from style_map import StyleModel
from sentence_transformers import SentenceTransformer

torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(0)
np.random.seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"

word_index_dictionary = {}


def build_text_cleaner(dict):
    _pad = "$"
    _punctuation = ';:,.!?¡¿—…"()“” '
    _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

    # Export all symbols:
    symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

    for i in range(len((symbols))):
        dict[symbols[i]] = i


build_text_cleaner(word_index_dictionary)


def text_clean(text):
    indexes = []
    for char in text:
        try:
            indexes.append(word_index_dictionary[char])
        except KeyError:
            print("Phoneme Error [" + char + "] " + text)
    return indexes


def length_to_mask(lengths):
    mask = (
        torch.arange(lengths.max())
        .unsqueeze(0)
        .expand(lengths.shape[0], -1)
        .type_as(lengths)
    )
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask


class Model:
    def __init__(self, stylepath, configpath, modelpath):
        self.style_mapper = None
        self.prosody_mapper = None
        self.sentence_model = None
        self.styledir = os.path.abspath(stylepath)
        had_path = self.styledir in sys.path
        if not had_path:
            sys.path.insert(0, self.styledir)
        import models
        import utils
        from Utils.PLBERT.util import load_plbert
        from Modules.diffusion.sampler import (
            DiffusionSampler,
            ADPM2Sampler,
            KarrasSchedule,
        )

        if not had_path:
            sys.path.remove(self.styledir)

        self.s_prev = None
        self.noise = torch.randn(1, 1, 256).to(device)

        with open(configpath, "r") as f:
            self.config = yaml.safe_load(f)

        # load pretrained ASR model
        ASR_config = self.config.get("ASR_config", False)
        ASR_path = self.config.get("ASR_path", False)
        text_aligner = models.load_ASR_models(
            self.styledir + "/" + ASR_path, self.styledir + "/" + ASR_config
        )

        # load pretrained F0 model
        F0_path = self.config.get("F0_path", False)
        pitch_extractor = models.load_F0_models(self.styledir + "/" + F0_path)

        # load BERT model
        BERT_path = self.config.get("PLBERT_dir", False)
        plbert = load_plbert(self.styledir + "/" + BERT_path)

        if "skip_downsamples" not in self.config["model_params"]:
            self.config["model_params"]["skip_downsamples"] = False
        self.model = models.build_model(
            utils.recursive_munch(self.config["model_params"]),
            text_aligner,
            pitch_extractor,
            plbert,
        )
        _ = [self.model[key].eval() for key in self.model]
        _ = [self.model[key].to(device) for key in self.model]

        params_whole = torch.load(modelpath, map_location="cpu")
        params = params_whole["net"]

        for key in self.model:
            if key in params:
                print("%s loaded" % key)
                try:
                    self.model[key].load_state_dict(params[key])
                except:
                    from collections import OrderedDict

                    state_dict = params[key]
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:]  # remove `module.`
                        new_state_dict[name] = v
                    # load params
                    self.model[key].load_state_dict(new_state_dict, strict=False)
        _ = [self.model[key].eval() for key in self.model]

        self.sampler = DiffusionSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(
                sigma_min=0.0001, sigma_max=3.0, rho=9.0
            ),  # empirical parameters
            clamp=False,
        )

    def calculate_style(self, text, diffusion_steps=5, embedding_scale=1):
        tokens = text_clean(text)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
            text_mask = length_to_mask(input_lengths).to(tokens.device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            return self.sampler(
                self.noise,
                embedding=bert_dur[0].unsqueeze(0),
                num_steps=diffusion_steps,
                embedding_scale=embedding_scale,
            ).squeeze(0)

    def calculate_audio_style(self, ref_path):
        to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=80, n_fft=2048, win_length=1200, hop_length=300
        )  # ,
        # sample_rate=24000)
        mean, std = -4, 4

        wave, sr = sf.read(ref_path)
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
        mel_tensor = mel_tensor.to(device)

        ref_s = self.model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = self.model.predictor_encoder(mel_tensor.unsqueeze(1))
        # print(ref_s.shape, ref_p.shape)
        sys.stderr.write("S")
        sys.stderr.flush()
        return torch.cat([ref_s, ref_p], dim=1)

    def context_infer(self, index, lines, diffusion_steps=5, embedding_scale=1):
        # results = []
        before = ""  # gather_before(index, lines)
        after = gather_after(index, lines, 100)
        line = lines[index].strip()
        context_len = min(100, (510 - len(line)))  # //2)
        begin_index = 0  # min(context_len, len(before))
        end_index = begin_index + len(line)
        text = before[-context_len:] + line + after[:context_len]
        audio, _, _ = self.encode(
            text, begin_index, end_index, diffusion_steps, embedding_scale
        )
        i = 0
        j = audio.shape[0] - 1
        threshold = 1000 / 32768
        while i < audio.shape[0] and abs(audio[i].data) < threshold:
            i += 1
        while j > 0 and abs(audio[j].data) < threshold:
            j -= 1
        j = min(audio.shape[0], j + 9000)
        return audio[i:j].numpy()

    def continuous(self, words, diffusion_steps=5, embedding_scale=1):
        results = []
        for i in range(len(words)):
            if words[i] == "":
                continue
            #    print("WORD: " + words[i])
            sys.stderr.write(".")
            sys.stderr.flush()
            if i % 50 == 0:
                sys.stderr.write("%d/%d\n" % (i, len(words)))
            word = words[i]
            before = ""  # gather_before(i, words, 100)
            after = gather_after(i, words, 150)
            begin_index = len(before)
            end_index = begin_index + len(word) + 1
            text = before + word + after
            # encoding, style, t_en, pred_aln_trg = self.encode(text, begin_index,
            #    print("Text [" + text[begin_index:end_index] + "]")
            # encoding, begin_padding, end_padding = self.encode(text, begin_index,
            # a, f, n, r = self.encode(
            audio, duration, pad_duration = self.encode(
                text, begin_index, end_index, diffusion_steps, embedding_scale
            )
            # window_size = 100
            # a2 = np.power(audio.numpy(), 2)
            # window = np.ones(window_size) / float(window_size)
            # volume = np.sqrt(np.convolve(a2, window, 'valid'))
            # min_amp = 1
            # min_index = 0
            j = audio.shape[0] - 1
            threshold = 1000 / 32768
            while i < audio.shape[0] and abs(audio[i].data) < threshold:
                i += 1
            # for i in range(pad_duration*300):
            #    if volume[i] < min_amp:
            #        min_amp = volume[i]
            #        min_index = i
            # print("MIN_INDEX", min_index)
            while j > i + duration * 300 and abs(audio[j].data) < threshold:
                j -= 1
            # audio = audio[min_index:j].numpy()
            audio = audio[i:j].numpy()
            results.append(audio)
            if audio.shape[0] < duration * 300:
                print("LESS")
                padding = duration * 300 - audio.shape[0]
                results.append(np.zeros([padding]))
            # asr.append(a)
            # f0.append(f)
            # npred.append(n)
            # styles.append(r)
            # print("ENCODING SHAPE", encoding.shape)
            # if result is None:
            #    result = encoding
            # else:
            #    result = blend_to_end(result, encoding)
            # if result is None:
            #    result = encoding[:-300*end_padding]
            # else:
            #    result = np.concatenate([result,
            #                             blend_together(loose_end, encoding[:300*begin_padding]),
            #                             encoding[300*begin_padding:-300*end_padding]])
            # loose_end = encoding[-300*end_padding:]
            # results.append(encoding)
            # results.append(self.decode(encoding, style, t_en, pred_aln_trg))
        return np.concatenate(results)
        # result = decode(torch.concatenate(asr, axis=2),
        #                torch.concatenate(f0, axis=2),
        #                torch.concatenate(npred, axis=2),
        #                torch.concatenate(styles, axis=2))
        # result = np.concatenate([result,
        #                         blend_together(loose_end, np.zeros([loose_end.shape[0]]))])
        # return result

    def encode(
        self, text, begin_index, end_index, diffusion_steps=5, embedding_scale=1
    ):
        # print("INDICES", begin_index, end_index)
        tokens = text_clean(text)
        # tokens.insert(0, 0)
        tokens.insert(begin_index, 0)
        tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)
        # begin_index += 1
        end_index += 1

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
            text_mask = length_to_mask(input_lengths).to(tokens.device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            current_style = self.sampler(
                self.noise,
                embedding=bert_dur[0].unsqueeze(0),
                num_steps=diffusion_steps,
                embedding_scale=embedding_scale,
            ).squeeze(0)

            prev_style = []
            if self.s_prev is not None:
                prev_style.append(self.s_prev)

            style = combine_styles(current_style, prev_style, 0.3)
            # style = combine_styles(current_style,
            #                       next_style,
            #                       0.1)
            # style = next_style[0]
            self.s_prev = current_style

            s = style[:, 128:]
            d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            begin_mel = None
            end_mel = None
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                if i == begin_index:
                    begin_mel = c_frame
                if i == end_index:
                    end_mel = c_frame
                pred_aln_trg[i, c_frame : c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)
            if end_mel is None:
                end_mel = c_frame
            pad_duration = int(pred_dur[begin_index].data)
            used_duration = end_mel - begin_mel - pad_duration
            # print("PRED_DUR [" + text[begin_index] + "][" + text[end_index-1-1] + "]", pred_dur[begin_index], pred_dur[end_index - 1], pred_dur[begin_index:end_index], text[begin_index:end_index-1])
            # encode prosody
            en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device)
            s = style[:, 128:]
            ref = style[:, :128]
            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)
            # print(F0_pred.shape, N_pred.shape)
            # print("MELS", begin_mel, end_mel)
            F0_pred = F0_pred[:, begin_mel * 2 : end_mel * 2]
            N_pred = N_pred[:, begin_mel * 2 : end_mel * 2]
            t_en = t_en[:, :, begin_index:end_index]
            pred_aln_trg = pred_aln_trg[begin_index:end_index, begin_mel:end_mel]
            # print(F0_pred.shape, N_pred.shape, t_en.shape, pred_aln_trg.shape)

            asr = t_en @ pred_aln_trg.unsqueeze(0).to(device)
            # print("ASR", asr.shape)
            out = self.model.decoder(
                (t_en @ pred_aln_trg.unsqueeze(0).to(device)),
                F0_pred,
                N_pred,
                ref.squeeze().unsqueeze(0),
            )

        # return asr, F0_pred, N_pred, ref
        return (
            out.squeeze().cpu(),
            used_duration,
            pad_duration,
        )  # .numpy()#[begin_mel*2*300:end_mel*2*300]
        #        , int(pred_dur[begin_index].data), int(pred_dur[end_index - 1].data))
        # return (en[:, begin_mel:end_mel], style, t_en, pred_aln_trg)

    def decode(self, encoding, style, t_en, pred_aln_trg):
        # with torch.no_grad():
        pass

    @torch.no_grad()
    def sbert_infer(self, phonemes, text):
        if self.sentence_model is None:
            self.sentence_model = SentenceTransformer("all-mpnet-base-v2")
            self.style_mapper = StyleModel(filename="style_mapper.pth")
            self.prosody_mapper = StyleModel(filename="prosody_mapper.pth")
        tokens = text_clean(phonemes)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
        text_mask = length_to_mask(input_lengths).to(tokens.device)

        t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

        # style = self.calculate_audio_style(ref_path)
        embedding = torch.from_numpy(self.sentence_model.encode(text))
        s_pred = self.style_mapper.net(embedding).unsqueeze(0)
        p_pred = self.prosody_mapper.net(embedding).unsqueeze(0)
        style = torch.cat([s_pred, p_pred], dim=1)

        s = style[:, 128:]
        ref = style[:, :128]

        d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

        x, _ = self.model.predictor.lstm(d)
        duration = self.model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame : c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # encode prosody
        en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device)
        F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)
        out = self.model.decoder(
            (t_en @ pred_aln_trg.unsqueeze(0).to(device)),
            F0_pred,
            N_pred,
            ref.squeeze().unsqueeze(0),
        )

        return out.squeeze().cpu().numpy()

    @torch.no_grad()
    def reference_infer(self, text, ref_path):
        tokens = text_clean(text)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
        text_mask = length_to_mask(input_lengths).to(tokens.device)

        t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

        style = self.calculate_audio_style(ref_path)

        s = style[:, 128:]
        ref = style[:, :128]

        d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

        x, _ = self.model.predictor.lstm(d)
        duration = self.model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame : c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # encode prosody
        en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device)
        F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)
        sys.stderr.write("E")
        sys.stderr.flush()
        out = self.model.decoder(
            (t_en @ pred_aln_trg.unsqueeze(0).to(device)),
            F0_pred,
            N_pred,
            ref.squeeze().unsqueeze(0),
        )

        return out.squeeze().cpu().numpy()

    def inference(
        self, text, alpha=0.3, diffusion_steps=5, embedding_scale=1, next_text=None
    ):
        tokens = text_clean(text + " ")
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
            text_mask = length_to_mask(input_lengths).to(tokens.device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            next_style = []
            if next_text is not None:
                next_style.append(
                    self.calculate_style(
                        next_text,
                        diffusion_steps=diffusion_steps,
                        embedding_scale=embedding_scale,
                    )
                )

            current_style = self.sampler(
                self.noise,
                embedding=bert_dur[0].unsqueeze(0),
                num_steps=diffusion_steps,
                embedding_scale=embedding_scale,
            ).squeeze(0)

            prev_style = []
            if self.s_prev is not None:
                prev_style.append(self.s_prev)

            style = combine_styles(current_style, prev_style + next_style, alpha)
            # style = combine_styles(current_style,
            #                       next_style,
            #                       0.1)
            # style = next_style[0]
            self.s_prev = current_style

            print(style.shape)
            s = style[:, 128:]
            ref = style[:, :128]

            d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame : c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device)
            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)
            out = self.model.decoder(
                (t_en @ pred_aln_trg.unsqueeze(0).to(device)),
                F0_pred,
                N_pred,
                ref.squeeze().unsqueeze(0),
            )

        return out.squeeze().cpu().numpy()


def gather_before(i, words, min_size):
    result = ""
    while i > 0 and len(result) < min_size:
        i = i - 1
        result = words[i].strip() + " " + result
    return result


def gather_after(i, words, min_size):
    result = ""
    while i < len(words) - 1 and len(result) < min_size:
        i = i + 1
        result = result + " " + words[i].strip()
    return result


def blend_together(loose_end, start_next):
    diff = loose_end.shape[0] - start_next.shape[0]
    if diff > 0:
        start_next = np.concatenate([np.zeros([diff]), start_next])
    elif diff < 0:
        loose_end = np.concatenate([loose_end, np.zeros([-diff])])
    weight = np.linspace(0, 1, loose_end.shape[0], dtype=loose_end.dtype)
    return weight * start_next + (1 - weight) * loose_end
    # for i in range(300):
    #    weight_a = i/299
    #    weight_b = 1 - weight_a
    #    a[-i] = a[-i]*weight_a + b[300-i-1]*weight_b
    # return np.concatenate([a, b[300:]])


def combine_styles(current, others, alpha):
    result = current
    if len(others) > 0:
        result = current * alpha
    for item in others:
        result += item * ((1 - alpha) / len(others))
    return result
