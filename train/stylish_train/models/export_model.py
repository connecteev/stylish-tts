import torch


class ExportModel(torch.nn.Module):
    def __init__(
        self,
        *,
        text_encoder,
        text_duration_encoder,
        textual_style_encoder,
        textual_prosody_encoder,
        textual_pitch_encoder,
        textual_energy_encoder,
        duration_predictor,
        pitch_energy_predictor,
        decoder,
        generator,
        device="cuda",
        **kwargs
    ):
        super(ExportModel, self).__init__()

        for model in [
            text_encoder,
            text_duration_encoder,
            textual_style_encoder,
            textual_prosody_encoder,
            textual_pitch_encoder,
            textual_energy_encoder,
            duration_predictor,
            pitch_energy_predictor,
            decoder,
            generator,
        ]:
            model.to(device).eval()
            for p in model.parameters():
                p.requires_grad = False

        self.device = device
        self.text_encoder = text_encoder
        self.text_duration_encoder = text_duration_encoder
        self.textual_style_encoder = textual_style_encoder
        self.textual_prosody_encoder = textual_prosody_encoder
        self.textual_pitch_encoder = textual_pitch_encoder
        self.textual_energy_encoder = textual_energy_encoder
        self.duration_predictor = duration_predictor
        self.pitch_energy_predictor = pitch_energy_predictor
        self.decoder = decoder
        self.generator = generator

    def decoding_single(
        self,
        text_encoding,
        duration,
        pitch,
        energy,
        style,
    ):
        style = style @ duration
        mel, _ = self.decoder(
            text_encoding @ duration, pitch, energy, style, probing=False
        )
        prediction = self.generator(mel=mel, style=style, pitch=pitch, energy=energy)
        return prediction

    def duration_predict(self, duration_encoding, prosody_embedding):
        d = self.duration_predictor.text_encoder.infer(
            duration_encoding, prosody_embedding
        )
        x, _ = self.duration_predictor.lstm(d)
        duration = self.duration_predictor.duration_proj(x)
        # duration = torch.sigmoid(duration).sum(axis=-1)
        duration = torch.exp(duration)

        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        # pred_dur = torch.ceil(duration).long().squeeze()
        indices = torch.repeat_interleave(
            torch.arange(duration_encoding.shape[2], device=self.device), pred_dur
        )
        pred_aln_trg = torch.zeros(
            (duration_encoding.shape[2], indices.shape[0]), device=self.device
        )
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0).to(self.device)

        prosody = d.permute(0, 2, 1) @ pred_aln_trg
        return pred_aln_trg, prosody

    def forward(self, texts, text_lengths):
        text_encoding, _, _ = self.text_encoder(texts, text_lengths)
        duration_encoding, _, _ = self.text_duration_encoder(texts, text_lengths)
        style_embedding = self.textual_style_encoder(text_encoding)
        prosody_embedding = self.textual_prosody_encoder(duration_encoding)
        duration_prediction, prosody = self.duration_predict(
            duration_encoding,
            prosody_embedding,
        )
        pitch_embedding = (
            self.textual_pitch_encoder(duration_encoding) @ duration_prediction
        )
        energy_embedding = (
            self.textual_energy_encoder(duration_encoding) @ duration_prediction
        )
        # prosody_embedding = prosody_embedding @ duration_prediction
        pitch_prediction, energy_prediction = self.pitch_energy_predictor(
            prosody, pitch_embedding, energy_embedding
        )
        prediction = self.decoding_single(
            text_encoding,
            duration_prediction,
            pitch_prediction,
            energy_prediction,
            style_embedding,
        )
        return prediction.audio.squeeze()
