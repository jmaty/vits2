#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path as osp
from argparse import ArgumentParser, RawTextHelpFormatter

import torch
from scipy.io.wavfile import write

import commons
import utils
from models import SynthesizerTrn
from text import cleaned_text_to_sequence


def get_text(text, text_symbols, add_blank=False):
    text = cleaned_text_to_sequence(text, text_symbols)
    if add_blank:
        text = commons.intersperse(text, 0)
    text = torch.LongTensor(text)
    return text

def main():
    # pylint: disable=bad-option-value
    parser = ArgumentParser(
        description="""VITS2 text-to-speech.\n\n
        """,
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "text_file",
        default=None,
        help="input text file")
    # Input checkpoint
    parser.add_argument(
        "model_path",
        default=None,
        help="model")
    # Input config
    parser.add_argument(
        "-c", "--config_path",
        default=None,
        help="config")
    # Output directory
    parser.add_argument(
        "-o", "--out_dir",
        default="./",
        help="output directory")
    # Output sentence wab'v filename
    parser.add_argument(
        "-u", "--utt_name",
        type=str,
        default="utt",
        help="output sentence wav filename")
    args = parser.parse_args()

    hps = utils.get_hparams_from_file(args.config_path)
    text_symbols = [hps["data"]["pad"]] + list(hps["data"]["punctuation"]) \
                    + list(hps["data"]["characters"])

    spec_channels = hps["data"]["filter_length"] // 2 + 1

    if "use_mel_posterior_encoder" in hps["model"]:
        spec_channels = 80 if hps["model"]["use_mel_posterior_encoder"] \
                        else hps["data"]["filter_length"] // 2 + 1

    # Init synthesizer
    net_g = SynthesizerTrn(
        len(text_symbols),
        spec_channels,
        hps["train"]["segment_size"] // hps["data"]["hop_length"],
        **hps["model"]).cuda()
    _ = net_g.eval()

    # Load checkpoint
    _ = utils.load_checkpoint(args.model_path, net_g, None)

    # Synthesize phonetic text file
    with open(args.text_file, 'rt', encoding='utf-8') as fr:
        texts = list(map(str.strip, fr.readlines()))
        n_digits = len(str(len(texts)))
    with torch.no_grad():
        for idx, t in enumerate(texts):
            t_tst = get_text(t, text_symbols, hps["data"]["add_blank"])
            x_tst = t_tst.cuda().unsqueeze(0)
            x_tst_lengths = torch.LongTensor([t_tst.size(0)]).cuda()
            audio = net_g.infer(x_tst,
                                x_tst_lengths,
                                noise_scale=.667,
                                noise_scale_w=0.8,
                                length_scale=1)[0][0,0].data.cpu().float().numpy()

            write(data=audio,
                  rate=hps["data"]["sampling_rate"],
                  filename=osp.join(args.out_dir, f"{args.utt_name}{(idx+1):0{n_digits}}.wav"))

if __name__ == "__main__":
    main()
