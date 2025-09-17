
# Stylish TTS (Text-To-Speech) System For Model Training
<img src="https://img.icons8.com/?size=512&id=i46MwMdULdEi&format=png" alt="Alt text" width="100">

# Quick Links:
1. [What is Stylish TTS?](#1-what-is-stylish-tts)
    1. [Overview](#11-overview)
    2. [Current Status](#12-current-status)
2. [Getting Started](#2-getting-started)
    1. [Dependencies](#21-dependencies)
    2. [Setup](#22-setup)
    3. [Installation](#23-installation)
3. [Training Your Model](#3-training-your-model)
    1. [Creating your config.yml file](#31-creating-your-configyml-file)
    2. [Preparing Your Dataset](#32-preparing-your-dataset)
    3. [Generating Pitch and Alignment Data](#33-generating-pitch-and-alignment-data)
    4. [Starting a Training Run](#34-starting-a-training-run)
    5. [(Optional) Loading a Checkpoint](#35-optional-loading-a-checkpoint)
    6. [Exporting to ONNX (for Deployment and Inference)](#36-exporting-to-onnx-for-deployment-and-inference)
4. [Other Forms of Model Training](#4-other-forms-of-model-training)
    1. [Training New Languages](#41-training-new-languages)
5. [Roadmap to v1.0 Release](#5-roadmap-to-v10-release)
6. [License](#6-license)
7. [Citations](#7-citations)

# 1. What is Stylish TTS?

### 1.1 Overview
- Stylish TTS is a lightweight, high-performance Text-To-Speech (TTS) system for training TTS models that are suitable for offline local use. It is possibly also the easiest and fastest way to train your own TTS model.
- The architecture was based on [StyleTTS 2](https://github.com/yl4579/StyleTTS2), but has now diverged substantially and has been made more performant.
- Our focus is to give you the ability to train high quality, single-speaker text-to-speech models (rather than zero-shot voice cloning), with the goal of offering consistent text-to-speech results for long-form text and in applications like screen reading.

### 1.2 Current Status
- Stylish TTS is currently in Beta, and our team has tested it on Ubuntu / Linux, Windows and Mac (Apple Silicon)! Stylish TTS is quickly approaching its [v1.0 release](#5-roadmap-to-v10-release), but it is ready for you to try out now!


# 2. Getting Started

### 2.1 Dependencies:
In order to train your model, you will need:
- a GPU (or a CPU and plenty of time) with PyTorch support
- (Nice-to-have) the NVIDIA / CUDA drivers / system
- a large(ish) [Dataset](#32-preparing-your-dataset)


### 2.2 Setup:

| Step | **Command(s)**
|--------|-----------------------------------------------------|
| **Install 📦 uv or 🐍 pip** <br/><br/><details><summary><b>Expand: Why use 📦 uv over 🐍 pip?</b></summary>⚡ 10-100x faster installation<br/>🔒 Better dependency resolution<br/>🐍 Automatic Python management<br/>🎯 Drop-in pip replacement<br/></details>| <details><summary><b>Expand: how to set up 📦 uv</b></summary>- `pipx install uv` # Installs uv if you don't have it already<br/>- `pipx ensurepath` # Needed to use uv from the command line <br/>- Remember:<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Do not run `uv` inside another virtual environment.<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- `uv run <COMMAND>` will always also update the pyproject.toml and related project files.<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- `uv run python --version` is equivalent to `python --version`, except it is run within the uv virtual environment.<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- `uv run <COMMAND>` is similar to: `source .venv; <COMMAND>; exit`<br/></details><details><summary><b>Expand: how to set up 🐍 pip</b></summary>- `curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py`  # Downloads the installer<br/>- # or use wget: `wget https://bootstrap.pypa.io/get-pip.py`<br/>- `python get-pip.py` # Installs pip</details>|
| **Install python 3.12** <br/>stylish-tts depends on python >= 3.12|- `uv python install 3.12 --preview --default` # Installs python 3.12 (stylish-tts depends on python >= 3.12)<br/>- `uv run python --version` # Verify that the python version is 3.12.x<br/><br/><details><summary><b>Expand: how to use with 🐍 pip</b></summary>- `pyenv install 3.12.7 && pyenv local 3.12.7` # Installs python 3.12 <br/>- `brew install python@3.12` # Install python 3.12 via Homebrew (on Mac)<br/>- `python --version` # Verify that the python version is 3.12.x</details>|
| **Remove lock file**|- `rm uv.lock` # Remove lock file|
| **Set up new empty project** |- `mkdir my_stylish_tts_model_training`<br/>- `cd my_stylish_tts_model_training` |
| **Create & activate new virtual env with python 3.12** |- `uv init` # will create pyproject.toml, main.py, & supporting files<br/><br/><details><summary><b>Expand: how to use with 🐍 pip</b></summary>- `python -m venv venv_py312`<br/>- `source venv_py312/bin/activate`<br/><br/><b>Note:</b><br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Make sure to activate your virtual environment (`source venv_py312/bin/activate` on Linux/Mac or `venv_py312\Scripts\activate` on Windows) before running any Python commands<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- The virtual environment needs to be activated each time you start a new terminal session to work on this project<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- You can deactivate the virtual environment by running `deactivate` when you're done working<br/></details>|
| **Install dependencies** |- `uv add torch torchaudio onnxruntime` <br/>Note: Use `onnxruntime` if you cannot install `onnxruntime-gpu` (one of these will be needed later when using the ONNX model for inference)<br/><br/><details><summary><b>Expand: how to use with 🐍 pip</b></summary>`pip install torch torchaudio onnxruntime` <br/>Note: Use `onnxruntime` if you cannot install `onnxruntime-gpu` (one of these will be needed later when using the ONNX model for inference)<br/></details>|
| **Install k2** |<details><summary><b>Expand: Find the right k2 version (for your OS, torch and python version)</b></summary>- First, find the appropriate k2 version (for your torch version, like 2.8.0 and cpython version, like 3.12) for your platform from https://huggingface.co/csukuangfj/k2/tree/main (there are subfolders for different platforms: cpu, cuda, macos, ubuntu-cuda, windows-cpu).<br/>- For example, if your platform is MacOS / Apple Silicon, you have cpython 3.12 and torch 2.8.0 installed, then the correct version of k2 would be: https://huggingface.co/csukuangfj/k2/resolve/main/macos/k2-1.24.4.dev20250807%2Bcpu.torch2.8.0-cp312-cp312-macosx_11_0_arm64.whl</details><details><summary><b>Expand: Download k2</b></summary>- `mkdir k2_installation_files`<br/>- `curl -L -o "k2_installation_files/k2-1.24.4.dev20250807+cpu.torch2.8.0-cp312-cp312-macosx_11_0_arm64.whl" "https://huggingface.co/csukuangfj/k2/resolve/main/macos/k2-1.24.4.dev20250807%2Bcpu.torch2.8.0-cp312-cp312-macosx_11_0_arm64.whl"`<br/><br/><b>Tips to prevent the above CURL command from failing...</b><br/>- In the first -o argument for the k2 file path (in this example, using `k2_installation_files/`), use `+` instead of `%2B`<br/>- In the second argument for the HuggingFace URL, use `%2B` and make sure the URL has `resolve/main` instead of `blob/main`</details><details><summary><b>Expand: Enable k2</b></summary>To enable k2, update pyproject.toml with the following:<br/>```requires-python = ">=3.12"```<br/>```dependencies = [```<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;```"k2",```<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;```"torch>=2.8.0",```<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;```"torchaudio>=2.8.0",```<br/>```]```<br/>```[tool.uv.sources]```<br/>```k2 = { path = "k2_installation_files/k2-1.24.4.dev20250807+cpu.torch2.8.0-cp312-cp312-macosx_11_0_arm64.whl" }```<br/><br/><b>Run:</b><br/>- `uv sync`<br/>- `uv run python --version`     # Verify that the python version is 3.12.x and it still works with k2 installed<br/>- `uv run python -c "import k2; print('k2 installed successfully')"` # Verify that k2 was installed successfully<br/><br/><b>How to use with 🐍 pip:</b><br/>- `pip install k2_installation_files/k2-1.24.4.dev20250807+cpu.torch2.8.0-cp312-cp312-macosx_11_0_arm64.whl` # Install k2 from the downloaded wheel<br/>- `python --version` # Verify that the python version is 3.12.x and it still works with k2 installed<br/>- `python -c "import k2; print('k2 installed successfully')"` # Verify that k2 was installed successfully<br/></details><details><summary><b>Expand: Troubleshooting k2 installation issues...</b></summary>- If `uv run python --version` fails because of k2, first remove k2, verify python version still works, and then try the k2 installation again with the proper k2 version from HuggingFace:<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- `uv remove k2` # remove k2, or manually remove ALL references to k2 in pyproject.toml<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- `uv run python --version` # Verify that the python version is 3.12.x<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Then re-add the correct version of k2 from HuggingFace<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- `uv run python --version` # Verify that the python version is 3.12.x with k2 installed<br/><br/><b>How to use with 🐍 pip:</b><br/>- If `python --version` fails because of k2, first uninstall k2, verify python version still works, and then try the k2 installation again with the proper k2 version from HuggingFace:<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- `pip uninstall k2` # remove k2<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- `python --version` # Verify that the python version is 3.12.x and it still works<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Then re-install the correct version of k2 from HuggingFace<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- `python --version` # Verify that the python version is 3.12.x and it still works with k2 installed<br/><br/></details>|


### 2.3 Installation:

| Step | **Command(s)**
|--------|-----------------------------------------------------|
| **Clone stylish-tts Repo**|- From your `my_stylish_tts_model_training` directory, run:<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- `git clone https://github.com/Stylish-TTS/stylish-tts.git`    # Clone the default branch of the style-tts repository (recommended)<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- OR<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- `git clone -b mactest https://github.com/Stylish-TTS/stylish-tts.git`  # Clone the style-tts repository, checkout the mactest branch instead of the default branch, and set mactest as your current working branch|
| **Install stylish-tts**|- `uv add --editable stylish-tts/`   # Installs `stylish-tts` as a local editable package from the `stylish-tts/` directory, also automatically rebuilds it if its contents change. Important: Don't forget the trailing slash `/`<br/><br/><details><summary><b>Expand: how to use with 🐍 pip</b></summary>- `pip install -e stylish-tts/`   # Installs `stylish-tts` as a local editable package from the `stylish-tts/` directory, also automatically rebuilds it if its contents change. Important: Don't forget the trailing slash `/`</details>|


# 3. Training Your Model

### 3.1 Creating your config.yml file
- You will need your own `config.yml` file (say `my_config.yml`) created from the template [here](https://github.com/Stylish-TTS/stylish-tts/blob/main/config/config.yml). You can store it anywhere, like at the root of your project.
- In your `my_config.yml` file, you will need to specify the device ("cuda", "mps", "cpu" or whatever will work with your torch installation). For example:
  ```
  training:
    device: "mps"
  ```

### 3.2 Preparing Your Dataset

- Note: A sample dataset can be found at [sample_dataset/](sample_dataset/). Please note that this has been provided as a reference only, and will in no way be sufficient to train a model.
- You will also need to fill in the `dataset` section of your `my_config.yml` file, as shown below. The `path` should be the root of your dataset, and the various other paths in this section are relative to that root. If you use the default file and directory names, your directory structure will look something like this:
  ```
  dataset:
    # All paths in this section are relative to the main path
    path: "../my_dataset/"
    train_data: "training-list.txt"
    val_data: "validation-list.txt"
    wav_path: "wav-dir"
    pitch_path: "pitch.safetensors"
    alignment_path: "alignment.safetensors"
    alignment_model_path: "alignment_model.safetensors"
  ```

- Your structure of your dataset folder might look something like this:
  ```
  path/to/your/dataset/             # Root
  |
  +-> training-list.txt             # Training list file (described below)
  |
  +-> validation-list.txt           # Validation list file (described below)
  |
  +-> wav-dir                       # Folder with audio wav files, one for each segment
  |   |
  |   +-> something.wav
  |   |
  |   +-> other.wav
  |   |
  |   +-> ...
  |
  +-> pitch.safetensors             # Pre-cached segment pitches (file gets generated at `pitch_path` in `my_config.yml`. See below)
  |
  +-> alignment.safetensors         # Pre-cached alignments (file gets generated at `alignment_path` in `my_config.yml`. See below)
  |
  +-> alignment_model.safetensors   # Model for generating alignments. You will train this (gets generated at `alignment_model_path`  in `my_config.yml`. See below)

  ```
- Note: A sample dataset can be found at [sample_dataset/](sample_dataset/). Please note that this has been provided as a reference only, and will in no way be sufficient to train a model.

- A dataset consists of many segments. Each segment has a written text and an audio file where that text is spoken by a reader.
- Your dataset should be in the following format:
  - your Training List (corresponding to train_data in `my_config.yml`)
  - your Validation List (corresponding to val_data in `my_config.yml`)
  - your Folder with audio wav files (resampled at 24 khz, mono), one for each segment (corresponding to wav_path in `my_config.yml`)

- Segment Distribution:
  - Segments must have 510 phonemes or less.
  - Audio segments must be at least 0.25 seconds long.
  - The upper limit on audio length is determined by your VRAM and the training stage. If you have enough VRAM, you can include even longer segments, though there is an upper limit (because of diminishing returns).
  - Generally speaking, you will want to have a distribution of segments between 0.25 seconds and 10 seconds long.
    - If your range doesn't cover the shortest lengths, your model will sound worse when doing short utterances of one word or a few words.
    - If your range doesn't cover longer lengths which include multiple sentences, your model will tend to skip past punctuation too quickly.

- Folder with audio wav files, one for each segment:
  - Each segment audio should be a .wav file (resampled at 24 khz, mono) in the wav_path folder specified in your `my_config.yml`.

- Training List and Validation List:
  - Training and Validation lists are a series of lines in the following format: `<filename>|<phonemes>|<speaker-id>|<plaintext>`
  - Once you have created a new Training List file, manually take ~1% of the entries from that file, and move it into your new Validation List file.
  - Examples of entries in the Training and Validation lists:
      - `1.wav|ɔnðə kˈɑːntɹɛɹi|0|On the contrary`
      - `2.wav|fɚðə fˈɜːst tˈaɪm|0|For the first time`
  - The filename is the name of the file for the segment audio. It should be a .wav file (24 khz, mono) in the wav_path folder specified in your `my_config.yml`.
  - The phonemes are the IPA representation of how your segment text is pronounced. You may use `espeak-ng` (or a similar G2P system) to create phonemes corresponding to each audio file.
  - Speaker ID is an arbitrary integer which should be applied to every segment that has the same speaker. For single-speaker datasets, this will typically always be '0'.
  - The plaintext is the original text transcript of your utterance before phonemization. It does not need to be tokenized or normalized, but obviously should not include the '|' character, which is to be used as the separator.


### 3.3 Generating Pitch and Alignment Data

- Pitch Data
  - Stylish TTS uses a pre-cached ground truth pitch (F0) for all your segments. To generate these pitches, run:
    ```
    uv run stylish-train pitch /path/to/your/config.yml --workers 16
    ```
  - The number of workers should be approximately equal to the number of cores on your machine. By default, Harvest, which is a CPU-based system, is used to extract pitch. If you find this to be too slow, there is also a GPU-based option available by passing `--method rmvpe` from the command line. When finished, it will write the pre-cached segment pitches at the `pitch_path` file path specified by your `my_config.yml`.


- Alignment Data
  - Alignment data is also pre-cached, and in order to generate the pre-cached data, you will need to train an alignment model first. This is a multi-step process, but only needs to be done ONCE for your dataset, after which you can just use the cached results (similar to the generated pitch data).
  - First, you run train.py using the special alignment stage. For a description of the other parameters, see below.
    ```
    uv run stylish-train train-align /path/to/your/config.yml --out /path/to/your/output
    ```
    - The `--out` option is where logs and checkpoints will end up. Once the alignment stage completes, it will provide a trained model at the file specified in your `my_config.yml`. It is important to realize that this is a MODEL, not the alignments themselves. We will use this model to generate the alignments.

    ```
    uv run stylish-train align /path/to/your/config.yml
    ```
    - This generates the actual cached alignments for all the segments for both training and validation data as configured in your config.yml. You should now add the resulting alignment.safetensors path to your `my_config.yml`.


    <details>
    <summary>(Expand to read) Expectations during alignment pre-training</summary>

    - Expectations during alignment pre-training:
      - In this stage, a special adjustment is made to the training parameters at the end of each epoch.
      - This adjustment means there will be a discontinuity in the training curve between epochs. This adjustment will eventually make the loss turn NEGATIVE. This is normal. If your training align_loss does not eventually turn negative, that is a sign that you likely need to train more.
      - At each validation step, both an un-adjusted align_loss and a confidence score are generated. 
        - align_loss should be going down.
        - Confidence score should be going up.
        - You want to pick a number of epochs so that these scores reach the knee in their curve. Do not keep training forever just because they are slowly going down. If you run into issues where things are not converging later, it is likely that you need to come back to this step and train a different amount to hit that "knee" in the loss curve.
      - During alignment pre-training, we ALSO train on the validation set. This is usually a very, very bad thing in Machine Learning (ML). But in this case, the alignment model will never be used for aligning out-of-distribution segments. Doing this gives us a more representative sample for acoustic and textual training and does not have any other effects on overall training.
    </details>

    <details>
    <summary>(Expand to read) OPTIONAL: Culling Bad Alignments</summary>

    - OPTIONAL: Culling Bad Alignments
      - Running `stylish-tts align` generates a "confidence value" score for every segment it processes. These scores are written to files in your dataset `path`.
      - Confidence is not a guarantee of accuracy, because the model could be confidently wrong. But it is a safe bet that the segments that it is the least confident about either:
        - have a problem (perhaps the text doesn't match the audio) or
        - are just a bad fit for the model's heuristics.
      - Culling the segments with the least confidence will make your model converge faster, though it also means it will "see" less training data.
      - Anecdotally, we have found that culling the 10% with the lowest confidence scores is a good balance.
    </details>

- Note: All of the commands above (for Pitch and Alignment) should only need to be done ONCE per dataset, as long as the dataset does not change. Once they are done, their results are kept in your dataset directory. Now we begin ACTUALLY training.


### 3.4 Starting a Training Run

- Here is a typical command to start off a new training run using a single machine:
  ```
  uv run stylish-train train /path/to/your/config.yml --out /path/to/your/output
  ```
  --out: This is the destination path for all checkpoints, training logs, and tensorboard data. A separate sub-directory is created for each stage of training. Make sure to have plenty of disk space available here as checkpoints can take a large amount of storage.

- Expectations During Training
  - It will take a LONG time to run this script. So, it is a good idea to run using `screen` or `tmux` to have a persistent shell that won't disappear if you get disconnected or close the window.
  - Training happens over the course of four stages:
    - The four main stages of training are `acoustic`, `textual`, `style`, and `duration`. 
    - Each stage has its own logs and tensorboard data in a separate subdirectory of the `out_dir`.
    - When you begin training, it will start with the `acoustic` stage by default.
    - As each stage ends, the next will automatically begin.
    - You can specify a stage with the `--stage` option, which is necessary if you are resuming from a checkpoint.
  - Stages advance automatically and a checkpoint is created at the end of every stage before moving to the next. Other checkpoints will be saved and validations will be periodically run based on your `my_config.yml` settings.
  - Each stage will have its own sub-directory of `out`, and its own training log and tensorboard graphs/samples.

  <details>
  <summary>(Expand to read) Expectations During Each of the 4 Training Phases</summary>

    Expectations During Each of the 4 Training Phases:
    - Stage 1: Acoustic training
      - Acoustic training is about training the fundamental acoustic speech prediction models which feed into the vocoder. We 'cheat' by feeding these models parameters derived directly from the audio segments. The pitch, energy, and alignments all come from our target audio. Pitch and energy are still being trained here, but they are not being used to generate predicted audio.
      - The main loss figure to look at is `mel` which is a perceptual similarity of the generated audio to the ground truth. It should slowly decrease during training, but the exact point at which it converges will depend on your dataset. The other loss figures can generally be ignored and may not vary much during training.
      - By the end of acoustic training, the samples should sound almost identical to ground-truth. These are probably going to be the best-sounding samples you listen to. But of course this is because it is doing the easiest version of the task.

    - Stage 2: Textual training
      - In textual training, the acoustic speech prediction is frozen while the focus of training becomes pitch and energy. An acoustic style model still 'cheats' by using audio to generate a prosodic style. This style along with the base text are what is used to calculate the pitch and energy values for each time location.
      - Here, `mel`, `pitch`, and `energy` losses are all important. You should expect mel loss to always be much higher in this stage than the acoustic stage. And it will only very gradually go down. Since there are three losses here, keeping an eye on total loss is more useful. It will be a lot less stable than in acoustic, but there is still a clear trend downwards.
      - As training goes on, the voice should sound less strained, less 'warbly', and more natural. Make sure you are listening for the tone of the sound and how loud it is rather than strict prosody because the samples are still using the ground truth alignment.

    - Stage 3: Style training
      - Here the only 'cheating' we do is to use the ground-truth alignment. The predicted pitch and energy are used to directly predict the audio. A textual style encoder is trained to produce the same outputs as the acoustic model from the previous stage.
      - Aside from that, the training regimen should look a lot like the previous stage. `mel`, `pitch`, and `energy` should all trend downward but expect `mel` to be higher than the previous stage.

    - Stage 4: Duration training
      - The final stage of training removes our last 'cheat' and trains the duration predictor to try to replicate the prosody of the original. The other models are frozen. All samples use only values predicted from the text.
      - The `duration` and `duration_ce` losses should both slowly go down. The main danger here is overfitting. So if you see validation loss stagnate or start going up you should stop training even if training loss is still going down. It is expected that one of the losses might plateau before the other.
      - When you listen to samples, you will get the same version you'd expect to hear during inference. Listen to make sure the voice as a whole is not going to fast or slow or just going past punctuation without pausing. You should no longer expect it to mirror the ground truth exactly, but it should have generalized to the point where it is a plausible and expressive reading. As training proceeds, it should sound more and more like fluent prosody. If there are still pitch or energy issues like warbles or loudness or tone, then those won't be fixed in this stage and you may need to train more in Textual or Acoustic before trying Duration training.

  </details>


### 3.5 (Optional) Loading a Checkpoint

  <details>
  <summary>(Expand to read) What is a Checkpoint?</summary>

  What is a Checkpoint?
  - A checkpoint is a snapshot of a model's state during training that can be used to resume training or for inference.
  - It is essentially a save file that captures the model at a specific point in time.
  - Checkpoints are training-centric and framework-specific.
  - A checkpoint typically contains:
    - Model weights/parameters - the learned values from training
    - Optimizer state - momentum, learning rate schedules, etc.
    - Training metadata - current epoch, step count, loss values
    - Model architecture info - though sometimes stored separately
    - Random number generator states - for reproducible training resumption

  - Checkpoints serve multiple purposes:
    - Recovery: Resume training if interrupted
    - Experimentation: Compare models at different training stages
    - Deployment: Use a trained model for inference
    - Fine-tuning: Start from a pre-trained state for additional training
  - ---
  </details>

- You can load a checkpoint from any stage via the `--checkpoint` argument.
- You still need to set `--stage` appropriately to one of "alignment|acoustic|textual|duration".
  - If you set it to the same stage as the checkpoint loaded from, it will continue in that stage at the same step number and epoch.
  - If it is a different stage, it will train the entire stage.
- To load a checkpoint:
  ```
  uv run stylish-train train /path/to/your/config.yml --stage <stage> --out /path/to/your/output --checkpoint /path/to/your/checkpoint
  ```
- Please note that Stylish TTS checkpoints are NOT compatible with StyleTTS 2 checkpoints.


### 3.6 Exporting to ONNX (for deployment and inference)

  <details>
  <summary>(Expand to read) What is an ONNX file?</summary>

What is an ONNX file?
- ONNX (Open Neural Network Exchange) is an open standard format for representing machine learning models.
- ONNX files are primarily used for deployment and inference rather than training. They're optimized for running the model efficiently in production environments and enable interoperability between different ML ecosystems.
- ONNX files are deployment-centric and framework-agnostic. An ONNX file contains a serialized neural network that can be moved between different ML frameworks like PyTorch, TensorFlow, or specialized inference engines.
- In the context of LLMs specifically, an ONNX file would contain:
  - The complete model architecture (layers, connections, operations)
  - All trained weights and parameters
  - Input/output specifications
  - Metadata about the model
- ---
  </details>

- Now, we will export two ONNX files, which will be used for deployment and inference.
- This command will export two ONNX files, one for predicting duration and the other for predicting speech.

  ```sh
  uv run stylish-train convert /path/to/your/config.yml --duration /path/to/your/duration.onnx --speech /path/to/your/speech.onnx --checkpoint /path/to/your/checkpoint
  ```

- Using the ONNX model for Inference:
  ```sh
  uv run stylish-tts/train/test_onnx.py --duration /path/to/your/output/duration.onnx --speech /path/to/your/output/speech.onnx \
      --text "ðˈiːz wˈɜː tˈuː hˈæv ˈæn ɪnˈɔːɹməs ˈɪmpækt , nˈɑːt ˈoʊnliː bɪkˈɔz ðˈeɪ wˈɜː əsˈoʊsiːˌeɪtᵻd wˈɪð kˈɑːnstəntˌiːn ," \
      --text "bˈʌt ˈɔlsoʊ bɪkˈɔz , ˈæz ɪn sˈoʊ mˈɛniː ˈʌðɚ ˈɛɹiːəz , ðə dɪsˈɪʒənz tˈeɪkən bˈaɪ kˈɑːnstəntˌiːn ( ˈɔːɹ ɪn hˈɪz nˈeɪm ) wˈɜː tˈuː hˈæv ɡɹˈeɪt səɡnˈɪfɪkəns fˈɔːɹ sˈɛntʃɚiːz tˈuː kˈʌm ." \
      --combine true
  ```
  The `text` parameters in the above example have content as phonemes, corresponding to the following transcript:
    - "These were to have an enormous impact, not only because they were associated with Constantine, but also because, as in so many other areas, the decisions taken by Constantine (or in his name) were to have great significance for centuries to come."


# 4. Other Forms of Model Training

### 4.1 Training New Languages

- Grapheme to Phoneme (G2P)
  - Grapheme-to-phoneme conversion (G2P) is the task of transducing graphemes (i.e. text or letter(s) that spells a sound) to phonemes (i.e. the sound).
  - Each language has its own phonetic rules, and therefore, requires a distinct G2P system. Accurate G2P is critical for the performance of text-to-speech (TTS).
  - The most effective G2P systems are typically tailored to specific languages. These can often be found in research papers focused on phonetics or TTS—try searching for terms like "[language] G2P/TTS site:arxiv.org" or "[language] G2P site:github.com". Libraries such as [misaki](https://github.com/hexgrad/misaki/) may also provide such G2P systems.
  - A commonly used multilingual G2P system is `espeak-ng`, though its accuracy can vary depending on the language. In some cases, a simple approach - using word-to-phoneme mappings from sources like Wiktionary - can be sufficient.

- Adjust model.yml

    <details>
    <summary>(Expand to read) What is model.yml used for?</summary>

    What is model.yml used for?
    - [model.yml](src/stylish_tts/train/config/model.yml) holds the hyperparameters to the model.
    - Most of the time, you will use this by default. However, in rare circumstances, it can be edited, say, if you want to experiment with different options or need to change a specific aspect of the model.
    - ---
    </details>

  - If the G2P don't share the same phonetic symbol set in `model.yml`, change the `symbol` section and `text_encoder.tokens`.
  - `text_encoder.tokens` should be equal to length of `symbol.pad` + `symbol.punctuation` + `symbol.letters` + `symbol.letters_ipa`
    ```
    ...
    text_encoder:
      tokens: 178 # number of phoneme tokens
      hidden_dim: 192
      filter_channels: 768
      heads: 2
      layers: 6
      kernel_size: 3
      dropout: 0.1

    ...

    symbol:
      pad: "$"
      punctuation: ";:,.!?¡¿—…\"()“” "
      letters: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
      letters_ipa: "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁᵊǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

    ```


# 5. Roadmap to v1.0 Release
Pending tasks:
- [x] Rework the CLI (Command Line Interface)
- [x] Merge disc-opt into main
- [x] Import pitch cache script and make it use a concurrent.futures worker pool
- [x] Do proper stage detection in dataloader to prevent mixups with precached alignment/pitch
- [ ] Verify final model architecture
- [ ] Verify ONNX conversion
- [ ] Make sure it can work as a PyPi package
- [ ] Replace checkpointing with safetensors instead of accelerator checkpoint
- [ ] Remove dependency on accelerator
- [ ] Audit dependencies
- [ ] Audit and fix any remaining torch warnings
- [ ] Move test_onnx to stylish-tts module and remake it into at least a barebones inferencer.
- [ ] Update this README with Sample / Demo audio clips


# 6. License
- All original code in this repository is <b>MIT-licensed.</b>
- Most code from other sources is <b>MIT-licensed.</b>
- A BSD license is included as a comment for the limited amount of code that was <b>BSD-licensed.</b>


# 7. Citations
<details>
<summary>View Citations</summary>

- The foundation of this work is StyleTTS and StyleTTS 2
  - "StyleTTS: A Style-Based Generative Model for Natural and Diverse Text-to-Speech Synthesis" by Yinghao Aaron Li, Cong Han, Nima Mesgarani [Paper](https://arxiv.org/abs/2205.15439) [Code](https://github.com/yl4579/StyleTTS)
  - "StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models" by Yinghao Aaron Li, Cong Han, Vinay S. Raghavan, Gavin Mischler, Nima Mesgarani [Paper](https://arxiv.org/abs/2306.07691) [Code](https://github.com/yl4579/StyleTTS2)

- Discriminators
  - "Improve GAN-based Neural Vocoder using Truncated Pointwise Relativistic Least Square GAN" by Yanli Li, Congyi Wang [Paper](https://dl.acm.org/doi/abs/10.1145/3573834.3574506)
  - Some code adapted from "Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis" [Paper](https://arxiv.org/pdf/2306.00814.pdf) [Code](https://github.com/gemelo-ai/vocos)
  - Discriminator Regulator: "Mind the (optimality) Gap: A Gap-Aware Learning Rate Scheduler for
  Adversarial Nets" by Hussein Hazimeh, Natalia Ponomareva [Paper](https://arxiv.org/abs/2302.00089) [Code](https://github.com/google-research/google-research/blob/master/adversarial_nets_lr_scheduler/demo.ipynb)
  - Only use MRD discriminator: "GAN Vocoder: Multi-Resolution Discriminator Is All You Need" by Jaeseong You, Dalhyun Kim, Gyuhyeon Nam, Geumbyeol Hwang, Gyeongsu Chae [Paper](https://www.isca-archive.org/interspeech_2021/you21b_interspeech.pdf)

- Text Alignment
  - "Less Peaky and More Accurate CTC Forced Alignment by Label Priors" by Ruizhe Huang, Xiaohui Zhang, Zhaoheng Ni, Li Sun, Moto Hira, Jeff Hwang, Vimal Manohar, Vineel Pratap, Matthew Wiesner, Shinji Watanabe, Daniel Povey, Sanjeev Khudanpur [Paper](https://arxiv.org/abs/2406.02560v3) [Code](https://github.com/huangruizhe/audio/tree/aligner_label_priors/examples/asr/librispeech_alignment)
  - "Evaluating Speech–Phoneme Alignment and Its Impact on Neural Text-To-Speech Synthesis" by Frank Zalkow, Prachi Govalkar, Meinard Müller, Emanuël A. P. Habets, and Christian Dittmar [Paper](https://ieeexplore.ieee.org/document/10097248) [Supplement](https://www.audiolabs-erlangen.de/resources/NLUI/2023-ICASSP-eval-alignment-tts)
  - "Phoneme-to-Audio Alignment with Recurrent Neural Networks for Speaking and Singing Voice" by Yann Teytaut, Axel Roebel [Paper](https://www.isca-archive.org/interspeech_2021/teytaut21_interspeech.html)

- Pitch Extraction
  - "Harvest: A high-performance fundamental frequency estimator from speech signals" by Masanori Morise [Paper](https://www.isca-archive.org/interspeech_2017/morise17b_interspeech.pdf)
  - "RMVPE: A Robust Model for Vocal Pitch Estimation in Polyphonic Music" by Haojie Wei, Xueke Cao, Tangpeng Dan, Yueguo Chen [Paper](https://arxiv.org/abs/2306.15412)

- Text Encoding
  - Taken from "Matcha-TTS: A fast TTS architecture with conditional flow matching", by Shivam Mehta, Ruibo Tu, Jonas Beskow, Éva Székely, and Gustav Eje Henter [Paper](https://arxiv.org/abs/2309.03199) [Code](https://github.com/shivammehta25/Matcha-TTS)
  - Originally from "Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search" by Jaehyeon Kim, Sungwon Kim, Jungil Kong, and Sungroh Yoon [Paper](https://arxiv.org/abs/2005.11129) [Code](https://github.com/jaywalnut310/glow-tts)

- Vocoder is a hybrid model with inspiration from several sources
  <!-- - Backbone: "RingFormer: A Neural Vocoder with Ring Attention and Convolution-Augmented Transformer" by Seongho Hong, Yong-Hoon Choi [Paper](https://arxiv.org/abs/2501.01182) [Code](https://github.com/seongho608/RingFormer) -->
  <!-- - Harmonics Generation: "Neural Source-Filter Waveform Models for Statistical Parametric Speech Synthesis" by Wang, X., Takaki, S. & Yamagishi, J. [Paper](https://ieeexplore.ieee.org/document/8915761) [Code](https://github.com/nii-yamagishilab/project-CURRENNT-scripts/tree/master/waveform-modeling/project-NSF-v2-pretrained) -->
  - "APNet2: High-quality and High-efficiency Neural Vocoder with Direct Prediction of Amplitude and Phase Spectra" by Hui-Peng Du, Ye-Xin Lu, Yang Ai, Zhen-Hua Ling [Paper](https://arxiv.org/abs/2311.11545)
  - "LightVoc: An Upsampling-Free GAN Vocoder Based On Conformer And Inverse Short-time Fourier Transform" by Dinh Son Dang, Tung Lam Nguyen, Bao Thang Ta, Tien Thanh Nguyen, Thi Ngoc Anh Nguyen, Dang Linh Le, Nhat Minh Le, Van Hai Do [Paper](https://www.isca-archive.org/interspeech_2023/dang23b_interspeech.pdf)
  - For phase loss and serial AP architecture (even though we found quality is better with discriminator and also the phase loss): "Is GAN Necessary for Mel-Spectrogram-based Neural Vocoder?" by Hui-Peng Du, Yang Ai, Rui-Chen Zheng, Ye-Xin Lu, Zhen-Hua Ling [Paper](https://arxiv.org/pdf/2508.07711)
  - For anti-wrapping phase loss: "Neural Speech Phase Prediction based on Parallel Estimation Architecture and Anti-Wrapping Losses" by Yang Ai, Zhen-Hua Ling [Paper](https://arxiv.org/abs/2211.15974)
  - Attention code from Conformer implementation by Lucidrains [Code](https://github.com/lucidrains/conformer/blob/fc70d518d3770788d17a5d9799e08d23ad19c525/conformer/conformer.py#L66)

- Duration prediction
  - Ordinal regression loss: "Class Distance Weighted Cross-Entropy Loss for Ulcerative Colitis Severity Estimation" by Gorkem Polat, Ilkay Ergenc, Haluk Tarik Kani, Yesim Ozen Alahdab, Ozlen Atug, Alptekin Temizel [Paper](https://arxiv.org/abs/2202.05167)

- ONNX Compatibility
  - Kokoro [Code](https://github.com/hexgrad/kokoro)
  - Custom STFT Contributed to Kokoro by [Adrian Lyjak](https://github.com/adrianlyjak)
  - Loopless Duration Contributed to Kokoro by [Hexgrad](https://github.com/hexgrad)

</details>
