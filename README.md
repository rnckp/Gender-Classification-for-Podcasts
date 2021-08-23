# How much more do men talk in podcasts than women do?
### Assessing the share of male and female voices in audio media with machine learning

- This notebook makes it easy to **automatically assess the voice share of men and women in batches of MP3 audio files.** 
- It is build for the **analysis of podcast media**. 
- This tool is derived from [this project](https://github.com/rnckp/EPFL-Capstone-Project). For the full context of the problem space have a look at [this notebook](https://github.com/rnckp/EPFL-Capstone-Project/blob/main/01_project_overview.ipynb).
- The tool is **meant to facilitate analysis of gender imbalances in audio media** and **to reduce time consuming manual tracking** for media companies and producers.
- Processing time on a local machine (Mac Mini M1) is around 1:10. So **10 hours of podcast episodes take roughly 1 hour of compute**. Analyzing for example the TOP 50 news podcasts in the German iTunes store (20 last epsiodes each, around 500hrs total playing time) takes roughly 50 hours. Retrieving the audio data from a fast SSD drive speeds up processing.

[Here you can find a larger analysis of around 2'500 hours of podcasts done with this code](http://www.rnck.me/posts/podcast_analysis/).

### How does it work?
- I use a subset of the German CommonVoice data set (where the speaker's biological sex is labeled) as well as 45k podcast audio samples (German/Swissgerman) that I selected and labeled myself (with the help of unsupervised/semisupervised methods like KMeans, UMAP etc.).
- I have gender balanced both data sets to avoid bias toward one biological sex as much as possible. Interestingly the CommonVoice data itself is [heavily gender imbalanced itself](https://github.com/rnckp/EPFL-Capstone-Project/blob/main/02_eda.ipynb). I also filtered the available data towards as much speaker variety as possible.
- From all audio samples I extracted as features: MFCCs, Chromagram and Contrast with librosa and voice embeddings with pyannote.
- I tried various classifiers on the data. A Support Vector Machine classifier yielded the best results (KNN coming in as second best with ~1% less accuracy).
- The SVC achieves a **crossvalidated accuracy of +99%** (evaluated on 10 validation folds of the 45k ground truth podcast samples). The trained model is available in the repository and simply can be loaded and used.
- In addition to the gender classification I added a **voice activity detection (VAD) to remove sections without human utterances**. For the VAD I use a pretrained model from [pyannote](https://www.researchgate.net/publication/337019697_pyannoteaudio_neural_building_blocks_for_speaker_diarization). **I assume that the total margin of error is around +/-5%** (SVC classifier & VAD).
- **The model will work with high accuracy on German / Swissgerman language**. Prediction proved to be quite language agnostic. Accuracy will decrease though.

### Requirements
Apart from the usual Data Science stack of pandas, matplot, seaborn and scikit-learn you need to have installed pyannote ([development branch](https://github.com/pyannote/pyannote-audio/tree/develop)), pytorch, librosa, pydub and ffmpeg.
