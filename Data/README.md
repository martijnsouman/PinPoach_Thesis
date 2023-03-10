# Dataset gather

This folder contains python scripts for:
- Selecting the gunshot files from UrbanSound8K
- Splitting the large background audio files to the desired frame size (savanna by day/night, rain,and thunder).
- Padding shorter audio files with silent sections to match the desired frame size (gunshots).

## Dependencies
Following applications are needed for running this script:
- `python3`
- `youtube-dl`
    - If you encounter the error <Youtube_dl : ERROR : YouTube said: Unable to extract video data>, you probably need to update it.
   See [stackoverflow](https://stackoverflow.com/questions/63816790/youtube-dl-error-youtube-said-unable-to-extract-video-data).
- ``ffmpeg``

Following data needs to be downloaded manually to run this script:
- Background noise from YouTube:
    1. Manually download Youtube videos for background noise:
        - https://www.youtube.com/watch?v=OcVtCTBTJ-4 (savanna by day: "Nature and wildlife sounds from the African savanna")
        - https://www.youtube.com/watch?v=Bm_Gc4MXqfQ (savanna by night: "Lions, hyenas and other wildlife calling in the Masai Mara")
        - https://www.youtube.com/watch?v=Mr9T-943BnE (rain: "Nature Sounds: Rain Sounds One Hour for Sleeping, Sleep Aid for Everybody")
        - https://www.youtube.com/watch?v=T9IJKwEspI8 (thunder: "RELAX OR STUDY WITH NATURE SOUNDS: Ultimate Thunderstorm / 1 hour")
    2. Convert the youtube videos to .wav files using an online converter.  
    3. Place the .wav files in a folder called 'Youtube' inside the general direcory 'Data'. 
    4. Give them the following names: 'african_savanna_day.wav', 'african_savanna_night.wav', 'rain.wav', and 'thunder.wav'.
- Gunshot signals from UrbanSound8k:
    1. Download from https://urbansounddataset.weebly.com/urbansound8k.html.
    2. To use this unzip the 'UrbanSound8k' directory so it is inside the general direcoty 'Data'.
    3. Check if the directory is called 'UrbanSound8k', and it contains the directories 'audio' (containing 10 fold directories with WAV files) and 'metadata' (containing 'UrbanSound8k.csv').

