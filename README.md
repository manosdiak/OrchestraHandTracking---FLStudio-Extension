# OrchestraHandTracking---FLStudio-Extension
Hand tracking extension for FLStudio. It allows the user to play notes from different keys, different organs, in different volumes.
**Guide**
It functions through your laptop's/pc's camera. With your right hand, you can change the volume (by moving it vertically) and instruments (by moving it horizontally). The preset has 6 instruments, but you can change that in the Configuration section to your liking. With your left hand, you can play different notes by weaving it horizontally and you can change keys by moving it vertically.
**Installation**
1. Download the code.
2. You'll need to download mediapipe. Paste this to your command prompt "pip install opencv-python mediapipe mido python-rtmidi".
3. Go to the following URL: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task. Take the TASK file that will be downloaded and drop it to the code's file.
4. If in windows, download loopMIDI (https://www.tobias-erichsen.de/software/loopmidi.html). Add 6 MIDIs (or the number of the instruments you've chosen to use). Name them like so: "HandMIDI-0", "HandMIDI-1". If in mac or linux, you'll find more info inside the code's file.
5. Open FL Studio. Press F10 and enable all the midis you've just created. Assign each one's relative port. Close settings
6. Open the channel rack. Add the instruments. Right click each one, choose "Receive notes from"-> "HandMIDI-X"->"All", where x the number of the instrument(starting from 0).
7. Run the code
8. On FL Studio, press F9. Right click on the Master Mixer's volume fader and choose link to controller.
9. You're ready to go! 
