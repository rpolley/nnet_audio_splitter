wget -N -l inf -r -np -p http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/
cp -r "http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio" ./Audio
rm -r "http://www.repository.voxforge1.org/"
rm -f Audio/Main/*.html*
