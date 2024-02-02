ffmpeg -i /dev/video0 -t 1 -f image2 -vframes 1 -y lol.jpg
viu lol.jpg

cd ~/ai-stuff/candle
caption=$(cargo run --release --example=blip -- --image ~/lol.jpg | tail -n 1)
echo $caption
cd

prompt="GPT4 User: Rephrase the following description as if it were a narration out of one of david attenborough \
nature documentary. Describe people as if they were animals \
in a scientific objective fashion. Be brief. Don't provide a summary. Rephrase the following description: ${caption}<|end_of_turn|>GPT4 Assistant:"
echo $prompt

cd ~/ai-stuff/llama.cpp
david_narrating=$(./main -m models/starling-lm-7b-alpha.Q5_K_M.gguf --color -c 8192 --temp 0.7 --repeat_penalty 1.1 -n -1 -p "$prompt" --no-display-prompt)
david_narrating=${david_narrating/<|end_of_turn|>/}
echo $david_narrating
cd

cd ~/ai-stuff/fresh-and-clean-tts-bin
venv/bin/tts --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
     --text "${david_narrating}" \
     --speaker_wav ~/Documents/emma.wav \
     --language_idx en --pipe_out | aplay
cd
