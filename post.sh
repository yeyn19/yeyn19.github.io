hexo clean
hexo gc -w=40
hexo ge
sleep 5
python generate_freq.py
sleep 3
python generate_insights.py
sleep 3
hexo d