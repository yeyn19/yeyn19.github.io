hexo clean
hexo gc -w=40
hexo ge
sleep 3
python generate_freq.py
python generate_insights.py
hexo d