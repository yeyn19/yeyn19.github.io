hexo clean
hexo generate
sleep 3
python generate_freq.py
python generate_insights.py
hexo s