hexo clean
hexo generate
sleep 5
python generate_freq.py
python generate_insights.py
sleep 3
hexo s