---
title: 给hexo添加日历
tags:
  - 探索
  - 计算机
categories: 开发记录
description: 如何在hexo中添加一个置顶的更新状态，如同github开发图表一样
abbrlink: c344b452
date: 2022-06-18 13:01:50
---

今天，我想要给hexo添加一个置顶的更新记录信息，显示每天是否有更新，以及更新的字数等信息，如图：

<img src="./../files/images/本月更新.png">

其实我的本意是想要给我的论文阅读笔记模块添加本月的论文阅读打卡记录，但总之是要先实现一个基础的打卡记录功能。在网上找了一圈，没有找到相似的功能，要不就是要外挂一些别的打卡平台来实现，太麻烦。就打算自己实现一个。

大体实现思路如下：
- 扫描`_posts`文件夹中所有已有的博客，看看哪些是这个月写/更新的记录下来。
- 用一个`python`程序更新它的信息，并用html的`table`语法实现。
- 最后，用一个博客来存下当天的更新信息，并把该博客置顶。
- 每次deploy前，都执行一遍这个代码。


总体逻辑还是比较清晰的，`python`语法也没什么难度。我目前大概做了这些事：
- 用笑脸😁个数来说明当天更新了几篇博客
- 用背景颜色来说明当天更新的字数(最多5000，越多越绿)

把总体代码放出来
```python
# coding=utf-8
import os
import re
from datetime import datetime

def is_leap_year(year):
    # 判断是否为闰年
    if year % 4 == 0 and year % 100 != 0 or year % 400 == 0:
        return True
    else:
        return False


def get_num_of_days_in_month(year, month):
    # 给定年月返回月份的天数
    if month in (1, 3, 5, 7, 8, 10, 12):
        return 31
    elif month in (4, 6, 9, 11):
        return 30
    elif is_leap_year(year):
        return 29
    else:
        return 28


def get_total_num_of_day(year, month):
    # 自1800年1月1日以来过了多少天
    days = 0
    for y in range(1800, year):
        if is_leap_year(y):
            days += 366
        else:
            days += 365

    for m in range(1, month):
        days += get_num_of_days_in_month(year, m)

    return days


def get_start_day(year, month):
    # 返回当月1日是星期几，由1800.01.01是星期三推算
    return 3 + get_total_num_of_day(year, month) % 7


# 月份与名称对应的字典
month_dict = {1: '一月', 2: '二月', 3: '三月', 4: '四月', 5: '五月', 6: '六月',
              7: '七月', 8: '八月', 9: '九月', 10: '十月', 11: '十一月', 12: '十二月'}


def get_month_name(month):
    # 返回当月的名称
    return month_dict[month]


def print_month_title(year, month):
    # 打印日历的首部

    cal.write('         ' + str(get_month_name(month)) +  '   ' + str(year) + '          \n')
    cal.write('星期日 | 星期一 | 星期二  | 星期三 | 星期四 | 星期五 | 星期六 \n')
    cal.write('---| ---| ---| ---| ---| ---| ---|\n')

def count_to_color(count):
    # 将0 - 5000 字映射到 0,255,0 -> 0,139,0
    count = min(count,5000)
    count = max(count,0)
    data = 139 + float(5000 - count) / 5000 * (255 - 139)
    data = int(data)
    data = min(data,255)
    data = max(data, 0)
    return f"#00{format(data,'02X')}00"

def print_table(year, month, blog_count, word_count):
    cal.write("<table style='text-align:center'>")
    cal.write("<tr > <td colspan='7'>"+str(year)+"年"+str(get_month_name(month)) +"</td></tr>")
    cal.write('<tr><td> 星期日 </td><td> 星期一 </td><td> 星期二  </td><td> 星期三 </td><td> 星期四 </td><td> 星期五 </td><td> 星期六 </td> </tr>')
   
    cal.write("<tr>")
    i = get_start_day(year, month)
    if i != 7:
        # cal.write(' ') # 打印行首的两个空格
        cal.write('  <td></td>' * (i %7))   # 从星期几开始则空5*几个空格
    for j in range(1, get_num_of_days_in_month(year, month)+1):
        if blog_count[j - 1] > 0:
            cal.write(f" <td bgcolor={count_to_color(word_count[j-1])}>" + "😁"*blog_count[j - 1] + ' </td>')# 宽度控制，4+1=5
        else:
            cal.write( "<td>"+ str(j)+ ' </td>')# 宽度控制，4+1=5
        i += 1
        if i % 7 == 0:  # i用于计数和换行
            cal.write('</tr>\n<tr>')   # 每换行一次行首继续空格
    cal.write("</table>")
    pass


def print_month_body(year, month, blog_count, word_count):
    '''
    打印日历正文
    格式说明：空两个空格，每天的长度为5
    需要注意的是print加逗号会多一个空格
    '''
    i = get_start_day(year, month)
    if i != 7:
        # cal.write(' ') # 打印行首的两个空格
        cal.write('  |' * (i %7))   # 从星期几开始则空5*几个空格
    for j in range(1, get_num_of_days_in_month(year, month)+1):
        if blog_count[j - 1] > 0:
            cal.write(" <font color = 'Hotpink' >" + str(j) + "/"+ str(blog_count[j - 1]) +"/"+ str(word_count[j-1])+ ' </font> |')# 宽度控制，4+1=5
        else:
            cal.write(str(j)+ ' |')# 宽度控制，4+1=5
        i += 1
        if i % 7 == 0:  # i用于计数和换行
            cal.write('\n |')   # 每换行一次行首继续空格


def parse_md(f):
    #将博客内容返回成前导和实际串
    start_pos = 1
    end_pos = -1
    cc = 0
    data = f.readlines()
    for k, cont in enumerate(data):
        if "---" in cont:
            cc += 1
            if cc == 2:
                end_pos = k
                break
    return [i.strip() for i in data[1:end_pos]], [i.strip() for i in data[end_pos+1:]]

def get_word_count(f_body):
    count = 0
    for i in f_body:
        count += len(i)
    return count

def get_writing_freq(root_dir = "./source/_posts", year = 2022, month = 6):
    # 返回每天更新的博客数量
    blog_count = [0 for i in range(get_num_of_days_in_month(year,month))] #统计开始前每天更新为0
    word_count = [0 for i in range(get_num_of_days_in_month(year,month))] #统计开始前每天更新为0
    for name in os.listdir(root_dir):
        if not name.endswith("md") or name == "本月更新.md":
            continue
        name = os.path.join(root_dir,name)
        #print(name)

        f = open(name,"r",encoding="utf-8")
        f_head,f_body = parse_md(f)
        #print(f_head)
        for cont in f_head:
            ma = re.findall(f"date: {year}-{format(month,'02d')}-(\d+)",cont)
            if ma != []:
                date_pos = int(ma[0]) - 1
                word_count[date_pos] += get_word_count(f_body)
                blog_count[date_pos] += 1
                #print(get_word_count(f_body))

    return blog_count,word_count



time = datetime.now()
year = int(time.strftime("%Y")) #今天的年
month = int(time.strftime("%m")) # 今天月份


bc, wc = get_writing_freq(root_dir = "./source/_posts")


data = open("./source/_posts/本月更新.md","r")
f_head,f_data = parse_md(data)

cal = open("./source/_posts/本月更新.md",'w')
cal.write("---\n")
for i in f_head:
    cal.write(i+"\n")
cal.write("---\n")

print_table(year, month,bc,wc)
cal.close()
```

其中，更新后的deploy方法如下：
```sh
python generate_freq.py
hexo clean
hexo d -g
```