# coding=utf-8
import os
import re
from datetime import datetime
from generate_insights import get_arxiv_writing_freq
from itertools import chain
from copy import deepcopy
from tqdm import tqdm


def get_prompt():
    end_year = 2023
    end_month = 10

    time = datetime.now()
    now_year = int(time.strftime("%Y")) #今天的年
    now_month = int(time.strftime("%m")) # 今天月份
    now_day = int(time.strftime("%d")) # 今天月份
    x_axis = []
    # print(f"{now_year}-{now_month}-{now_day}")
    id = 0
    paper_read,paper_cite,paper_pub,paper_scan,paper_recommend = [],[],[],[],[]
    while now_year > end_year or (now_year == end_year and now_month >= end_month): #反着排序
        blog_count,word_count,paper_interesting_count, paper_total_count, abbr_count, paper_read_count, paper_cite_count, paper_pub_count = get_arxiv_writing_freq("./source/_posts",now_year, now_month)
        assert len(paper_interesting_count) == len(paper_total_count) == len(paper_cite_count) == len(paper_pub_count) == len(paper_read_count)
        if id == 0:
            paper_read_count = paper_read_count[:now_day]
            paper_cite_count = paper_cite_count[:now_day]
            paper_pub_count = paper_pub_count[:now_day]
            paper_total_count = paper_total_count[:now_day]
            paper_interesting_count = paper_interesting_count[:now_day]
        month_days = [f"{now_year:04d}-{now_month:02d}-{(i+1):02d}" for i in range(len(paper_read_count))]
        x_axis = [month_days] + x_axis
        paper_read = [paper_read_count] + paper_read
        paper_cite = [paper_cite_count] + paper_cite
        paper_pub = [paper_pub_count] + paper_pub
        paper_scan = [paper_total_count] + paper_scan
        paper_recommend = [[int(num) if num !="" else 0 for num in paper_interesting_count]] + paper_recommend


        now_month -= 1
        if now_month == 0:
            now_month = 12
            now_year -= 1
        id += 1

    paper_read = list(chain(*paper_read))
    paper_cite = list(chain(*paper_cite))
    paper_pub = list(chain(*paper_pub))
    paper_scan = list(chain(*paper_scan))
    paper_recommend = list(chain(*paper_recommend))
    x_axis = list(chain(*x_axis))

    for lists in [paper_scan, paper_recommend]:
        """这两者需要从前到后累加"""
        now_sum = 0
        for day_id in range(len(lists)):
            if lists[day_id] > 0:
                now_sum += lists[day_id]
            lists[day_id] = now_sum

    for lists in [paper_read, paper_cite, paper_pub]:
        """这几个需要把0的位置补充成最后一个位置"""
        next_num = 0
        for day_id in range(len(lists))[::-1]:
            if lists[day_id] > 0:
                next_num = lists[day_id]
            else:
                lists[day_id] = next_num
        #最后面有连着的0要顺便补上
        pos = len(lists) -1
        while lists[pos] == 0:
            pos -= 1
        flag = lists[pos] 
        pos += 1
        while pos <= len(lists) -1:
            lists[pos] = flag
            pos += 1
    # import pdb; pdb.set_trace()

    paper_scan = [num/10 for num in paper_scan]

    timelines = {
            "100 citation": "2023-10-15",
            "Gemini": "2023-12-06",
            "publish 10 paper": "2024-01-26",
            "Sora": "2024-02-15",
            "Twitter 100 followers": "2024-02-22",
            "scan 10000 paper": "2024-04-18",
        }

    chart_head = ["read", "cite", "publish", "paper-scan-in-arxiv/10", "paper-recommend-in-arxiv"]
    title =  ""
    interval = int(len(paper_scan)/10)

    vertical_line = """
    {
        name: '{{line_name}}',
        type: 'line',
        lineStyle: {
            color: 'black', // 竖线颜色设置为黑色
        },
        markLine: {
            symbol: ['none', 'none'], // 不显示标记线两端的标记
            label: {
                show: true, // 显示标签
                position: 'end', // 在线的末端显示标签
                formatter: '{{obj_name}}', // 自定义显示的文本
                rotate: 30, // 旋转角度
            },
            lineStyle: {
                color: 'black',
            },
            data: [
                {
                    name: '{{obj_name}}', // 这里是竖线的名字
                    xAxis: '{{obj_time}}' // 假设你想在 2023-12-15 这个日期上添加竖线
                }
            ]
        }
    },
    """

    vertical_line_prompt = ""
    for k, (obj_name, obj_time) in enumerate(timelines.items()):
        vertical_line_new = deepcopy(vertical_line)
        vertical_line_new = vertical_line_new.replace("{{line_name}}", f"obj_{k}")
        vertical_line_new = vertical_line_new.replace("{{obj_name}}", obj_name)
        vertical_line_new = vertical_line_new.replace("{{obj_time}}", obj_time)
        vertical_line_prompt += vertical_line_new
    
    echart_prompt = """
option = {
title: {
    text: \"""" + title +"""\"
},
tooltip: {
    trigger: 'axis',
    axisPointer: {
    type: 'cross',
    label: {
        backgroundColor: '#6a7985'
    }
    }
},
legend: {
    data: """ + str(chart_head) +"""
},
toolbox: {
    feature: {
    saveAsImage: {}
    }
},
grid: {
    left: '3%',
    right: '4%',
    bottom: '3%',
    containLabel: true
},
xAxis: [
    {
    type: 'category',
    boundaryGap: false,
    data: """ + str(x_axis) +""",
    axisLabel: {
        interval: """ +str(interval) + """,
        rotate: 45 // 设置标签旋转角度为45度
    }
    }
],
yAxis: [
    {
    type: 'value'
    }
],
series: [
    """+ vertical_line_prompt+"""
    {
    name: 'read',
    type: 'line',
    emphasis: {
        focus: 'series'
    },
    itemStyle: {
        borderWidth: 0
    },
    data: """ + str(paper_read) +"""
    },
    {
    name: 'cite',
    type: 'line',
    emphasis: {
        focus: 'series'
    },
    itemStyle: {
        borderWidth: 0
    },
    data: """ + str(paper_cite) +"""
    },
    {
    name: 'publish',
    type: 'line',
    emphasis: {
        focus: 'series'
    },
    itemStyle: {
        borderWidth: 0
    },
    data: """ + str(paper_pub) +"""
    },
    {
    name: 'paper-scan-in-arxiv/10',
    type: 'line',
    emphasis: {
        focus: 'series'
    },
    itemStyle: {
        borderWidth: 0
    },
    data: """ + str(paper_scan) +"""
    },
    {
    name: 'paper-recommend-in-arxiv',
    type: 'line',
    emphasis: {
        focus: 'series'
    },
    itemStyle: {
        borderWidth: 0
    },
    data: """ + str(paper_recommend) +"""
    }
]
};
"""
    return echart_prompt

def is_leap_year(year):
    # 判断是否为闰年
    if (year % 4 == 0) and (year % 100 != 0) or (year % 400 == 0):
        return True
    else:
        return False


def get_num_of_days_in_month(year, month):
    # 给定年月返回月份的天数
    if month in [1, 3, 5, 7, 8, 10, 12]:
        return 31
    elif month in [4, 6, 9, 11]:
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
month_dict = {1: '一月January', 2: '二月February', 3: '三月March', 4: '四月April', 5: '五月May', 6: '六月June',
              7: '七月July', 8: '八月August', 9: '九月September', 10: '十月October', 11: '十一月November', 12: '十二月December'}


def get_month_name(month):
    # 返回当月的名称
    return month_dict[month]


def print_month_title(year, month):
    # 打印日历的首部

    cal.write('         ' + str(get_month_name(month)) +  '   ' + str(year) + '          \n')
    cal.write('星期日Sunday | 星期一Monday | 星期二Tuesday | 星期三Wednesday | 星期四Thursday | 星期五Friday | 星期六Saturday \n')
    cal.write('----------- | ----------- | ------------ | -------------- | ------------- | ----------- | ------- |\n')

def count_to_color(count):
    # 将0 - 5000 字映射到 0,255,0 -> 0,139,0
    count = min(count,5000)
    count = max(count,0)
    data = 139 + float(5000 - count) / 5000 * (255 - 139)
    data = int(data)
    data = min(data,255)
    data = max(data, 0)
    return f"#00{format(data,'02X')}00"

def print_table(year, month, blog_count, word_count,status,id,hidden):
    if hidden:
        cal.write(f"<table id='{id}' hidden='hidden' style='text-align:center'>")
    else:
        cal.write(f"<table id='{id}' style='text-align:center'>")
    cal.write("<tr>")
    cal.write(''' <td>
        <form action="">
            <input type="button" value="上月Last Month" onclick="tips(-1)">
        </form>
    </td>
    ''')
    cal.write("<td colspan='5'>"+str(year)+"年"+str(get_month_name(month)) +"</td>")
    cal.write(''' <td>
        <form action="">
            <input type="button" value="下月Next Month" onclick="tips(1)">
        </form>
    </td>
    ''')
    cal.write("</tr>")
    cal.write('<tr><td> 星期日</br>Sunday </td><td> 星期一</br>Monday </td><td> 星期二</br>Tuesday  </td><td> 星期三</br>Wednesday </td><td> 星期四</br>Thursday </td><td> 星期五</br>Friday </td><td> 星期六</br>Saturday </td> </tr>')
   
    cal.write("<tr>")
    i = get_start_day(year, month)
    if i != 7:
        # cal.write(' ') # 打印行首的两个空格
        cal.write('  <td></td>' * (i %7))   # 从星期几开始则空5*几个空格
    for j in range(1, get_num_of_days_in_month(year, month)+1):
        if blog_count[j - 1] > 0:
            if status[j-1] != "":
                status[j - 1] += ": "
            cal.write(f" <td bgcolor={count_to_color(word_count[j-1])}>" + "<b>"+status[j - 1]  + "</b>"+"😁"*blog_count[j - 1] + ' </td>')# 宽度控制，4+1=5
        else:
            cal.write( "<td>"+ str(j)+ ' </td>')# 宽度控制，4+1=5
        i += 1
        if i % 7 == 0:  # i用于计数和换行
            cal.write('</tr>\n<tr>')   # 每换行一次行首继续空格
    cal.write("</tr></table>")
    cal.write("\n\n")
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
    code = 0
    for i in f_body:
        if code == 0:
            count += len(i)
        if i.startswith("```"):
            code = 1 - code
    return count

def get_all_blogs(root_dir = "./source/_posts"):
    names = []
    for name in os.listdir(root_dir):
        path = os.path.join(root_dir, name)
        if os.path.isdir(path):
            sub_names = get_all_blogs(path)
            names.extend(sub_names)
        else:
            if not name.endswith("md") or name == "本月更新.md":
                continue
            names.append(os.path.join(root_dir,name))
    return names

def get_writing_freq(root_dir = "./source/_posts", year = 2022, month = 6):
    # 返回每天更新的博客数量
    blog_count = [0 for i in range(get_num_of_days_in_month(year,month))] #统计开始前每天更新为0
    word_count = [0 for i in range(get_num_of_days_in_month(year,month))] #统计开始前每天更新为0
    status_count = ["" for i in range(get_num_of_days_in_month(year,month))] #统计开始前每天更新为0
    
    names = get_all_blogs(root_dir)
    for name in names:

        f = open(name,"r",encoding="utf-8")
        f_head,f_body = parse_md(f)
        date_pos = -1
        sta = ""
        for cont in f_head:
            ma = re.findall(f"date: {year}-{format(month,'02d')}-(\d+)",cont)
            if ma != []:
                print(name)
                date_pos = int(ma[0]) - 1
                # print(date_pos)
                word_count[date_pos] += get_word_count(f_body)
                blog_count[date_pos] += 1
                #print(get_word_count(f_body))
                for cont in f_head:
                    # 找到状态：
                    if "status" in cont:
                        # print(cont)
                        sta = cont.split(":")[-1].strip()
        if sta != "":
            status_count[date_pos] = sta
    # print(status_count)
    return blog_count,word_count,status_count



time = datetime.now()
now_year = int(time.strftime("%Y")) #今天的年
now_month = int(time.strftime("%m")) # 今天月份
print(f"now {now_year}-{now_month}")
end_year = 2022
end_month = 6



data = open("./source/_posts/本月更新.md","r").readlines()[:10]

cal = open("./source/_posts/本月更新.md",'w')

for i in data:
    cal.write(i)

echart_prompt = get_prompt()
cal.write("""
<script src="https://github.com/TransformersWsz/TransformersWsz.github.io/releases/download/echarts/echarts.min.js"></script>

{% echarts 500 '100%' %} 
"""+  echart_prompt +"""
{% endecharts %}
\n""")


id = 0
while now_year > end_year or (now_year == end_year and now_month >= end_month): #反着排序
    id += 1
    #下一个月
    now_month -= 1
    if now_month == 0:
        now_month = 12
        now_year -= 1

cal.write('''
<script type="text/javascript">
    var now_id = '''+str(0)+''';
    var max_id = '''+str(id-1)+''';
    function tips(num){
        document.getElementById(now_id).hidden = "hidden";
        now_id -= num;
        if (now_id > max_id) {now_id = max_id;}
        if (now_id < 0) {now_id = 0;}
        document.getElementById(now_id).hidden = "";
    }
</script>
''')


now_year = int(time.strftime("%Y")) #今天的年
now_month = int(time.strftime("%m")) # 今天月份
id = 0
while now_year > end_year or (now_year == end_year and now_month >= end_month): #反着排序
    bc, wc,sc = get_writing_freq("./source/_posts",now_year, now_month)
    hidden = not (now_year == int(time.strftime("%Y")) and now_month == int(time.strftime("%m")))
    print_table(now_year, now_month,bc,wc,sc,id,hidden)
    id += 1
    #下一个月
    now_month -= 1
    if now_month == 0:
        now_month = 12
        now_year -= 1
    # exit()
cal.close()

