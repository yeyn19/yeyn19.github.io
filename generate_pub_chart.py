# def generate_chart_md(data, labels, filename="chart.md"):
#     """
#     Generates a Markdown file with a Chart.js chart.

#     :param data: A dictionary with dataset labels and values.
#     :param labels: A list of labels for the X-axis.
#     :param filename: The name of the Markdown file to be created.
#     """
#     with open(filename, "w") as md_file:
#         # 添加 Chart.js CDN
#         md_file.write("```html\n<script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>\n```\n\n")

#         # 添加 canvas 元素
#         md_file.write("```html\n<canvas id=\"myChart\" width=\"400\" height=\"400\"></canvas>\n```\n\n")

#         # 构建并添加 JavaScript
#         js_code = f"""
#         <script>
#         var ctx = document.getElementById('myChart').getContext('2d');
#         var myChart = new Chart(ctx, {{
#             type: 'line',
#             data: {{
#                 labels: {labels},
#                 datasets: [{', '.join([f"{{ label: '{label}', data: {data[label]}, borderColor: '{color}' }}" for label, color in zip(data.keys(), ["red", "blue", "green", "purple"])])}]
#             }},
#             options: {{
#                 scales: {{
#                     'y-axis-1': {{
#                         type: 'linear',
#                         display: true,
#                         position: 'left',
#                     }},
#                     'y-axis-2': {{
#                         type: 'linear',
#                         display: true,
#                         position: 'right',
#                         grid: {{
#                             drawOnChartArea: false,
#                         }},
#                     }},
#                     'y-axis-3': {{
#                         type: 'linear',
#                         display: true,
#                         position: 'right',
#                         grid: {{
#                             drawOnChartArea: false,
#                         }},
#                     }}
#                 }}
#             }}
#         }});
#         </script>
#         """
#         md_file.write(f"```html\n{js_code}\n```\n")

# # 示例数据
# data_example = {
#     "论文发布数量": [10, 20, 30, 40],
#     "论文引用数量": [15, 25, 35, 45],
#     "已读论文数量": [5, 15, 25, 35]
# }
# labels_example = ['2020', '2021', '2022', '2023']

# # 生成 Markdown 文件
# generate_chart_md(data_example, labels_example, "./test.md")


import matplotlib.pyplot as plt

# 示例数据
years = ['2020', '2021', '2022', '2023']
publication_counts = [10, 20, 30, 40]
citation_counts = [15, 25, 35, 45]
read_paper_counts = [5, 15, 25, 35]

# 创建图表
plt.figure(figsize=(10, 6))

plt.plot(years, publication_counts, label='论文发布数量', marker='o')
plt.plot(years, citation_counts, label='论文引用数量', marker='s')
plt.plot(years, read_paper_counts, label='已读论文数量', marker='^')

# 添加图例
plt.legend()

# 添加标题和轴标签
plt.title('论文相关数据随时间的变化')
plt.xlabel('年份')
plt.ylabel('数量')

# 显示图表
plt.show()
