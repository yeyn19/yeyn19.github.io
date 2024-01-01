   {% chart 90% 300 %}
   {
        type: 'line',
        data: {
            labels: ['2020', '2021', '2022', '2023'],
            datasets: [{ label: '论文发布数量', data: [10, 20, 30, 40], borderColor: 'red' }, { label: '论文引用数量', data: [15, 25, 35, 45], borderColor: 'blue' }, { label: '已读论文数量', data: [5, 15, 25, 35], borderColor: 'green' }]
        },
        options: {
            scales: {
                'y-axis-1': {
                    type: 'linear',
                    display: true,
                    position: 'left',
                },
                'y-axis-2': {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    grid: {
                        drawOnChartArea: false,
                    },
                },
                'y-axis-3': {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    grid: {
                        drawOnChartArea: false,
                    },
                }
            }
        }
    }
{% endchart %}
