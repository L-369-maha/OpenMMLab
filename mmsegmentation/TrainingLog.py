# 首先设置matplotlib的中文显示
import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt
matplotlib.rc("font",family='SimHei') # 中文字体
 
# 日志文件路径
log_path = './project/work_dirs/Watermelon/20230619_105616/vis_data/scalars.json'
 
 
with open(log_path, "r") as f:
    json_list = f.readlines()
len(json_list)
eval(json_list[4])
 
df_train = []
df_test = []

for each in json_list[:-1]:
    if 'aAcc' in each:
        df_test.append(eval(each))
    else:
        df_train.append(eval(each))# df_train.concat(eval(each), ignore_index=True)
df_train = pd.DataFrame(df_train)
df_test = pd.DataFrame(df_test)
 
# 导出训练日志表格
df_train.to_csv('./project/work_dirs/Watermelon/训练日志-训练集.csv', index=False)
df_test.to_csv('./project/work_dirs/Watermelon/训练日志-测试集.csv', index=False)
 
 
# 可视化辅助函数
from matplotlib import colors as mcolors
import random
random.seed(124)
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'black', 'indianred', 'brown', 'firebrick', 'maroon', 'darkred', 'red', 'sienna', 'chocolate', 'yellow', 'olivedrab', 'yellowgreen', 'darkolivegreen', 'forestgreen', 'limegreen', 'darkgreen', 'green', 'lime', 'seagreen', 'mediumseagreen', 'darkslategray', 'darkslategrey', 'teal', 'darkcyan', 'dodgerblue', 'navy', 'darkblue', 'mediumblue', 'blue', 'slateblue', 'darkslateblue', 'mediumslateblue', 'mediumpurple', 'rebeccapurple', 'blueviolet', 'indigo', 'darkorchid', 'darkviolet', 'mediumorchid', 'purple', 'darkmagenta', 'fuchsia', 'magenta', 'orchid', 'mediumvioletred', 'deeppink', 'hotpink']
markers = [".",",","o","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","X","D","d","|","_",0,1,2,3,4,5,6,7,8,9,10,11]
linestyle = ['--', '-.', '-']
 
def get_line_arg():
    '''
    随机产生一种绘图线型
    '''
    line_arg = {}
    line_arg['color'] = random.choice(colors)
    # line_arg['marker'] = random.choice(markers)
    line_arg['linestyle'] = random.choice(linestyle)
    line_arg['linewidth'] = random.randint(1, 4)
    # line_arg['markersize'] = random.randint(3, 5)
    return line_arg


# 训练集损失函数
metrics = ['loss', 'decode.loss_ce', 'aux.loss_ce']
 
plt.figure(figsize=(16, 8))
 
x = df_train['step']
for y in metrics:
    plt.plot(x, df_train[y], label=y, **get_line_arg())
 
plt.tick_params(labelsize=20)
plt.xlabel('step', fontsize=20)
plt.ylabel('loss', fontsize=20)
plt.title('训练集损失函数', fontsize=25)
plt.savefig('./project/work_dirs/Watermelon/训练集损失函数.png', dpi=120, bbox_inches='tight')
 
plt.legend(fontsize=20)
 
plt.show()


df_train.columns
'''
['lr', 'data_time', 'loss', 'decode.loss_ce', 'decode.acc_seg', 'aux.loss_ce', 'aux.acc_seg', 'time', 'memory', 'step'], dtype='object'
'''
metrics = ['decode.acc_seg', 'aux.acc_seg']
 
plt.figure(figsize=(16, 8))
 
x = df_train['step']
for y in metrics:
    plt.plot(x, df_train[y], label=y, **get_line_arg())
 
plt.tick_params(labelsize=20)
plt.xlabel('step', fontsize=20)
plt.ylabel('loss', fontsize=20)
plt.title('训练集准确率', fontsize=25)
plt.savefig('./project/work_dirs/Watermelon/训练集准确率.png', dpi=120, bbox_inches='tight')
 
plt.legend(fontsize=20)
 
plt.show()

df_test.columns
# ['aAcc', 'mIoU', 'mAcc', 'data_time', 'time', 'step']
 
metrics = ['aAcc', 'mIoU', 'mAcc']
plt.figure(figsize=(16, 8))
 
x = df_test['step']
for y in metrics:
    plt.plot(x, df_test[y], label=y, **get_line_arg())
 
plt.tick_params(labelsize=20)
plt.ylim([0, 100])
plt.xlabel('step', fontsize=20)
plt.ylabel(y, fontsize=20)
plt.title('测试集评估指标', fontsize=25)
plt.savefig('./project/work_dirs/Watermelon/测试集分类评估指标.png', dpi=120, bbox_inches='tight')
 
plt.legend(fontsize=20)
 
plt.show()