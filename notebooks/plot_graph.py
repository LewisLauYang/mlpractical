import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

picPath = '/Users/liuwenyang/Edinburgh/MLP/mlp2018/report/pic/part2'

class plot_model:
    def __init__(self, fileNameArray, labels,title):
        self.fileNameArray = fileNameArray
        self.labels = labels
        self.title = title

def plot_graph(plotModels,saveFileName,acc=True):

    column = 2

    rows = int((len(plotModels) - 1) / 2) + 1

    fig1 = plt.figure(figsize=(6.4 * column, 4.8 * rows))



    i = 0
    for plot_model in plotModels:
        plt.subplot(rows,2,i + 1)
        files = plot_model.fileNameArray
        labels = plot_model.labels
        title = plot_model.title
        colors = ['blue', 'red', 'orange', 'black']
        j = 0
        for file in files:
            isAcc = 'val_loss'
            if acc:
                isAcc = 'val_acc'
            dataPath = os.path.join(os.getcwd(),file,'result_outputs','summary.csv')
            data = pd.read_csv(dataPath, delimiter=',')
            valAcc = data[isAcc]

            plt.plot(np.arange(100), valAcc,color=colors[j],label=labels[j])
            j += 1
        i += 1
        plt.title(title)

        # ticks = np.arange(start=0, stop=1.0, step=0.1)

        # plt.yticks(ticks)
        plt.xlabel('Epoch')
        yalbel = 'Valid Loss'
        if acc:
            yalbel = 'Valid Accuracy'
        plt.ylabel(yalbel)
        plt.legend()


    plt.show()
    saveFileName = saveFileName + '.pdf'
    fig1.tight_layout()  # This minimises whitespace around the axes.
    path = os.path.join(picPath,saveFileName)
    fig1.savefig(path)
    return plt

#ave and dilated

plot_model1 = plot_model(["stride2_filter32_layer1_type1","stride2_filter32_layer1_type3"],
           ['dilated','average pooling'],
           'Accuracy of layer 1')

plot_model2 = plot_model(["stride2_filter32_layer2_type1","stride2_filter32_layer2_type3"],
           ['dilated','average pooling'],
           'Accuracy of layer 2')

plot_model3 = plot_model(["stride2_filter32_layer3_type1","stride2_filter32_layer3_type3"],
           ['dilated','average pooling'],
           'Accuracy of layer 3')

plot_model4 = plot_model(["stride2_filter32_layer4_type1","stride2_filter32_layer4_type3"],
           ['dilated','average pooling'],
           'Accuracy of layer 4')



plot_graph([plot_model1,plot_model2,plot_model3,plot_model4],'dilated_average_layer_acc',acc=True)

plot_model1 = plot_model(["stride2_filter32_layer1_type1","stride2_filter32_layer1_type3"],
           ['dilated','average pooling'],
           'Loss of layer 1')

plot_model2 = plot_model(["stride2_filter32_layer2_type1","stride2_filter32_layer2_type3"],
           ['dilated','average pooling'],
           'Loss of layer 2')

plot_model3 = plot_model(["stride2_filter32_layer3_type1","stride2_filter32_layer3_type3"],
           ['dilated','average pooling'],
           'Loss of layer 3')

plot_model4 = plot_model(["stride2_filter32_layer4_type1","stride2_filter32_layer4_type3"],
           ['dilated','average pooling'],
           'Loss of layer 4')



plot_graph([plot_model1,plot_model2,plot_model3,plot_model4],'dilated_average_layer_loss',acc=False)


# plot_model1 = plot_model(["stride2_filter64_layer1_type2","stride2_filter64_layer1_type3"],
#            ['max pooling','average pooling'],
#            'Accuracy of layer 1')
#
# plot_model2 = plot_model(["stride2_filter64_layer2_type2","stride2_filter64_layer2_type3"],
#            ['max pooling','average pooling'],
#            'Accuracy of layer 2')
#
# plot_model3 = plot_model(["stride2_filter64_layer3_type2","stride2_filter64_layer3_type3"],
#            ['max pooling','average pooling'],
#            'Accuracy of layer 3')
#
# plot_model4 = plot_model(["stride2_filter64_layer4_type2","stride2_filter64_layer4_type3"],
#            ['max pooling','average pooling'],
#            'Accuracy of layer 4')
#
#
#
# plot_graph([plot_model1,plot_model2,plot_model3,plot_model4],'max_average_layer_acc',acc=True)
#
#
# plot_model1 = plot_model(["stride2_filter64_layer1_type2","stride2_filter64_layer1_type3"],
#            ['max pooling','average pooling'],
#            'Loss of layer 1')
#
# plot_model2 = plot_model(["stride2_filter64_layer2_type2","stride2_filter64_layer2_type3"],
#            ['max pooling','average pooling'],
#            'Loss of layer 2')
#
# plot_model3 = plot_model(["stride2_filter64_layer3_type2","stride2_filter64_layer3_type3"],
#            ['max pooling','average pooling'],
#            'Loss of layer 3')
#
# plot_model4 = plot_model(["stride2_filter64_layer4_type2","stride2_filter64_layer4_type3"],
#            ['max pooling','average pooling'],
#            'Loss of layer 4')
#
# plot_graph([plot_model1,plot_model2,plot_model3,plot_model4],'max_average_layer_loss',acc=False)
#
#
#
#
# plot_model1 = plot_model(["stride2_filter32_layer1_type2","stride2_filter32_layer1_type3"],
#            ['max pooling','average pooling'],
#            'Accuracy of layer 1')
#
# plot_model2 = plot_model(["stride2_filter32_layer2_type2","stride2_filter32_layer2_type3"],
#            ['max pooling','average pooling'],
#            'Accuracy of layer 2')
#
# plot_model3 = plot_model(["stride2_filter32_layer3_type2","stride2_filter32_layer3_type3"],
#            ['max pooling','average pooling'],
#            'Accuracy of layer 3')
#
# plot_model4 = plot_model(["stride2_filter32_layer4_type2","stride2_filter32_layer4_type3"],
#            ['max pooling','average pooling'],
#            'Accuracy of layer 4')
#
#
#
# plot_graph([plot_model1,plot_model2,plot_model3,plot_model4],'filter32_max_average_layer_acc',acc=True)
#
#
# plot_model1 = plot_model(["stride2_filter32_layer1_type2","stride2_filter32_layer1_type3"],
#            ['max pooling','average pooling'],
#            'Loss of layer 1')
#
# plot_model2 = plot_model(["stride2_filter32_layer2_type2","stride2_filter32_layer2_type3"],
#            ['max pooling','average pooling'],
#            'Loss of layer 2')
#
# plot_model3 = plot_model(["stride2_filter32_layer3_type2","stride2_filter32_layer3_type3"],
#            ['max pooling','average pooling'],
#            'Loss of layer 3')
#
# plot_model4 = plot_model(["stride2_filter32_layer4_type2","stride2_filter32_layer4_type3"],
#            ['max pooling','average pooling'],
#            'Loss of layer 4')
#
# plot_graph([plot_model1,plot_model2,plot_model3,plot_model4],'filter32_max_average_layer_loss',acc=False)





