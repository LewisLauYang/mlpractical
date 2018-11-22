import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

class plot_model:
    def __init__(self, fileNameArray, labels,title):
        self.fileNameArray = fileNameArray
        self.labels = labels
        self.title = title

def plot_graph(plotModels):



    plt.figure(figsize=(6.4 * len(plotModels), 4.8))

    i = 0
    for plot_model in plotModels:
        plt.subplot(1,len(plotModels),i + 1)
        files = plot_model.fileNameArray
        labels = plot_model.labels
        title = plot_model.title
        colors = ['blue', 'red', 'orange', 'black']
        j = 0
        for file in files:
            dataPath = os.path.join(os.getcwd(),file,'result_outputs','summary.csv')
            data = pd.read_csv(dataPath, delimiter=',')
            valAcc = data['val_acc']

            plt.plot(np.arange(100), valAcc,color=colors[j],label=labels[j])
            j += 1
        i += 1
        plt.title(title)

        ticks = np.arange(start=0, stop=1.0, step=0.1)

        plt.yticks(ticks)
        plt.xlabel('Epoch')
        plt.ylabel('Valid Accuracy')
        plt.legend()


    plt.show()
    return plt



plot_model1 = plot_model(["strided_convolution_avgpooling_1","strided_convolution_avgpooling_2","strided_convolution_avgpooling_3","strided_convolution_avgpooling_4"],
           ['stride=1','stride=2','stride=3','stride=4'],
           'Valid Accuracy with different stride \n combined with avgpooling')

plot_model2 = plot_model(["strided_convolution_maxpooling_1","strided_convolution_maxpooling_2","strided_convolution_maxpooling_3","strided_convolution_maxpooling_4"],
           ['stride=1','stride=2','stride=3','stride=4'],
           'Valid Accuracy with different stride \n combined with maxpooling')

plot_graph([plot_model1,plot_model2])


