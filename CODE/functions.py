import pandas as pd
import os

import cv2  # 画像認識するのはcv2
import keras

import numpy as np
import tensorflow as tf

from keras import backend as K
from keras import layers
from keras.backend import set_session

from keras.layers import *
from keras.models import Model

from keras.utils.np_utils import to_categorical
from sklearn.model_selection import *

import plotly
import plotly.graph_objs as go
import plotly.offline as offline
import matplotlib.pyplot as plt

import seaborn as sns

import warnings



def load_data():
    path_dir = "../New-RGB/"

    filelist = pd.read_csv("./new_Mandible_check.csv", encoding="SHIFT-JIS")
    #filelist.columns = ["num","Filename", "Species", "Japanese"]
    japanese_group = filelist.groupby(["Japanese"])
    FILELIST = filelist.sort_values("Filename")
    japanese_file = list(FILELIST.Japanese)
    
    num_list = []

    for i,_ in filelist.iterrows():
            num_list.append(i)

    X = []
    y = []
    y_jap = []
    y_species = []
    y_food = []
    filelise_index = []
    for i in num_list:
        PATH = os.path.join(*[path_dir, filelist.Filename[i]])
        if os.path.isfile(PATH) == True:
            img = cv2.imread(PATH)
            #img = cv2.resize(img, (128,128))
            img = img /255
            X.append(img)
            y.append(filelist.Family_num[i])
            y_jap.append(filelist.Family[i])
            y_species.append(filelist.Japanese[i])
            y_food.append(filelist.Food[i])
        else:
            print(PATH)
        
    X = np.array(X)
    y = np.array(y)
    
    return X,y


    """
    Composition of Data.
    
    data ━━━┳━━> train 67 % ┳━━ train  75 %
            ┃               ┗━━   val  25 %
            ┃
            ┗-- test  33 %
    
    test data ... data for check accuracy, loss et al
    
    
    """




def make_train_test(X,y,seed = 223):

    X_test =[]
    y_test =[]
    X_train = []
    y_train = []
    group_train = []
    group_test = []
        
    filelist = pd.read_csv("./new_Mandible_check.csv", encoding="SHIFT-JIS")
    #filelist.columns = ["num","Filename", "Species", "Japanese"]
    for i in range(7):
        Group = filelist[y == i].Tag
        yy = y[y == i]
        #yyy = y_tag[y ==i]
        XX = X[y==i]
        gss = GroupShuffleSplit(n_splits=2, train_size=.67, random_state=seed)
        for train_idx, test_idx in gss.split(XX, yy, Group):
            continue
        #print(yyy[test_idx])
        X_test.append(XX[test_idx])
        X_train.append(XX[train_idx])
        y_test.append(yy[test_idx])
        y_train.append(yy[train_idx])
        group_train.append(np.array(Group)[train_idx])
        group_test.append(np.array(Group)[test_idx])

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    group_train = np.array(group_train)
    group_test = np.array(group_test)

    for j in range(X_train.shape[0]):
        if j == 0 :
            tmp = X_train[j]
            tmp2 = y_train[j]
            tmp3 = group_train[j]
        else:
            tmp = np.concatenate([tmp, X_train[j]])
            tmp2 = np.concatenate([tmp2,y_train[j]])
            tmp3 = np.concatenate([tmp3,group_train[j]])
    X_train = tmp
    y_train = tmp2
    group_train = tmp3

    for j in range(X_test.shape[0]):
        if j == 0 :
            tmp = X_test[j]
            tmp2 = y_test[j]
            tmp3 = group_test[j]
        else:
            tmp = np.concatenate([tmp, X_test[j]])
            tmp2 = np.concatenate([tmp2,y_test[j]])
            tmp3 = np.concatenate([tmp3,group_test[j]])
    X_test = tmp
    y_test = tmp2
    group_test = tmp3
    print("Train size :{}, Test size:{}".format(np.array(X_train).shape,np.array(X_test).shape))
    Y_train = to_categorical(y_train)
    Y_test = to_categorical(y_test)

    return X_train, y_train,Y_train, X_test, y_test ,Y_test, group_train, group_test


def make_val_train(X_train,y_train,group_train,seed = None):
    if seed == None:
        seed = np.random.randint(1, 1000)
    else:
        seed = int(seed)
    X_val =[]
    y_val =[]
    X_train_2 = []
    y_train_2 = []
    group_train_2 = []
    group_val = []
    for i in range(7):
        Group = group_train[y_train == i]
        yy = y_train[y_train == i]
        #yyy = y_tag[y ==i]
        XX = X_train[y_train==i]
        gss = GroupShuffleSplit(n_splits=2, train_size=.75, random_state=seed)
        for train_idx, test_idx in gss.split(XX, yy, Group):
            continue
        X_val.append(XX[test_idx])
        X_train_2.append(XX[train_idx])
        y_val.append(yy[test_idx])
        y_train_2.append(yy[train_idx])
        group_train_2.append(np.array(Group)[train_idx])
        group_val.append(np.array(Group)[test_idx])
        
    X_train_2 = np.array(X_train_2)
    X_val = np.array(X_val)
    group_train_2 = np.array(group_train_2)
    group_val = np.array(group_val)
    
    for j in range(X_train_2.shape[0]):
            if j == 0 :
                tmp = X_train_2[j]
                tmp2 = y_train_2[j]
                tmp3 = group_train_2[j]
            else:
                tmp = np.concatenate([tmp, X_train_2[j]])
                tmp2 = np.concatenate([tmp2,y_train_2[j]])
                tmp3 = np.concatenate([tmp3,group_train_2[j]])
    X_train_2 = tmp
    y_train_2 = tmp2
    group_train_2 = tmp3
    
    for j in range(X_val.shape[0]):
        if j == 0 :
            tmp =X_val[j]
            tmp2 = y_val[j]
            tmp3 = group_val[j]
        else:
            tmp = np.concatenate([tmp, X_val[j]])
            tmp2 = np.concatenate([tmp2,y_val[j]])
            tmp3 = np.concatenate([tmp3,group_val[j]])
    X_val = tmp
    y_val = tmp2
    group_val = tmp3

    Y_train_2 = to_categorical(y_train_2)
    Y_val = to_categorical(y_val)
    print("Train2 size :{}, val size:{}".format(np.array(X_train_2).shape,np.array(X_val).shape))
    
    return X_train_2, y_train_2,Y_train_2, X_val, y_val ,Y_val, seed, group_train_2, group_val



def create_model(num_layer, activation, num_filters, latent_dim = 3):
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean =0., stddev =1.)
        return z_mean + K.exp(z_log_var) * epsilon
  
   
    inputs = Input((128,128,3))

    x0 = Lambda(lambda x: x[ :,:,:,0:1], output_shape=(128,128,1),name = 'R_extract')(inputs)
    x1 = Lambda(lambda x: x[ :,:,:,1:2], output_shape=(128,128,1),name = 'G_extract')(inputs)
    x2 = Lambda(lambda x: x[:,:,:,2:], output_shape=(128,128,1),name = 'B_extract')(inputs)

    x0 = Convolution2D(filters=num_filters[0], kernel_size=(3,3), padding="same", activation=activation,
                      name = 'R_input')(x0)
    #print(K.int_shape(x0))
    for i in range(1,num_layer):
        x0 = MaxPooling2D((2,2), padding='same', name = 'R_MaxPool_{}'.format(i))(x0)
        x0 = Convolution2D(filters=num_filters[i], kernel_size=(3,3), padding="same", activation=activation,
                          name = 'R_conv_{}'.format(i))(x0)
    x0 = layers.Flatten(name = 'R_flatten')(x0)

    x1 = Convolution2D(filters=num_filters[0], kernel_size=(3,3), padding="same", activation=activation,
                      name = 'G_input')(x1)
    for i in range(1,num_layer):
        x1 = MaxPooling2D((2,2), padding='same',name = 'G_MaxPool_{}'.format(i))(x1)
        x1 = Convolution2D(filters=num_filters[i], kernel_size=(3,3), padding="same", activation=activation
                          ,name = 'G_conv_{}'.format(i))(x1)
    x1 = layers.Flatten(name = 'G_flatten')(x1)

    x2 = Convolution2D(filters=num_filters[0], kernel_size=(3,3), padding="same", activation=activation,
                      name = 'B_input')(x2)
    for i in range(1,num_layer):
        x2 = MaxPooling2D((2,2), padding='same',name = 'B_MaxPool_{}'.format(i))(x2)
        x2 = Convolution2D(filters=num_filters[i], kernel_size=(3,3), padding="same", activation=activation
                          ,name = 'B_conv_{}'.format(i))(x2)
    x2 = layers.Flatten(name = 'B_flatten')(x2)

    x = keras.layers.Concatenate(name = 'Concatenate')([x0, x1, x2])
    shape_before_flattening = K.int_shape(x)


    ## Intermediate Layer Part
    z_mean = layers.Dense(latent_dim, name ='z_mean')(x) #latent Spaceに圧縮
    z_log_var = layers.Dense(latent_dim, name ='z_log_var')(x) #z_sigma に対応している
    z = layers.Lambda(sampling, name = 'sampling')([z_mean, z_log_var])

    encoder = Model(inputs,z)

    decoder_input = layers.Input(K.int_shape(z)[1:])
    x = layers.Dense(np.prod(shape_before_flattening[1:]),activation=activation)(decoder_input)

    Shape = int(128 /(2 ** (num_layer - 1)))
    flatten_shape = int(Shape * Shape * num_filters[-1])

    ## Convolution
    x0 = Lambda(lambda x: x[:, :flatten_shape], output_shape=(Shape,Shape,num_filters[-1]))(x)
    x1 = Lambda(lambda x: x[:,  flatten_shape:2 * flatten_shape], output_shape=(Shape,Shape,num_filters[-1]))(x)
    x2 = Lambda(lambda x: x[:, 2*flatten_shape:], output_shape=(Shape,Shape,num_filters[-1]))(x)

    x0 =layers.Reshape((Shape, Shape, num_filters[-1]))(x0)
    x1 =layers.Reshape((Shape, Shape, num_filters[-1]))(x1)
    x2 =layers.Reshape((Shape, Shape, num_filters[-1]))(x2)

    for i in reversed(range(num_layer - 1)):
            x0 = Convolution2D(filters=num_filters[i], kernel_size=(3,3),padding="same", activation=activation)(x0)
            x0 = UpSampling2D((2,2))(x0)
    x0 = Convolution2D(1,(3,3), padding='same', activation='sigmoid')(x0)

    for i in reversed(range(num_layer - 1)):
            x1 = Convolution2D(filters=num_filters[i], kernel_size=(3,3),padding="same", activation=activation)(x1)
            x1 = UpSampling2D((2,2))(x1)
    x1 = Convolution2D(1,(3,3), padding='same', activation='sigmoid')(x1)

    for i in reversed(range(num_layer - 1)):
            x2 = Convolution2D(filters=num_filters[i], kernel_size=(3,3),padding="same", activation=activation)(x2)
            x2 = UpSampling2D((2,2))(x2)
    x2 = Convolution2D(1,(3,3), padding='same', activation='sigmoid')(x2)

    decoder_output = layers.Concatenate()([x0, x1, x2])
    decoder = Model(decoder_input, decoder_output)
    z_decoded = decoder(z)

    classification_output = layers.Dense(7 ,activation='softmax',name = 'Family_Classifier')(z)

    model = Model(inputs, [classification_output, z_decoded])

    Classifier = Model(inputs,classification_output, name ='Classifier')
    
    return model, z_mean, z_log_var, encoder, decoder, Classifier



## Function of Score-CAM
def score_cam(num , layer_name ,X_test, model):
    img_array = X_test[num].reshape(1,X_test[num].shape[0],X_test[num].shape[1],3)
    cls = np.argmax((model.predict(img_array))[0])
    act_map_array = Model(inputs=model.input, outputs=model.get_layer(layer_name).output).predict(img_array)
    
    input_shape = model.layers[0].output_shape[1:]  # get input shape
    # 1. upsampled to original input size
    act_map_resized_list = [cv2.resize(act_map_array[0,:,:,k], input_shape[:2], interpolation=cv2.INTER_LINEAR) for k in range(act_map_array.shape[3])]
    # 2. normalize the raw activation value in each activation map into [0, 1]
    act_map_normalized_list = []
    for act_map_resized in act_map_resized_list:
        if np.max(act_map_resized) - np.min(act_map_resized) != 0.:
            act_map_normalized = act_map_resized / (np.max(act_map_resized) - np.min(act_map_resized))
        else:
            act_map_normalized = act_map_resized
        act_map_normalized_list.append(act_map_normalized)


    # 3. project highlighted area in the activation map to original input space by multiplying the normalized activation map
    masked_input_list = []
    for act_map_normalized in act_map_normalized_list:
        masked_input = np.copy(img_array)
        for k in range(3):
            masked_input[0,:,:,k] *= act_map_normalized
        masked_input_list.append(masked_input)
    masked_input_array = np.concatenate(masked_input_list, axis=0)

    # 4. feed masked inputs into CNN model and softmax
    pred_from_masked_input_array = softmax(model.predict(masked_input_array)[0])
    # 5. define weight as the score of target class
    weights = pred_from_masked_input_array[:,cls]
    # 6. get final class discriminative localization map as linear weighted combination of all activation maps
    cam = np.dot(act_map_array[0,:,:,:], weights)
    cam = np.maximum(0., cam)  # Passing through ReLU
    cam /= np.max(cam)  # scale 0 to 1.0
    
    
    return img_array, cam, act_map_array

def softmax(x):
    f = np.exp(x)/np.sum(np.exp(x), axis = 1, keepdims = True)
    return f

def vae_loss(inputs, z_decoded,z_mean, z_log_var):
    x = K.flatten(inputs)
    z_decoded = K.flatten(z_decoded)
    xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
    kl_loss = -5e-4 * K.mean(
        1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return K.mean(xent_loss + kl_loss)

def get_colorpalette(colorpalette, n_colors):
    palette = sns.color_palette(
        colorpalette, n_colors)
    rgb = ['rgb({},{},{})'.format(*[x*256 for x in rgb])
           for rgb in palette]
    return rgb


def plot_3d(X,Y, color_num = 7):
    colors = get_colorpalette("gist_ncar", color_num)
    traces = []
    for num, name in enumerate(np.unique(Y)):
        trace = go.Scatter3d(
            x = X[Y == name ,0],
            y = X[Y == name ,1],
            z = X[Y == name ,2],
            mode = 'markers',
            name = name,
            marker = dict(
                sizemode = 'diameter',
                opacity = 1.0,
                size = 3,
                line=dict(width=1,
                                            color='black'),
                color = colors[num],
            ),
        )
        traces.append(trace)
    
    plot_data = list(traces)
    layout = dict(height=700, width= 700, showlegend = True) 
    fig = dict(data =plot_data, layout=layout)
    offline.iplot(fig)    

def plot_2d(XxX,Y, color_num = 7):
    traces = []
    colors = get_colorpalette("gist_ncar", color_num)
    for num, name in enumerate(np.unique(Y)):
        trace = go.Scatter(
            x = XxX[Y == name ,0],
            y = XxX[Y == name ,1],
            mode = 'markers',
            name = name,
            marker = dict(
                sizemode = 'diameter',
                opacity = 1.0,
                size = 6,
                color = colors[num],
            ),
        )
        traces.append(trace)

    plot_data = list(traces)
    layout = dict(height=600, width= 600, showlegend = True)

    fig = go.Figure(data =plot_data, layout=layout)

    offline.iplot(fig)    
    
    

def plot_latent_and_PCAplane(x_predict, Y, zi):
    from sklearn.decomposition import PCA
    pca = PCA()
    feature = pca.fit(x_predict)
    feature = pca.transform(x_predict)

    points = np.concatenate([np.mean(x_predict, axis = 0)[:,np.newaxis],np.mean(x_predict, axis = 0)[:,np.newaxis] +  pca.components_[0][:,np.newaxis]],axis = 1)
    points2 = np.concatenate([np.mean(x_predict, axis = 0)[:,np.newaxis],np.mean(x_predict, axis = 0)[:,np.newaxis] +  pca.components_[1][:,np.newaxis]],axis = 1)
    points3 = np.concatenate([np.mean(x_predict, axis = 0)[:,np.newaxis],np.mean(x_predict, axis = 0)[:,np.newaxis] +  pca.components_[2][:,np.newaxis]],axis = 1)



    xs = points[0]
    ys = points[1]
    zs = points[2]

    traces = []



    colors = get_colorpalette("gist_ncar", 7)
    traces = []
    for num, name in enumerate(np.unique(Y)):
        trace = go.Scatter3d(
            x = x_predict[Y == name ,0],
            y = x_predict[Y == name ,1],
            z = x_predict[Y == name ,2],
            mode = 'markers',
            name = name,
            marker = dict(
                sizemode = 'diameter',
                opacity = 1.0,
                size = 3,
                line=dict(width=1,
                                                color='black'),
                color = colors[num],
            ),
        )
        traces.append(trace)





    # traceを作成
    trace2 = go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        name = 'PC1',
        mode='lines',
        line=dict(
            color='rgb(100,100,200)',
            width=20
        )
    )

    xs = points2[0]
    ys = points2[1]
    zs = points2[2]

    trace3 = go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode='lines',
        name = 'PC2',
        line=dict(
            color='rgb(200,100,200)',
            width=20,
        )
    )


    xs = points3[0]
    ys = points3[1]
    zs = points3[2]

    trace4 = go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        name = 'PC3',
        mode='lines',
        line=dict(
            color='rgb(260,100,200)',
            width=20,
        )
    )


    surf_ = np.concatenate([np.mean(x_predict, axis = 0)[:,np.newaxis]  + 1 *pca.components_[0][:,np.newaxis] + 1 *pca.components_[1][:,np.newaxis]+zi * pca.components_[2][:,np.newaxis] ,
                    np.mean(x_predict, axis = 0)[:,np.newaxis]  + 1 *pca.components_[0][:,np.newaxis] + -1 *pca.components_[1][:,np.newaxis]+zi * pca.components_[2][:,np.newaxis] ,
                    np.mean(x_predict, axis = 0)[:,np.newaxis]  + -1 *pca.components_[0][:,np.newaxis] + 1 *pca.components_[1][:,np.newaxis]+zi * pca.components_[2][:,np.newaxis] ,
                    np.mean(x_predict, axis = 0)[:,np.newaxis]  + -1 *pca.components_[0][:,np.newaxis] + -1 *pca.components_[1][:,np.newaxis]+zi * pca.components_[2][:,np.newaxis] 
                   ],axis  = 1)

    surf_ = np.concatenate([np.mean(x_predict, axis = 0)[:,np.newaxis]  + 10 *pca.components_[0][:,np.newaxis] + 10 *pca.components_[1][:,np.newaxis]+zi * pca.components_[2][:,np.newaxis] ,
                    np.mean(x_predict, axis = 0)[:,np.newaxis]  + 10 *pca.components_[0][:,np.newaxis] + -10 *pca.components_[1][:,np.newaxis]+zi * pca.components_[2][:,np.newaxis] ,
                    np.mean(x_predict, axis = 0)[:,np.newaxis]  + -10 *pca.components_[0][:,np.newaxis] + 10 *pca.components_[1][:,np.newaxis]+zi * pca.components_[2][:,np.newaxis] ,
                    np.mean(x_predict, axis = 0)[:,np.newaxis]  + -10 *pca.components_[0][:,np.newaxis] + -10 *pca.components_[1][:,np.newaxis]+zi * pca.components_[2][:,np.newaxis] 
                   ],axis  = 1)


    surface = go.Mesh3d(
            x = surf_[0],
            y = surf_[1],
            z = surf_[2],
            color='rgb(400,300,300)',
            name = 'PC Plane',
            opacity = 0.7

    )



    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        ),
        scene=dict(aspectmode='cube'),
    )
    fig = go.Figure(data=[traces[0] ,traces[1],traces[2],traces[3],traces[4],traces[5],traces[6],trace2, trace3,trace4, surface], layout=layout)
    offline.iplot(fig)    
    
def reconst_grid_images(size,x_predict, decoder, digit_size = 128,THR = 0.8, n = 7, zi = 0):
    """
    digit_size = 128  the size of rectangle (default)
    THE = 0.8 threshold value of image
    n = 7 how many figures reconstrcuted
    """
    start = -size
    end = size
    interval = (end - start)/(n-1) * 0.5
    grid_x = np.linspace(start + interval,end -interval,n-1)
    grid_y = np.linspace(end- interval, start+interval,n-1) 
    
    from sklearn.decomposition import PCA
    pca = PCA()
    feature = pca.fit(x_predict)
    feature = pca.transform(x_predict)
    
    xyz = np.mean(x_predict, axis = 0)[:,np.newaxis]+ zi * pca.components_[2][:,np.newaxis] 


    d = xyz[2]+xyz[0]*pca.components_[2][0]/pca.components_[2][2]+xyz[1]*pca.components_[2][1]/pca.components_[2][2], 
    a = pca.components_[2][0]/pca.components_[2][2]
    b = pca.components_[2][1]/pca.components_[2][2]


    plt.figure(figsize=(20,20))
    figure = np.zeros((digit_size * (n-1), digit_size * (n-1)))

    for j, yi in enumerate(grid_y):
            for k, xi in enumerate(grid_x):
                Z_sample = np.array([xi, yi, zi])
                z_sample = pca.inverse_transform(Z_sample)
                digit = decoder.predict(z_sample[:,np.newaxis].T)[0,:,:,1]
                digit = np.where(digit > THR, 1 , 0)
                figure[j * digit_size: (j + 1) * digit_size,
                    k * digit_size: (k + 1) * digit_size] = digit
                #plt.imshow(figure)

    plt.xticks([])
    plt.yticks([])
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    figure[0,:] = 0
    figure[:,0] = 0
    for i in range(1,7):
        print(digit_size * i -1)
        figure[digit_size * i -1,:] = 0
        figure[:,digit_size*i-1] = 0
    plt.imshow(figure, cmap = 'gray')

