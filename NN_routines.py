import torch
import os
import time
import cv2
#from tqdm.notebook import tqdm
from tqdm import tqdm
import pickle 
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from imutils import paths
from sklearn.model_selection import train_test_split
import shutil

## --------------------------------------------  Building Networks  --------------------------------------------  
### general architecture (Block)
# ------  defining block module of which the U-Nets will be constructed  ------

class Block(nn.Module):                                                                                             
    def __init__(self,inChannels,outChannels,cl = [15,3]):                                  # allow varying input and output channels
        super().__init__()
        self.conv1 = nn.Conv2d(inChannels,outChannels,  cl[0],padding = 'same')     # big convolutional layer (15x15)
        self.conv2 = nn.Conv2d(outChannels,outChannels, cl[1],padding = 'same')     # small convolutional layer (3x3)
        self.Dropout = nn.Dropout(p = 0.1)                                      # dropout
        self.BN1 = nn.BatchNorm2d(outChannels,track_running_stats=True)        # first batchnorm      
        self.BN2 = nn.BatchNorm2d(outChannels,track_running_stats=True)        # second batchnorm
        self.relu  = nn.ReLU()                                                  # nonlinear = ReLU
    def forward(self,x): 
        return self.BN2(self.conv2(self.relu(self.BN1(self.conv1(self.Dropout(x))))))
    
### Macro-Net

# ----------------  building macro net  ----------------

class Macro_Net(nn.Module):
    def __init__(self,channels = (3,16,16,1),cl = [15,3]):                                 # channel sizes (4 because the first is only input and the last is only output)
        super().__init__()
        self.encBlocks = nn.ModuleList([Block(channels[i],channels[i+1],cl = cl) for i in range(len(channels) -1)])
        self.pool = nn.MaxPool2d(2)                                             # Adding Maxpool layer to shrink image size
        self.resize = transforms.Resize(10)
    def forward(self,x):
        for block in self.encBlocks:                                            # call the three block iteratively
            x = block(x)
            x = self.pool(x)                                                    # shrink the image after every block
        x = self.resize(x)  
        #print(x.shape,'turn over')
        return x
    
### U-Net

# ----------------  building encoder that consists of 3 blocks  ----------------

class Encoder(nn.Module):
    def __init__(self,channels = (3,16,32,64),cl = [15,3]):                                 # channel sizes (4 because the first is only input and the last is only output)
        super().__init__()
        self.encBlocks = nn.ModuleList([Block(channels[i],channels[i+1],cl = cl) for i in range(len(channels) -1)])
        self.pool = nn.MaxPool2d(2)                                             # Adding Maxpool layer to shrink image size
        
    def forward(self,x):
        blockOutputs = []
        for block in self.encBlocks:                                            # call the three block iteratively
            x = block(x)
            blockOutputs.append(x)
            x = self.pool(x)                                                    # shrink the image after every block   
        return blockOutputs
# ----------------  building decoder that consists of 2 blocks  ----------------

class Decoder(nn.Module):
    def __init__(self,channels = (64,32,16),cl = [15,3]):                                   # channel sizes (3 because the first is only input and the last is only output)
        super().__init__()
        self.channels = channels
        self.dec_blocks = nn.ModuleList([Block(channels[i],channels[i+1],cl = cl) for i in range(len(channels)-1)])
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(channels[i], channels[i + 1], 2, 2) for i in range(len(channels) - 1)])
                                                                                # ^ upward convolution for larger image size, and lower channel size 
    
    def forward(self, x, encFeatures):
        for i in range(len(self.channels)-1):                                   # call the upconvolutions iteratively
            x = self.upconvs[i](x)                                              # ''
            x = torch.cat([x,encFeatures[i]], dim  = 1)                         # concatenate the output with the comparable values from the encoder
            x = self.dec_blocks[i](x)                                           # decode to new channel size via the block routine 
        return x
# ---------------------------  completing the U-Net  ---------------------------

class UNet(nn.Module):
    def __init__(self, encChannels = (3,16,32,64),decChannels = (64,32,16),nbClasses = 1,cl = [15,3]):
        super().__init__()
        self.encoder = Encoder(encChannels)                                     # initialize with encoder
        self.decoder = Decoder(decChannels)                                     # and decoder
        
        self.head = nn.Conv2d(decChannels[-1],nbClasses,1)                      # define last convolution, that brings the shape to a 1 channel mask
                
    def forward(self,x):
        encFeatures = self.encoder(x)
        decFeatures = self.decoder(encFeatures[::-1][0],encFeatures[::-1][1:])
        
        return self.head(decFeatures)                                           # output: prediction for the mask
    
    
## ----------------------------------------  Dataloader routine  -----------------------------------------  
# ---------  class to create dataset that connects images and masks  ---------
def shaping_(x,thresh = 0.5,nump = False):
    if nump:
      p = x.copy()
      x[:,:] = 128
      x[p< thresh*255] = 0
      x[p>=(1-thresh)*255] = 255
    else:
      p = x.detach().clone()
      x[:,:] = 0.5
      x[p< thresh] = 0
      x[p>=1-thresh] = 1
    
    return x
    
def mask_cut(b, size = (10,10),thresh = 0.5,macro = True):
  if macro:
    BB = shaping_(cv2.resize(b, dsize=size, interpolation=cv2.INTER_AREA),thresh = thresh,nump = True)

    return BB
  else: return b

class SegmentationDataset(Dataset):                                            
    def __init__(self,imagePaths,maskPaths,transform,macro = False,thresh = 0.5):
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths        
        self.transforms = transform
        
        self.macro = macro
        self.resize = transforms.Resize(128)
        self.thresh = thresh

    def __len__(self):
        return len(self.imagePaths)
    def __getitem__(self,idx):
       
        image = cv2.imread(self.imagePaths[idx]) 
        mask =  cv2.imread(self.maskPaths[idx],0)                               # I decided to leave the input in a 3 channel format. image layers inclide lw mask and original img ->
                                                                                # however this leaves place for a further raise in complexity. As for now, a 1 channel input achieves the same result.
        if self.transforms is not None:                                         # this enables transformations on the images and masks (see below)                                       
            if self.macro: 
                image = self.resize(self.transforms(image))
                mask  = self.transforms(mask_cut(mask,thresh = self.thresh)) 
            else:     
                image = self.transforms(image)
                mask  = self.transforms(mask)
        return (image,mask)

    
# -----------------  split into training and validation data  ------------------

def splitter(Folder, split = 0.2):
  imagePaths = sorted(list(paths.list_images(Folder+'/training_data')))         # sort lists of images to be able to reproduce results
  maskPaths = sorted(list(paths.list_images(Folder+'/training_labels')))
  print(f'[INFO] total images: {len(imagePaths)}')                              # check length

  split = train_test_split(imagePaths,maskPaths,test_size=split,random_state=7)# fraction into 80% training and 20% validation (split = 0.2). For testing an other dataset was created 

  (trainImages,validImages) = split[:2]                                           
  (trainMasks, validMasks)  = split[2:]
  return [trainImages,trainMasks,validImages,validMasks]

# --------------  create Dataloaders for training and validation  --------------

def Dataloading(Folder,batch_size = 32,macro = False,thresh = 0.5,DEVICE = 'cpu'):
  Train_Valid = splitter(Folder)
  Transforms = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])
                                                                                # ^ transform images to torch tensors of PIL images to get ready for the model
  trainDS = SegmentationDataset(Train_Valid[0],Train_Valid[1],Transforms,macro = macro,thresh = thresh)       # combine images and masks and apply transformation 
  validDS = SegmentationDataset(Train_Valid[2],Train_Valid[3],Transforms,macro = macro,thresh = thresh)
  
  print(f'[INFO] found {len(trainDS)} examples in the training set')            # check distribution
  print(f'[INFO] found {len(validDS)} examples in the validation set')

  pin_memory = True if DEVICE == 'cuda' else False                                
  trainLoader = DataLoader(trainDS,shuffle = True, batch_size = batch_size, pin_memory= pin_memory,num_workers=os.cpu_count())
  validLoader = DataLoader(validDS,shuffle = False, batch_size = batch_size, pin_memory= pin_memory,num_workers=os.cpu_count())
  return trainLoader,validLoader


## -----------------------------  Training and visualization routines  ---------------------------  




### Training

#------------------------  routine to train the u-net  -----------------------

def Network_Training(trainLoader,validLoader,n_epochs,learning_rate,title = None,load = None,macro = False,CL = [15,3], DEVICE = 'cpu'):
  if macro: net = Macro_Net(cl = CL).to(DEVICE)
  else: net = UNet(cl = CL).to(DEVICE)                                                      # initialize the network
  if load: net.load_state_dict(load)                                           # load a state if wanted
  lossFunc = nn.BCEWithLogitsLoss()                                             # Binary Cross Entropy as loss function
  optimizer  = torch.optim.Adam(net.parameters(),lr = learning_rate)            # Adam optimizer
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma = 0.99)    # Updating the loss function (exponential decrease to base gamma = 0.99)

  startTime = time.time()               
  trainSteps = len(trainLoader.dataset)/ trainLoader.batch_size                 # make training and test loss comparable
  validSteps  = len(validLoader.dataset) / validLoader.batch_size             
  H = {'train_loss':[],'valid_loss':[]}                                         # track loss
  saving = 'Model_params/'                                                      # directory for saving

  print('[INFO] training the network...')

  for e in tqdm(range(n_epochs)):                                               # training the network over n_epochs
      net.train()
      epochTrainLoss = 0
      epochValidLoss  = 0
      for (i,(x,y)) in enumerate(trainLoader):                                  # training
          (x,y) = (x.to(DEVICE),y.to(DEVICE))
          pred = net(x)
          
          #print(pred.shape,y.shape) 
          loss = lossFunc(pred,y)         
          epochTrainLoss += loss
          optimizer.zero_grad()            
          loss.backward()                                                       # network backward
          optimizer.step()                                                      # network optimization
      scheduler.step()

      with torch.no_grad():                                                     # validation / evaluation
          net.eval()
          for (x,y) in validLoader:
              (x,y) = (x.to(DEVICE),y.to(DEVICE))
              pred = net(x)
              epochValidLoss += lossFunc(pred,y)
      avgTrainLoss = epochTrainLoss/ trainSteps                                 # averaging loss over sample sizes
      avgValidLoss  = epochValidLoss / validSteps
      
      if e in [0,4,9]:                                                          # track (and save) the model states after different epochs
        temp_state = deepcopy(net.state_dict())
        if title: 
          if macro: torch.save(temp_state,saving+title+'_macro_epoch_'+str(e+1)+'.pt')
          else: torch.save(temp_state,saving+title+'_epoch_'+str(e+1)+'.pt')

      H['train_loss'].append(avgTrainLoss.cpu().detach().numpy())               # track loss
      H['valid_loss'].append(avgValidLoss.cpu().detach().numpy())            
      if avgValidLoss.cpu().detach().numpy() == min(H['valid_loss']): 
        best_state = deepcopy(net.state_dict())                                # always track best state

      #print("[INFO] EPOCH: {}/{}".format(e + 1, n_epochs))
      #print("       Train loss: {:.4f}, Validation loss: {:.4f}".format(avgTrainLoss, avgValidLoss))

  if title: 
    if macro: torch.save(best_state,saving+title+'_macro_best.pt')
    else: torch.save(best_state,saving+title+'_best.pt')

  print('\n[INFO] total time taken to train the model: {:.2f}s'.format(time.time()-startTime))
  return H, best_state

#---------------------------  show loss propagation  ---------------------------

def loss_propagation(H,title = None):
  plt.figure()
  plt.plot(H["train_loss"], label="training loss")
  plt.plot(H["valid_loss"], label="validation loss")
  plt.title("Training Loss on Dataset")
  plt.xlabel("Epoch ")
  plt.ylabel("Loss")
  plt.legend(loc="lower left")
  if title: plt.savefig('Model_params/loss_plot_'+title+'.png')
  else: plt.show()
    
    
### Visualization 
def prediction(img_path,model,model_state, macro = False,compare = False,thresh = 0.5):
  #if macro: 
  #  model = Macro_Net(cl = conv_layers[0])
  #else: 
  #  model = UNet(cl = conv_layers[1])
  #model.load_state_dict(model_state)                                            # load state
  #model.eval()                                                                  # set model to evaluate
  if macro: Transforms = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor(),transforms.Resize(128,interpolation = transforms.InterpolationMode.NEAREST)])
  else:     Transforms = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])

  with torch.no_grad():                                                         
    image = cv2.imread(img_path)                                                    # read image
    Timage = Transforms(image)                                            # transform image for the network  

    Pmask = model(Timage[None,:])                                            # perform mask prediction
    Pmask = Pmask.squeeze()   
    Pmask = shaping_(Pmask,thresh = thresh)
    if compare:
      mask_path = img_path[:img_path.find('img_train')-6]+'_labels/label_train'+img_path[img_path.find('img_train')+9:]
      mask = mask_cut(cv2.imread(mask_path,0),macro = macro,thresh = thresh)
      Trafo = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])
      Tmask = Trafo(mask).squeeze()     
                                               # transform mask for the loss evaluation
      lossFunc = nn.BCEWithLogitsLoss()
      loss = lossFunc(Pmask,Tmask)
    else: Tmask,loss = None,None
  return Pmask, Tmask, loss


def intensity_field(pix):
    a = np.zeros((pix,pix),dtype = 'uint8')
    for i in range(pix):
        for j in range(pix):
            a[i,j] = np.exp(-0.5*((i-(pix-1)/2)**2+(j-(pix-1)/2)**2)/(pix/4)**2)*255
    return a

def big_picture(model_state,name,thresh = 0.5,conv_layers = [[15,3],[15,3]],macro = True,compare = True,show = True,out = False,folder = None,overwrite_lw = True):
  if folder: Folder = folder
  else: Folder = name
  if macro:
    n,m,pix,mpix = 5,8,640,10
    Folder +='/macro'
    model = Macro_Net(cl = conv_layers[0])
  else:
    n,m,pix,mpix =25,40,128,128
    Folder +='/micro'
    model = UNet(cl = conv_layers[1])
    
  model.load_state_dict(model_state)                                            # load state
  model.eval()                                                                  # set model to evaluate
    
  orig = np.zeros((m*pix,n*pix,3),dtype = 'uint8')
  Mask = np.zeros((m*mpix,n*mpix),dtype = 'uint8')
  P_Mask = np.zeros((m*mpix,n*mpix,2),dtype = 'uint8')
  Weights = P_Mask.copy()
  Final_Mask = Mask.copy()


  for h in tqdm(range(m)):
    for i in range(n):
      orig[h*pix:(h+1)*pix,i*pix:(i+1)*pix] = cv2.imread(Folder+'/training_data/img_train_'+str(name)+'_'+str(h).rjust(2, '0')+str(i).rjust(2, '0')+'.png')
      pmask,tmask,loss = prediction(Folder+'/training_data/img_train_'+str(name)+'_'+str(h).rjust(2, '0')+str(i).rjust(2, '0')+'.png',
                                    model,model_state, macro = macro,compare = compare,thresh = thresh)

      if compare: Mask[h*mpix:(h+1)*mpix,i*mpix:(i+1)*mpix]   = tmask.detach().numpy()*255
      P_Mask[h*mpix:(h+1)*mpix,i*mpix:(i+1)*mpix,0] = pmask.detach().numpy()*255
      Weights[h*mpix:(h+1)*mpix,i*mpix:(i+1)*mpix,0] = intensity_field(mpix)

      if h<m-1 and i<n-1:
        pmask,tmask,loss = prediction(Folder+'/training_data/img_train_'+str(name)+'_'+str(h).rjust(2, '0')+str(i).rjust(2, '0')+'_s.png',
                                      model,model_state, macro = macro,compare = compare,thresh = thresh)

        P_Mask[int((h+0.5)*mpix):int((h+1.5)*mpix),int((i+0.5)*mpix):int((i+1.5)*mpix),1] = pmask.detach().numpy()*255
        Weights[int((h+0.5)*mpix):int((h+1.5)*mpix),int((i+0.5)*mpix):int((i+1.5)*mpix),1] = intensity_field(mpix)

  mini_lw = shaping_(cv2.resize(orig[:,:,1], dsize=(mpix*n,mpix*m), interpolation=cv2.INTER_AREA),thresh =0.4,nump = True)
  Final_Mask= np.array(np.average(P_Mask,weights = Weights,axis = 2),dtype = 'uint8')
  if macro == False:   Final_Mask = shaping_(Final_Mask,thresh = 0.5,nump = True)
  else:                Final_Mask = shaping_(Final_Mask,thresh = thresh,nump = True)

  if overwrite_lw:
    Final_Mask[mini_lw==255] = 255
    Final_Mask[mini_lw==0] = 0
  orig[:,:,2] = cv2.resize(Final_Mask, dsize=(pix*n,pix*m), interpolation=cv2.INTER_NEAREST_EXACT)
  lossFunc = nn.BCEWithLogitsLoss()
  n = 5
  if compare:
    Transforms = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])
    M_Loss = lossFunc(Transforms(Final_Mask),Transforms(Mask))
    perc_error = len(np.where(Mask!=Final_Mask)[0])/len(np.where(mini_lw == 128)[0])*100
  if show:
    fig, ax = plt.subplots(1,n,figsize = (6*n,12))
    ax[0].imshow(orig[:,:,0],'gray')
    ax[0].set_title('original radar capture (input)',fontsize = 22)
    ax[1].imshow(orig[:,:,1],'gray')
    ax[1].set_title('land/water mask (input)',fontsize = 22)
    ax[2].imshow(orig[:,:,2],'gray')
    ax[2].set_title('prediction (output)',fontsize = 22)
    if compare:
      ax[3].imshow(Mask,'gray')
      ax[3].set_title('mask (expected output)',fontsize = 22)
      ax[4].imshow(np.array(Mask,dtype = 'i4')-np.array(Final_Mask,dtype = 'i4'),'coolwarm')
      ax[4].set_title('loss %1.2f'%M_Loss+' %1.1f percent of pixels do not match'%perc_error,fontsize = 22)
  if out:   return orig

def reshape(b,pix):      
    im = np.zeros(((len(b)//pix+1)*pix,(len(b[0])//pix+1)*pix,3),dtype = b.dtype)
    im[:len(b),:len(b[0])] = b
    return im

def Augmentation(A):
    A_r = cv2.rotate(A,cv2.ROTATE_90_CLOCKWISE)
    A_u = cv2.rotate(A,cv2.ROTATE_180)
    A_l = cv2.rotate(A,cv2.ROTATE_90_COUNTERCLOCKWISE)
    return A_r,A_u,A_l

def write_images(img,mask,folder,name,i,j,shift,aug = False):
    cv2.imwrite(folder+'/training_data/img_train_'+str(name)+'_'+str(i).rjust(2, '0')+str(j).rjust(2, '0')+str(shift)+'.png', img)
    if mask is not None:    cv2.imwrite(folder+'/training_labels/label_train_'+str(name)+'_'+str(i).rjust(2, '0')+str(j).rjust(2, '0')+str(shift)+'.png', mask) 
    if aug:
        B_r,B_u,B_l = Augmentation(img)
        cv2.imwrite(folder+'/training_data/img_train_'+str(name)+'_'+str(i).rjust(2, '0')+str(j).rjust(2, '0')+'_90'+str(shift)+'.png', B_r)
        cv2.imwrite(folder+'/training_data/img_train_'+str(name)+'_'+str(i).rjust(2, '0')+str(j).rjust(2, '0')+'_180'+str(shift)+'.png', B_u)
        cv2.imwrite(folder+'/training_data/img_train_'+str(name)+'_'+str(i).rjust(2, '0')+str(j).rjust(2, '0')+'_270'+str(shift)+'.png', B_l)
        if mask is not None:
            A_r,A_u,A_l = Augmentation(mask)
            cv2.imwrite(folder+'/training_labels/label_train_'+str(name)+'_'+str(i).rjust(2, '0')+str(j).rjust(2, '0')+'_90'+str(shift)+'.png', A_r)                    
            cv2.imwrite(folder+'/training_labels/label_train_'+str(name)+'_'+str(i).rjust(2, '0')+str(j).rjust(2, '0')+'_180'+str(shift)+'.png', A_u)                    
            cv2.imwrite(folder+'/training_labels/label_train_'+str(name)+'_'+str(i).rjust(2, '0')+str(j).rjust(2, '0')+'_270'+str(shift)+'.png', A_l)     
            
def image_process(image,mask, macro = True,lw = 'Supply/lw_mask.png'):
    a = None 
    if macro:
        b = reshape(image,640)    
        if mask is not None: a = reshape(mask,640)
        b[:,:,1] = reshape(cv2.imread(lw),640)[:,:,0]
        b[:,:,2] = reshape(cv2.imread(lw),640)[:,:,0]
        return b , a
    else: print('error, not applicable on micro')

def overhead_write(image,mask,pix,folder,name,aug):
  A = None
  for i in tqdm(np.arange(0,image.shape[0]//pix)):
        for j in np.arange(0,image.shape[1]//pix):
                if mask is not None: A = mask[i*pix:(i+1)*pix,j*pix:(j+1)*pix]
                B = image[i*pix:(i+1)*pix,j*pix:(j+1)*pix]
                if B[:,:,0].shape == (pix,pix): write_images(B,A,folder,name,i,j,shift = '',aug = aug)
                    
                if mask is not None: A = mask[int(pix*(i+1/2)):int(pix*(i+3/2)),int(pix*(j+1/2)):int(pix*(j+3/2))]
                B = image[int(pix*(i+1/2)):int(pix*(i+3/2)),int(pix*(j+1/2)):int(pix*(j+3/2))]
                if B[:,:,0].shape == (pix,pix): write_images(B,A,folder,name,i,j,shift = '_s',aug = aug)
 

def dataset_produce(image,mask,name,macro = True,aug = True):
    if macro: 
        if name not in os.listdir('../NN_water_lines'):
            os.mkdir(name)
        if 'macro' not in os.listdir(name):
            os.mkdir(os.path.join(str(name),'macro'))
            os.mkdir(name+'/macro/training_labels')
            os.mkdir(name+'/macro/training_data')
        folder = str(name)+'/macro'
        pix = 640   
    else:
        if name not in os.listdir('../NN_water_lines'):
            os.mkdir(name)
        if 'micro' not in os.listdir(name):
            os.mkdir(name+'/micro')
            os.mkdir(name+'/micro/training_labels')
            os.mkdir(name+'/micro/training_data')
        folder = str(name)+'/micro'
        pix = 128
    overhead_write(image,mask,pix,folder,name,aug)
    
    
def predict_img(name,model_state,lw_path='Supply/new_lw_mask.png',show = True,save = False,conv_layers = [[15,3],[15,3]],only_macro = False):

  img_path = 'Image_storage/image_'+name+'.png'
  mask_path = 'Image_storage/mask_'+name+'.png'

  if not only_macro: micro_model_state = torch.load('Model_params/'+model_state+'_best.pt',map_location=torch.device('cpu'))                          # load model state if no training 
  macro_model_state = torch.load('Model_params/'+model_state+'_macro_best.pt',map_location=torch.device('cpu'))                          # load model state if no training 

  startTime = time.time()               
  print('load image...')
  o_img = cv2.imread(img_path)
  if os.path.exists(mask_path):
    print('load mask...')
    o_mask = cv2.imread(mask_path)  
    compare = True  
  else: o_mask,compare = None,False
  o_lw = cv2.imread(lw_path,0)
  print('create macro dataset...')
  img,mask = image_process(o_img,o_mask,lw = lw_path)
  dataset_produce(img,mask,name,aug = False)
  if not only_macro: 
    print('transfer macro prediction...')
    macro_img = big_picture(macro_model_state,name,macro = True,thresh = 0.15,conv_layers = conv_layers,compare = compare,show = False,out = True)
    macro_predict = macro_img[:len(o_img),:len(o_img[0]),2]
    print('create micro dataset...')
    dataset_produce(macro_img,mask,name,macro = False,aug = False)
    print('create raw prediction...')
    micro_img = big_picture(micro_model_state,name,macro = False,thresh = 0.5,conv_layers = conv_layers,compare = compare,show = False,out = True) 
    print('floodfill...')
    raw_predict = micro_img[:len(o_img),:len(o_img[0]),2]
    cv2.imwrite(name+'/raw_prediction_'+name+'.png',raw_predict)
    if save: cv2.imwrite('Save/raw_prediction_'+name+'.png',raw_predict)

    ff_predict = cv2.imread(name+'/raw_prediction_'+name+'.png',0)
    cv2.floodFill(ff_predict,None,(0,len(ff_predict[0])-1),128)
    ff_predict[ff_predict != 128] = 255
    ff_predict[ff_predict == 128] = 0
    cv2.imwrite(name+'/ff_prediction_'+name+'.png',ff_predict)
    if save: cv2.imwrite('Save/ff_prediction_'+name+'.png',ff_predict)
      
    if show:      
      n = 5
      if compare:  
        e_p = len(np.where(o_mask[:,:,0] != raw_predict)[0])/len(np.where(o_lw==128)[0])*100
        e_ff = len(np.where(o_mask[:,:,0] != ff_predict)[0])/len(np.where(o_lw==128)[0])*100
        n+=1

      fig, ax = plt.subplots(1,n,figsize = (30,12))
      titlefont = 25
      ax[0].imshow(o_img)
      ax[0].set_title('SAR '+name,fontsize = titlefont,fontweight="bold")
      ax[1].imshow(micro_img[:len(o_img),:len(o_img[0]),1],'gray')
      ax[1].set_title('land-water mask',fontsize = titlefont,fontweight="bold")
      ax[n-3].imshow(macro_predict,'plasma')
      ax[n-3].set_title('macro prediction',fontsize = titlefont,fontweight="bold")
      ax[n-2].imshow(raw_predict,'plasma')
      ax[n-2].set_title('raw prediction',fontsize = titlefont,fontweight="bold")
      ax[n-1].imshow(ff_predict,'plasma')
      ax[n-1].set_title('floodfill prediction',fontsize = titlefont,fontweight="bold")
      if compare: 
        ax[2].imshow(o_mask)
        ax[2].set_title('original mask',fontsize = titlefont)    
        ax[n-2].set_title('prediction, %1.1f pc error'%e_p,fontsize = titlefont,fontweight="bold")
        ax[n-1].set_title('floodfill, %1.1f pc error'%e_ff,fontsize = titlefont,fontweight="bold")
      for i in range(n):
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])
      #plt.tight_layout()
      plt.savefig('Save/fig_'+name+'_'+model_state+'.png')

  else:
    macro_img = big_picture(macro_model_state,name,macro = True,thresh = 0.15,conv_layers = conv_layers,compare = compare,show = True,out = True)

  print('total time taken for prediction: {:.2f}s'.format(time.time()-startTime))







def training_dataset(Folder,aug = False,create_ds= True, macro_training = True,macro_epochs = 40,
                     title = 'ma1', micro_training = True,micro_epochs = 20,
                     conv_layers = [[15,3],[15,3]],lw_path = 'Supply/new_lw_mask.png'):
  names = []
  DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'                         # connecting to GPU (cuda)
  print('device: '+DEVICE)

  if create_ds:
    if 'Training' not in os.listdir('../NN_water_lines'):
        os.mkdir('Training')
        os.mkdir('Training/macro')
        os.mkdir('Training/macro/training_labels')
        os.mkdir('Training/macro/training_data')
        os.mkdir('Training/micro')
        os.mkdir('Training/micro/training_labels')
        os.mkdir('Training/micro/training_data')
    folder = 'Training/macro'
    for file in sorted(os.listdir(Folder)):
      if file[:5] == 'image' and os.path.exists(Folder+'mask'+file[5:]):
        image,mask = image_process(cv2.imread(Folder+file),cv2.imread(Folder+'mask'+file[5:]),lw = lw_path)
        name = file[-12:-4]
        names.append(name)
        print(name+'   in process')
        overhead_write(image,mask,640,folder,name,aug)
  else:
    for file in sorted(os.listdir(Folder)):
      if file[:5] == 'image' and os.path.exists(Folder+'mask'+file[5:]):
        names.append(file[-12:-4])
  if macro_training: 
    print('macro training started')
    tLoader, vLoader = Dataloading(folder,batch_size = 16, macro = True,thresh = 0.15)                                        # build dataloader
    H,macro_result = Network_Training(tLoader,vLoader,macro_epochs,0.001,CL = conv_layers[0],macro = True,title =title,DEVICE = DEVICE) # training execution: 20 epochs, learning rate = 0.001
    loss_propagation(H,title =title+'_macro')                                       # show loss propagation
  else:
    macro_result = torch.load('Model_params/'+title+'_macro_best.pt',map_location=torch.device('cpu'))                          # load model state if no training 
  print('transfer started')
  folder = 'Training/micro'
  for name in names:    
    print(name+'   in process')
    mp_img = big_picture(macro_result,name,macro = True,thresh = 0.15,conv_layers = conv_layers,show = False,out = True,folder = 'Training')
    image,mask = image_process(cv2.imread(Folder+'image_'+name+'.png'),cv2.imread(Folder+'mask_'+name+'.png'),lw = lw_path)
    overhead_write(image,mask,128,folder,name,aug)

  if micro_training: 
    print('micro training started')
    tLoader, vLoader = Dataloading(folder,macro = False,thresh = 0.5)                                        # build dataloader
    H,micro_result = Network_Training(tLoader,vLoader,micro_epochs,0.001,CL = conv_layers[1],macro = False,title =title,DEVICE = DEVICE) # training execution: 20 epochs, learning rate = 0.001
    loss_propagation(H,title =title+'_micro')                                       # 
  #else:
    #micro_result = torch.load('Model_params/'+title+'_best.pt',map_location=torch.device('cpu'))

    
def clear_routine(Save):
    basic = ['Model_params', 'NN_routine.ipynb','NN_routines.py','NN_execution.ipynb', 'README.md', '.git', 'Training_images', 'Test', 'Supply', 'Save', 'Image_storage'] 
    for directory in os.listdir('../NN_water_lines'): 
        if directory not in basic and directory not in Save:
            shutil.rmtree(directory,ignore_errors = True)
    for state in os.listdir('./Model_params'):
        if state[-10:-5] == 'epoch':
            os.remove('Model_params/'+state)
        elif state[-11:-6] == 'epoch':
            os.remove('Model_params/'+state)
        

            